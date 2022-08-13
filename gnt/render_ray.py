import torch
from collections import OrderedDict

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    """
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    """

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i : i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)  # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1.0 / near_depth  # [N_rays,]
        step = (1.0 / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]
        z_vals = 1.0 / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]
    return pts, z_vals


########################################################################################################################
# ray rendering of nerf
########################################################################################################################


def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    """
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    """
    rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]  # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1.0 - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T  # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - torch.sum(weights, dim=-1, keepdim=True))

    mask = (
        mask.float().sum(dim=1) > 8
    )  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

    ret = OrderedDict(
        [
            ("rgb", rgb_map),
            ("depth", depth_map),
            ("weights", weights),  # used for importance sampling of fine samples
            ("mask", mask),
            ("alpha", alpha),
            ("z_vals", z_vals),
        ]
    )

    return ret


def sample_fine_pts(inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals):
    if inv_uniform:
        inv_z_vals = 1.0 / z_vals
        inv_z_vals_mid = 0.5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        inv_z_vals = sample_pdf(
            bins=torch.flip(inv_z_vals_mid, dims=[1]),
            weights=torch.flip(weights, dims=[1]),
            N_samples=N_importance,
            det=det,
        )  # [N_rays, N_importance]
        z_samples = 1.0 / inv_z_vals
    else:
        # take mid-points of depth samples
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        z_samples = sample_pdf(
            bins=z_vals_mid, weights=weights, N_samples=N_importance, det=det
        )  # [N_rays, N_importance]

    z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

    # samples are sorted with increasing depth
    z_vals, _ = torch.sort(z_vals, dim=-1)
    N_total_samples = N_samples + N_importance

    viewdirs = ray_batch["ray_d"].unsqueeze(1).repeat(1, N_total_samples, 1)
    ray_o = ray_batch["ray_o"].unsqueeze(1).repeat(1, N_total_samples, 1)
    pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
    return pts, z_vals


def render_rays(
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    ret_alpha=False,
    single_net=True,
):
    """
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :param ret_alpha: if True, will return learned 'density' values inferred from the attention maps
    :param single_net: if True, will use single network, can be cued with both coarse and fine points
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    """

    ret = {"outputs_coarse": None, "outputs_fine": None}
    ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"]

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=ray_batch["depth_range"],
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det,
    )

    N_rays, N_samples = pts.shape[:2]
    rgb_feat, ray_diff, mask = projector.compute(
        pts,
        ray_batch["camera"],
        ray_batch["src_rgbs"],
        ray_batch["src_cameras"],
        featmaps=featmaps[0],
    )  # [N_rays, N_samples, N_views, x]
    # TODO: include pixel mask in ray transformer
    # pixel_mask = (
    #     mask[..., 0].sum(dim=2) > 1
    # )  # [N_rays, N_samples], should at least have 2 observations

    rgb = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d)
    if ret_alpha:
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        depth_map = torch.sum(weights * z_vals, dim=-1)
    else:
        weights = None
        depth_map = None
    ret["outputs_coarse"] = {"rgb": rgb, "weights": weights, "depth": depth_map}

    if N_importance > 0:
        # detach since we would like to decouple the coarse and fine networks
        weights = ret["outputs_coarse"]["weights"].clone().detach()  # [N_rays, N_samples]
        pts, z_vals = sample_fine_pts(
            inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals
        )

        rgb_feat_sampled, ray_diff, mask = projector.compute(
            pts,
            ray_batch["camera"],
            ray_batch["src_rgbs"],
            ray_batch["src_cameras"],
            featmaps=featmaps[1],
        )

        # TODO: Include pixel mask in ray transformer
        # pixel_mask = (
        #     mask[..., 0].sum(dim=2) > 1
        # )  # [N_rays, N_samples]. should at least have 2 observations

        if single_net:
            rgb = model.net_coarse(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        else:
            rgb = model.net_fine(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        depth_map = torch.sum(weights * z_vals, dim=-1)
        ret["outputs_fine"] = {"rgb": rgb, "weights": weights, "depth": depth_map}

    return ret
