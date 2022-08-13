import torch
from collections import OrderedDict
from gnt.render_ray import render_rays


def render_single_image(
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    render_stride=1,
    featmaps=None,
    ret_alpha=False,
    single_net=False,
):
    """
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param ret_alpha: if True, will return learned 'density' values inferred from the attention maps
    :param single_net: if True, will use single network, can be cued with both coarse and fine points
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    """

    all_ret = OrderedDict([("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())])

    N_rays = ray_batch["ray_o"].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ["camera", "depth_range", "src_rgbs", "src_cameras"]:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i : i + chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(
            chunk,
            model,
            featmaps,
            projector=projector,
            N_samples=N_samples,
            inv_uniform=inv_uniform,
            N_importance=N_importance,
            det=det,
            white_bkgd=white_bkgd,
            ret_alpha=ret_alpha,
            single_net=single_net,
        )

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret["outputs_coarse"]:
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k] = []

            if ret["outputs_fine"] is None:
                all_ret["outputs_fine"] = None
            else:
                for k in ret["outputs_fine"]:
                    if ret["outputs_fine"][k] is not None:
                        all_ret["outputs_fine"][k] = []

        for k in ret["outputs_coarse"]:
            if ret["outputs_coarse"][k] is not None:
                all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k].cpu())

        if ret["outputs_fine"] is not None:
            for k in ret["outputs_fine"]:
                if ret["outputs_fine"][k] is not None:
                    all_ret["outputs_fine"][k].append(ret["outputs_fine"][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret["outputs_coarse"]:
        if k == "random_sigma":
            continue
        tmp = torch.cat(all_ret["outputs_coarse"][k], dim=0).reshape(
            (rgb_strided.shape[0], rgb_strided.shape[1], -1)
        )
        all_ret["outputs_coarse"][k] = tmp.squeeze()

    # TODO: if invalid: replace with white
    # all_ret["outputs_coarse"]["rgb"][all_ret["outputs_coarse"]["mask"] == 0] = 1.0
    if all_ret["outputs_fine"] is not None:
        for k in all_ret["outputs_fine"]:
            if k == "random_sigma":
                continue
            tmp = torch.cat(all_ret["outputs_fine"][k], dim=0).reshape(
                (rgb_strided.shape[0], rgb_strided.shape[1], -1)
            )

            all_ret["outputs_fine"][k] = tmp.squeeze()

    return all_ret
