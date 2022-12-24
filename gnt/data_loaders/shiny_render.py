import os
import numpy as np
import imageio
import torch
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .shiny_data_utils import load_llff_data, batch_parse_llff_poses


class ShinyRenderDataset(Dataset):
    def __init__(self, args, scenes=(), **kwargs):
        self.folder_path = os.path.join(args.rootdir, "data/shiny/")
        self.num_source_views = args.num_source_views

        print("loading {} for rendering".format(scenes))

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.h = []
        self.w = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, intrinsic, i_test = load_llff_data(
                scene_path, load_imgs=False, factor=8, render_style="", split_train_val=0
            )
            if len(intrinsic) == 3:
                H, W, f = intrinsic
                cx = W / 2.0
                cy = H / 2.0
                fx = f
                fy = f
            else:
                H, W, fx, fy, cx, cy = intrinsic
            H = H / 8
            W = W / 8
            fx = fx / 8
            fy = fy / 8
            cx = cx / 8
            cy = cy / 8
            image_dir = os.path.join(scene_path, "images_8")
            # image_dir = os.path.join(scene_path, 'images')
            rgb_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
            _, c2w_mats = batch_parse_llff_poses(poses)
            _, render_c2w_mats = batch_parse_llff_poses(render_poses)

            intrinsics_ = np.array(
                [
                    [fx, 0, cx, 0],
                    [0, fy, cy, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ).astype(np.float32)
            intrinsics = intrinsics_[None, :, :].repeat(len(c2w_mats), axis=0)
            render_intrinsics = intrinsics_[None, :, :].repeat(len(render_c2w_mats), axis=0)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            i_test = [i_test]
            i_val = i_test
            i_train = np.array(
                [i for i in np.arange(len(rgb_files)) if (i not in i_test and i not in i_val)]
            )

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(render_intrinsics)
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics])
            self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)
            self.h.extend([int(H)] * num_render)
            self.w.extend([int(W)] * num_render)

    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )

        id_render = -1
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views,
            tar_id=id_render,
            angular_dist_method="dist",
        )

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {
            "camera": torch.from_numpy(camera),
            "rgb_path": "",
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
