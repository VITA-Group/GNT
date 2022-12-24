import os
import numpy as np
import imageio
import torch
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
import glob
from skimage.transform import resize


def parse_pose(path, num_views):
    cameras = np.load(path)

    intrinsics = []
    c2w_mats = []

    for i in range(num_views):
        # ShapeNet
        wmat_inv_key = "world_mat_inv_" + str(i)
        wmat_key = "world_mat_" + str(i)
        kmat_key = "camera_mat_" + str(i)
        if wmat_inv_key in cameras:
            c2w_mat = cameras[wmat_inv_key]
        else:
            w2c_mat = cameras[wmat_key]
            if w2c_mat.shape[0] == 3:
                w2c_mat = np.vstack((w2c_mat, np.array([0, 0, 0, 1])))
            c2w_mat = np.linalg.inv(w2c_mat)

        intrinsics.append(cameras[kmat_key])
        c2w_mats.append(c2w_mat)

    intrinsics = np.stack(intrinsics, 0)
    c2w_mats = np.stack(c2w_mats, 0)

    return intrinsics, c2w_mats


class NMRDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "data/nmr/")
        self.args = args
        if mode == "validation":
            mode = "val"
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop

        cats = [x for x in glob.glob(os.path.join(self.folder_path, "*")) if os.path.isdir(x)]
        list_prefix = "softras_"  # Train on all categories and eval on them

        if mode == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif mode == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif mode == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs

        self.intrinsics = []
        self.poses = []
        self.rgb_paths = []
        cntr = 0
        for _, path in self.all_objs:
            curr_paths = sorted(glob.glob(os.path.join(path, "image", "*")))
            self.rgb_paths.append(curr_paths)

            pose_path = os.path.join(path, "cameras.npz")
            intrinsics, c2w_mats = parse_pose(pose_path, len(curr_paths))

            self.poses.append(c2w_mats)
            self.intrinsics.append(intrinsics)
            print(f"{cntr} / {len(self.all_objs)}", end="\r")
            cntr += 1

        self.rgb_paths = np.array(self.rgb_paths)
        self.poses = np.stack(self.poses, 0)
        self.intrinsics = np.array(self.intrinsics)

        assert len(self.rgb_paths) == len(self.poses)

        # Default near/far plane depth
        # self.z_near = 1.2
        # self.z_far = 4.0
        self.z_near = 2.0
        self.z_far = 4.0

    def __len__(self):
        # return len(self.intrinsics) * 24
        return len(self.intrinsics)

    def __getitem__(self, indx):
        idx = indx
        # idx = indx // 24
        # render_idx = indx % 24
        train_poses = self.poses[idx]

        # render_idx = np.random.choice(len(train_poses), 1, replace=False)[0]
        render_idx = 8

        intrinsic = self.intrinsics[idx][render_idx].copy()

        rgb_path = self.rgb_paths[idx, render_idx]
        render_pose = train_poses[render_idx]

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views,
            tar_id=render_idx,
            angular_dist_method="vector",
        )
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        img = imageio.imread(rgb_path).astype(np.float32) / 255.0
        intrinsic[0, 0] *= img.shape[1] / 2.0
        intrinsic[1, 1] *= img.shape[0] / 2.0
        intrinsic[0, 2] = img.shape[1] / 2.0
        intrinsic[1, 2] = img.shape[0] / 2.0

        img_size = img.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsic.flatten(), render_pose.flatten())
        ).astype(np.float32)

        src_rgb_paths = [self.rgb_paths[idx][x] for x in nearest_pose_ids]
        src_c2w_mats = np.array([train_poses[x] for x in nearest_pose_ids])
        src_intrinsics = np.array(self.intrinsics[idx][nearest_pose_ids])

        src_intrinsics[..., 0, 0] *= img.shape[1] / 2.0
        src_intrinsics[..., 1, 1] *= img.shape[0] / 2.0
        src_intrinsics[..., 0, 2] = img.shape[1] / 2.0
        src_intrinsics[..., 1, 2] = img.shape[0] / 2.0

        src_rgbs = []
        src_cameras = []
        for i, rgb_path in enumerate(src_rgb_paths):
            rgb = imageio.imread(rgb_path).astype(np.float32) / 255.0
            src_rgbs.append(rgb)
            src_camera = np.concatenate(
                (list(img_size), src_intrinsics[i].flatten(), src_c2w_mats[i].flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)
        depth_range = np.array([self.z_near, self.z_far])
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        return {
            "rgb": torch.from_numpy(img),
            "camera": torch.from_numpy(camera),
            "rgb_path": self.rgb_paths[idx, render_idx],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
