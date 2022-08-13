import torch
import os
from gnt.transformer_network import GNT
from gnt.feature_network import ResUNet


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class GNTModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))
        # create coarse GNT
        self.net_coarse = GNT(
            args,
            in_feat_ch=self.args.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
        ).to(device)
        # single_net - trains single network which can be used for both coarse and fine sampling
        if args.single_net:
            self.net_fine = None
        else:
            self.net_fine = GNT(
                args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.net_fine.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net_coarse": de_parallel(self.net_coarse).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
        }

        if self.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        # print(to_load["net_coarse"].keys())
        # exit()
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.net_coarse.load_state_dict(to_load["net_coarse"])
        self.feature_net.load_state_dict(to_load["feature_net"])

        if self.net_fine is not None and "net_fine" in to_load.keys():
            self.net_fine.load_state_dict(to_load["net_fine"])

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step
