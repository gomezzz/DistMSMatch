from termcolor import colored
from .SSL_Dataset import SSL_Dataset
from .data_utils import get_data_loader


def create_node_dataloaders(cfg):
    print(colored("Loading datasets", "red"))
    node_dls = []
    for node_indx in range(cfg.nodes):

        train_dset = SSL_Dataset(
            name=cfg.dataset,
            train=True,
            data_dir=None,
            seed=cfg.seed,
            alpha=cfg.alpha,
            nodes=cfg.nodes,
            node_indx=node_indx,
        )

        lb_dset, ulb_dset = train_dset.get_ssl_dset(cfg.num_labels)

        if node_indx == 0:
            cfg.num_classes = train_dset.num_classes
            cfg.num_channels = train_dset.num_channels

        _eval_dset = SSL_Dataset(
            name=cfg.dataset,
            train=False,
            data_dir=None,
            seed=cfg.seed,
            alpha=cfg.alpha,
            nodes=cfg.nodes,
            node_indx=node_indx,
        )
        eval_dset = _eval_dset.get_dset()

        loader_dict = {}
        dset_dict = {"train_lb": lb_dset, "train_ulb": ulb_dset, "eval": eval_dset}

        loader_dict["train_lb"] = get_data_loader(
            dset_dict["train_lb"],
            cfg.batch_size,
            data_sampler="RandomSampler",
            num_iters=cfg.num_train_iter,
            num_workers=1,
            distributed=False,
        )

        loader_dict["train_ulb"] = get_data_loader(
            dset_dict["train_ulb"],
            cfg.batch_size * cfg.uratio,
            data_sampler="RandomSampler",
            num_iters=cfg.num_train_iter,
            num_workers=4,
            distributed=False,
        )

        loader_dict["eval"] = get_data_loader(
            dset_dict["eval"], cfg.eval_batch_size, num_workers=1
        )
        node_dls.append(loader_dict)
        print(
            colored(
                "---------------------------------------------------------------------------------------------------------------------------------",
                "red",
            )
        )
    return node_dls, cfg
