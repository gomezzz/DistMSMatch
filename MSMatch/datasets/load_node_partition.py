from .SSL_Dataset import SSL_Dataset
from .data_utils import get_data_loader


def load_node_partition(comm, cfg):
    """Load a data partition for a given node.
    Data partitions are created if they do not exist.

    Args:
        comm (_type_): comm object to sync processes
        cfg (_type_): config file for data partitioning

    Returns:
        _type_: _description_
    """

    rank = comm.Get_rank()
    dset = SSL_Dataset(
        name=cfg.dataset,
        data_dir=None,
        num_labels=cfg.num_labels,
        seed=cfg.seed,
        alpha=cfg.alpha,
        nodes=cfg.nodes,
        node_indx=rank,
    )

    # load all the datasets for the given node
    lb_dset, ulb_dset, eval_dset = dset.get_ssl_dset(cfg.num_labels)
    cfg.num_channels = dset.num_channels
    cfg.num_classes = dset.num_classes

    loader_dict = {}
    dset_dict = {"train_lb": lb_dset, "train_ulb": ulb_dset, "eval": eval_dset}

    # dataloader for labeled data
    loader_dict["train_lb"] = get_data_loader(
        dset_dict["train_lb"],
        cfg.batch_size,
        data_sampler="RandomSampler",
        num_iters=cfg.num_train_iter,
        num_workers=1,
        distributed=False,
    )
    # dataloader for unlabeled data
    loader_dict["train_ulb"] = get_data_loader(
        dset_dict["train_ulb"],
        cfg.batch_size * cfg.uratio,
        data_sampler="RandomSampler",
        num_iters=cfg.num_train_iter,
        num_workers=4,
        distributed=False,
    )
    # dataloader for test data
    loader_dict["eval"] = get_data_loader(
        dset_dict["eval"], cfg.eval_batch_size, num_workers=1
    )

    return loader_dict, cfg
