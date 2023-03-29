def create_dir_str(cfg):
    """Creates a string from the arguments.

    Args:
        cfg (DotMap): config dictionary/dotmap

    Returns:
        _type_: _description_
    """
    # fmt: off
    dir_name = (
        cfg.dataset
        + "/FixMatch_arch"  + cfg.net
        + "_batch"          + str(cfg.batch_size)
        + "_confidence"     + str(cfg.p_cutoff)
        + "_lr"             + str(cfg.lr)
        + "_uratio"         + str(cfg.uratio)
        + "_wd"             + str(cfg.weight_decay)
        + "_wu"             + str(cfg.ulb_loss_ratio)
        + "_seed"           + str(cfg.seed)
        + "_numlabels"      + str(cfg.num_labels)
        + "_opt"            + str(cfg.opt)
        + "_nodes"          + str(cfg.nodes)
    )
    # fmt: on
    if cfg.pretrained:
        dir_name = dir_name + "_pretrained"
    return dir_name
