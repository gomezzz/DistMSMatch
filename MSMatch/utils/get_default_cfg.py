from dotmap import DotMap

import os

from .create_dir_string import create_dir_str


def get_default_cfg():
    """Returns the default configuration for MSMatch.

    Returns:
        DotMap: the default configuration
    """
    cfg = DotMap(_dynamic=False)
    cfg.dataset = "eurosat_rgb"
    cfg.net = "efficientnet-lite0"
    cfg.batch_size = 32
    cfg.p_cutoff = 0.95
    cfg.lr = 0.03
    cfg.uratio = 7
    cfg.weight_decay = 7.5e-4
    cfg.ulb_loss_ratio = 1.0
    cfg.seed = 42
    cfg.num_labels = 100
    cfg.opt = "SGD"
    cfg.pretrained = False
    cfg.save_dir = "../results/"
    cfg.ema_m = 0.999
    cfg.bn_momentum = 1.0 - cfg.ema_m
    cfg.eval_batch_size = 1024
    cfg.momentum = 0.9
    cfg.T = 0.5
    cfg.amp = False
    cfg.hard_label = True
    cfg.multiprocessing_distributed = False
    cfg.num_eval_iter = 250 # iterations in an epoch
    cfg.local_epochs = 5
    cfg.scale = 1
    cfg.training_rounds = 50
    cfg.alpha = 100
    cfg.nodes = 2
    cfg.thread_number = 2

    dir_name = create_dir_str(cfg)
    cfg.save_path = os.path.join(cfg.save_dir, dir_name)

    return cfg
