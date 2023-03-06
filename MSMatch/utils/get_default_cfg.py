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
    cfg.uratio = 3
    cfg.weight_decay = 7.5e-4
    cfg.ulb_loss_ratio = 1.0
    cfg.seed = 42
    cfg.num_labels = 200
    cfg.opt = "SGD"
    cfg.pretrained = False
    cfg.save_dir = "./results/"
    cfg.ema_m = 0.99
    cfg.eval_batch_size = 512
    cfg.momentum = 0.9
    cfg.T = 0.5
    cfg.amp = False
    cfg.hard_label = True
    cfg.lb_epochs = 10
    cfg.scale = 1
    cfg.alpha = 100
    cfg.nodes = 8
    cfg.planes = 1
    cfg.time_multiplier = 1
    cfg.mode = "Swarm"# "FL_ground", "FL_geostat", "Swarm" 
    
    cfg.standby_period = 900  # how long to standby if necessary

    dir_name = create_dir_str(cfg)
    cfg.save_path = os.path.join(cfg.save_dir, dir_name)

    return cfg
