from dotmap import DotMap
import os
import time
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
    cfg.num_labels = 50
    cfg.num_train_iter = (100 * cfg.num_labels) // cfg.batch_size
    cfg.opt = "SGD"
    cfg.pretrained = True
    cfg.save_dir = "./results/"
    cfg.ema_m = 0.99
    cfg.eval_batch_size = 512
    cfg.momentum = 0.9
    cfg.T = 0.5
    cfg.amp = False
    cfg.hard_label = True
    cfg.scale = 1
    cfg.alpha = 100
    cfg.nodes = 8
    cfg.planes = 1
    cfg.time_multiplier = 1
    cfg.mode = "FL_ground"# "FL_ground", "FL_geostat", "Swarm" 
    
    cfg.standby_period = 900  # how long to standby if necessary
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if cfg.mode == "Swarm":
        cfg.sim_path = cfg.save_dir + f"ISL/{timestr}"
    elif cfg.mode == "FL_ground":
        cfg.sim_path = cfg.save_dir + f"FL_ground/{timestr}"
    elif cfg.mode == "FL_geostat":
        cfg.sim_path = cfg.save_dir + f"FL_geostat/{timestr}"
    
    return cfg
