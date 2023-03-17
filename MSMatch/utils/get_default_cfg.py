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
    cfg.dataset = "eurosat_ms"  # "eurosat_ms" # "eurosat_rgb"
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
    cfg.eval_batch_size = 64
    cfg.momentum = 0.9
    cfg.T = 0.5
    cfg.amp = False
    cfg.hard_label = True
    cfg.scale = 1
    cfg.alpha = 100
    cfg.nodes = 4
    cfg.planes = 1
    cfg.mode = "Swarm"  # "FL_ground", "FL_geostat", "Swarm"

    # PASEOS specific configuration
    cfg.start_time = "2023-Dec-17 14:42:42"
    cfg.constellation_altitude = 786 * 1000  # altitude above the Earth's ground [m]
    cfg.constellation_inclination = 98.62  # inclination of the orbit

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if cfg.mode == "Swarm":
        cfg.sim_path = cfg.save_dir + f"ISL/{timestr}"
    elif cfg.mode == "FL_ground":
        cfg.sim_path = cfg.save_dir + f"FL_ground/{timestr}"
    elif cfg.mode == "FL_geostat":
        cfg.sim_path = cfg.save_dir + f"FL_geostat/{timestr}"

    return cfg
