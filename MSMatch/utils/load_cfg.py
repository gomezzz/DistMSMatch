from .create_dir_string import create_dir_str
from dotmap import DotMap
import toml
import os

def load_cfg(cfg_path):
    try:
        with open(cfg_path) as cfg_file:
            cfg=DotMap(toml.load(cfg_file), _dynamic=False)
            cfg.bn_momentum = 1.0 - cfg.ema_m
            dir_name = create_dir_str(cfg)
            cfg.save_name = os.path.join(cfg.save_name_root, dir_name)
            cfg.save_path = os.path.join(cfg.save_dir, cfg.save_name)
            return cfg
    except:
        raise ValueError("Impossible to find \""+cfg_path+"\".")
        