from .create_dir_string import create_dir_str
from dotmap import DotMap
import toml
import os
from .print_cfg import print_cfg
from .get_default_cfg import get_default_cfg


def load_cfg(cfg_path):
    """Load configuration from a specific path. If path is None, default cfg is loaded.
    Args:
        cfg_path (str): path to the cfg file.
    Raises:
        ValueError: Impossible to find: path.
    Returns:
        dotmap: configuration.
    """
    try:
        if cfg_path is None:
            print("Using default configuration...")
            cfg = get_default_cfg()
        else:
            with open(cfg_path) as cfg_file:
                cfg = DotMap(toml.load(cfg_file), _dynamic=False)
                cfg.bn_momentum = 1.0 - cfg.ema_m
                dir_name = create_dir_str(cfg)
                cfg.save_path = os.path.join(cfg.save_dir, dir_name)
        #print_cfg(cfg)
        return cfg
    except:
        raise ValueError('Impossible to find "' + cfg_path + '".')
