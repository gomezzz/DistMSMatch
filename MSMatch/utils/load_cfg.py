from .create_dir_string import create_dir_str
from dotmap import DotMap
import toml
import os
from .print_cfg import print_cfg
from .get_default_cfg import get_default_cfg

def load_cfg(cfg_path, rank = 0):
    """Load configuration from a specific path. If path is None, default cfg is loaded.
    Args:
        cfg_path (str): path to the cfg file.
        rank (int): rank of the mpi process
    Raises:
        ValueError: Impossible to find: path.
    Returns:
        dotmap: configuration.
    """
    try:
        if cfg_path is None:
            print("Using default configuration...")
            cfg = get_default_cfg()
            dir_name = create_dir_str(cfg)
            cfg.sim_path = os.path.join(cfg.save_dir, cfg.mode, dir_name)
        else:
            with open(cfg_path) as cfg_file:
                cfg = DotMap(toml.load(cfg_file), _dynamic=False)
                dir_name = create_dir_str(cfg)
                cfg.sim_path = os.path.join(cfg.save_dir, cfg.mode, dir_name)
        if rank == 0:
            print_cfg(cfg)
        return cfg
    except:
        raise ValueError('Impossible to find "' + cfg_path + '".')
