from dotmap import DotMap
import numpy as np


def print_cfg(cfg: DotMap):
    """Prints the config in a more readable way.
    Args:
        cfg (DotMap): Config to print.
    """
    # Print the config three values per line
    idx = 0
    for key, value in cfg.items():
        if isinstance(value, list) or isinstance(value, np.ndarray):
            print()
            print(f"{key}: {value}")
            idx = 0
        elif key == "save_name":
            print()
        elif key == "save_path":
            save_path = value
        else:
            if idx % 3 == 2:
                if isinstance(value, float):
                    print(f"{key:<28}: {value:<15.{3}}|")
                else:
                    print(f"{key:<28}: {value:<15}|")
            else:
                if isinstance(value, float):
                    print(f"{key:<28}: {value:<15.{3}}|", end="")
                else:
                    print(f"{key:<28}: {value:<15}|", end="")

        idx += 1
