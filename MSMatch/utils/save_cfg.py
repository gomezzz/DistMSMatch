import toml
import os


def save_cfg(cfg):
    """Save configuration to the run directory.

    Args:
        cfg (Dotmap): configuration.

    Raises:
        ValueError: Impossible to save the configuration.
    """
    try:
        cfg_filename = os.path.join(cfg.save_path, "cfg.toml")
        with open(cfg_filename, "w") as handle:
            toml.dump(cfg.toDict(), handle)
    except Exception as e:
        raise ValueError(
            "An error occured saving the configuration.\n" "Error: {}".format(e)
        )
