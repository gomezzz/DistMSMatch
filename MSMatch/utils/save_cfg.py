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

        # Converting dotmp to a string and creating a list of couples (key, value).
        cfg_couples = [
            [x.split("=")[0], x.split("=")[1]] for x in str(cfg)[7:].split(",")
        ]
        # Couples of key values are converted in a dictionary  {key1 : value1, key2: value2, ...}, requested by toml.dump to save the configuration.
        cfg_dict = dict(zip([x[0] for x in cfg_couples], [x[1] for x in cfg_couples]))

        cfg_filename = os.path.join(cfg.save_path, "cfg.toml")
        with open(cfg_filename, "w") as handle:
            toml.dump(cfg_dict, handle)
    except:
        raise ValueError("Impossible to save the configuration.")
