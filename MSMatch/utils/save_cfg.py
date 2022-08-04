
import toml
import os

def save_cfg(cfg):
    try:
        cfg_couples=[[x.split("=")[0], x.split("=")[1]]  for x in str(cfg)[7:].split(",")]
        cfg_dict=dict(zip([x[0] for x in cfg_couples], [x[1] for x in cfg_couples]))
        cfg_filename=os.path.join(cfg.save_path, "cfg.toml")
        with open(cfg_filename, "w") as handle:
            toml.dump(cfg_dict, handle)
    except:
        raise ValueError("Impossible to save the configuration.")