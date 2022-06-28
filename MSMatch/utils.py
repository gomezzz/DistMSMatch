import os
from efficientnet_pytorch import EfficientNet
import logging
from .models.nets.unet_encoder import UNetEncoder

def net_builder(net_name, pretrained=False, in_channels=3):
    """Creates the kind of neural network specified by net_name.

    Args:
        net_name (str): name of the network, can be efficientnet-b0, b1, b2, b3, b4, b5, b6, b7 or unet
        pretrained (bool, optional): if True, loads pretrained weights. Defaults to False.
        in_channels (int, optional): number of input channels. Defaults to 3.

    Returns:
        torch.nn.Module: the created network
    """
    
    if "efficientnet" in net_name:
        if pretrained:
            print("Using pretrained", net_name, "...")
            return lambda num_classes, in_channels: EfficientNet.from_pretrained(
                net_name, num_classes=num_classes, in_channels=in_channels
            )

        else:
            print("Using not pretrained model", net_name, "...")
            return lambda num_classes, in_channels: EfficientNet.from_name(
                net_name, num_classes=num_classes, in_channels=in_channels
            )
    elif "unet" in net_name.lower():
        assert in_channels == 3
        assert not pretrained
        return lambda num_classes, in_channels: UNetEncoder(
            num_classes=num_classes, in_channels=in_channels, scale=1.0
        )
    else:
        assert Exception("Not Implemented Error")


def get_logger(name, save_path=None, level="INFO"):
    """Initializes the logger

    Args:
        name (str): logger name
        save_path (str, optional): path to save the log file. Defaults to None. 
        level (str, optional): Logging level. Defaults to "INFO".

    Returns:
        Logger: the created logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger

def count_parameters(model):
    """Counts the number of parameters in a model.

    Args:
        model (torch.model): model to count parameters of

    Returns:
        int: number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_dir_str(cfg):
    """Creates a string from the arguments.

    Args:
        cfg (DotMap): config dictionary/dotmap

    Returns:
        _type_: _description_
    """
    # fmt: off
    dir_name = (
        cfg.dataset
        + "/FixMatch_arch"  + cfg.net
        + "_batch"          + str(cfg.batch_size)
        + "_confidence"     + str(cfg.p_cutoff)
        + "_lr"             + str(cfg.lr)
        + "_uratio"         + str(cfg.uratio)
        + "_wd"             + str(cfg.weight_decay)
        + "_wu"             + str(cfg.ulb_loss_ratio)
        + "_seed"           + str(cfg.seed)
        + "_numlabels"      + str(cfg.num_labels)
        + "_opt"            + str(cfg.opt)
    )
    # fmt: on
    if cfg.pretrained:
        dir_name = dir_name + "_pretrained"
    return dir_name