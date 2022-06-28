import os
from efficientnet_pytorch import EfficientNet
import logging
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from .models.nets.unet_encoder import UNetEncoder

def net_builder(
    net_name, net_conf=None, pretrained=False, in_channels=3
):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
        pre_trained: Specifies if a pretrained network should be loaded (only works for efficientNet)
        in_channels: Input channels to the network
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_dir_str(args):
    dir_name = (
        args.dataset
        + "/FixMatch_arch"
        + args.net
        + "_batch"
        + str(args.batch_size)
        + "_confidence"
        + str(args.p_cutoff)
        + "_lr"
        + str(args.lr)
        + "_uratio"
        + str(args.uratio)
        + "_wd"
        + str(args.weight_decay)
        + "_wu"
        + str(args.ulb_loss_ratio)
        + "_seed"
        + str(args.seed)
        + "_numlabels"
        + str(args.num_labels)
        + "_opt"
        + str(args.opt)
    )
    if args.pretrained:
        dir_name = dir_name + "_pretrained"
    return dir_name