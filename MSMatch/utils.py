import os
from efficientnet_pytorch import EfficientNet
import logging
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from .models.nets.unet_encoder import UNetEncoder

def plot_examples(images,labels,encoding, figsize=(8, 5),dpi=150, labels_fontsize=5, prediction=None, save_fig_name=None):
    """Plotting 32 randomly sampled image examples for a target dataset, ensuring that at least one image for each class is got. If `prediction` is given, both predicted and expected classes are shown for each image.

    Args:
        images ([list]): list of images to plot.
        labels ([list]): list of predicted classes.
        encoding ([list]): classes label encoding.
        figsize (tuple, optional): size of the output figure. Defaults to (8, 5).
        dpi (int, optional): Dots for inch. Defaults to 150.
        labels_fontsize ([str]): label fontsize. Default to 5.
        prediction ([list], optional): List of predicted classes. Defaults to None.
        save_fig_name ([str], optional): output figure name. If 'None', no output figure is saved. Defaults to None.
    """
    def sort_x_according_to_y(x,y):
        return [x for _,x in sorted(zip(y,x))]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    class_found=[]
    shuffled_idx=list(np.random.permutation(len(labels)))
    
    labels=sort_x_according_to_y(labels, shuffled_idx)
    images=sort_x_according_to_y(images, shuffled_idx)
    if prediction is not None:
        prediction=sort_x_according_to_y(prediction, shuffled_idx)
        
        
    labels_idx=[]
    class_found=[]

    for l in range(len(labels)):
        if not(labels[l] in class_found):
            labels_idx.append(l)
            class_found.append(labels[l])
            
    print("Number of different classes found:", len(class_found))
            
    n_to_add= 32 - len(labels_idx)
            
    for l in range(len(labels)):
        if n_to_add == 0:
            break
            
        if not(l in labels_idx):
            labels_idx.append(l)
            n_to_add-=1
            
    
    #rand_indices=sample(range(len(images)), 32)
    for idx, rand_idx in enumerate(labels_idx):
        img = images[rand_idx]
        ax = fig.add_subplot(4, 8, idx+1, xticks=[], yticks=[])
        if np.max(img) > 1.5:
            img = img / 255
        plt.imshow(img)
        if prediction is not None:
            label = "GT: " + encoding[labels[rand_idx]] + "\n PR: " + encoding[prediction[rand_idx]]
        else:
            label = encoding[labels[rand_idx]]    
        plt.title(str(label),fontsize=labels_fontsize)

    if save_fig_name is not None:
        plt.savefig(save_fig_name)


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(
                f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}"
            )
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = "hello"

    test_cls = _test_cls()
    config = {"a": 3, "b": "change_hello", "c": 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")


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


def test_net_builder(net_name, from_name, net_conf=None, pretrained=False):
    builder = net_builder(net_name, from_name, net_conf, pretrained)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


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