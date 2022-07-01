from efficientnet_pytorch import EfficientNet
from ..models.nets.unet_encoder import UNetEncoder


def get_net_builder(net_name, pretrained=False, in_channels=3):
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
