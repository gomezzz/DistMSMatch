from efficientnet_pytorch import EfficientNet
from ..models.nets.unet_encoder import UNetEncoder
import efficientnet_lite_pytorch
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile


def get_net_builder(net_name, pretrained=False, in_channels=3, scale=1):
    """Creates the kind of neural network specified by net_name.

    Args:
        net_name (str): name of the network, can be efficientnet-b0, b1, b2, b3, b4, b5, b6, b7 or unet
        pretrained (bool, optional): if True, loads pretrained weights. Defaults to False.
        in_channels (int, optional): number of input channels. Defaults to 3.
        scale (float, optional): scale of the network. Defaults to 1.0.
    Returns:
        torch.nn.Module: the created network
    """
    if "efficientnet-lite" in net_name:
        if pretrained:
            if net_name == "efficientnet-lite0":
                print("Using pretrained", net_name, "...")
                weights_path = EfficientnetLite0ModelFile.get_model_file_path()

                return lambda num_classes, in_channels: efficientnet_lite_pytorch.EfficientNet.from_pretrained(
                    "efficientnet-lite0",
                    weights_path=weights_path,
                    num_classes=num_classes,
                    in_channels=in_channels,
                )
            else:
                print("ERROR. Only efficientnet-lite0 pretrained is supported.")
                print("Using not pretrained model", net_name, "...")
                return lambda num_classes, in_channels: efficientnet_lite_pytorch.EfficientNet.from_name(
                    net_name, num_classes=num_classes, in_channels=in_channels
                )
        else:
            print("Using not pretrained model", net_name, "...")
            return lambda num_classes, in_channels: efficientnet_lite_pytorch.EfficientNet.from_name(
                net_name, num_classes=num_classes, in_channels=in_channels
            )

    elif "efficientnet" in net_name:
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
            num_classes=num_classes, in_channels=in_channels, scale=scale
        )
    else:
        assert Exception("Not Implemented Error")
