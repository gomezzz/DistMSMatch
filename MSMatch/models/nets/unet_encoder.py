import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        scale=1.0,
    ):
        super(UNetEncoder, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, int(scale * 64), 3)
        self.conv1_2 = nn.Conv2d(int(scale * 64), int(scale * 64), 3)

        self.conv2_1 = nn.Conv2d(int(scale * 64), int(scale * 128), 3)
        self.conv2_2 = nn.Conv2d(int(scale * 128), int(scale * 128), 3)

        self.conv3_1 = nn.Conv2d(int(scale * 128), int(scale * 256), 3)
        self.conv3_2 = nn.Conv2d(int(scale * 256), int(scale * 256), 3)

        # Potential fourth layer commented out for now to save memory
        # self.conv4_1 = nn.Conv2d(int(scale * 256), int(scale * 512), 3)
        # self.conv4_2 = nn.Conv2d(int(scale * 512), int(scale * 512), 3)

        self.fc = nn.Linear(int(scale * 256), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(F.relu(self.conv1_2(x)), 2)

        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(F.relu(self.conv2_2(x)), 2)

        x = F.relu(self.conv3_1(x))
        x = F.max_pool2d(F.relu(self.conv3_2(x)), 2)

        # Potential fourth layer commented out for now to save memory
        # x = F.relu(self.conv4_1(x))
        # x = F.max_pool2d(F.relu(self.conv4_2(x)), 2)

        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()

        x = self.fc(x)
        return x


if __name__ == "__main__":
    # For testing
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = UNetEncoder(3, 10)
    print(f"Number of Trainable Params: {count_parameters(net)}")
    img = torch.ones((16, 3, 64, 64))
    net(img)