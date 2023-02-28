import torch

from .data_utils import split_ssl_data
from .BasicDataset import BasicDataset
from .EurosatRGBDataset import EurosatRGBDataset

from torchvision import transforms

mean, std = {}, {}
mean["eurosat_rgb"] = [x / 255 for x in [87.78644464, 96.96653968, 103.99007906]]
std["eurosat_rgb"] = [x / 255 for x in [51.92045453, 34.82338243, 29.26981551]]


def get_transform(mean, std, train=True):
    """Get weak augmentation transforms.

    Args:
        mean (float): mean of the dataset.
        std (float): std of the dataset.
        train (bool, optional): Whether training, in test only normalization is applied.

    Returns:
        torchvision.transforms.Compose: transforms.
    """
    if train:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0, 0.125)),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )


def get_inverse_transform(mean, std):
    """Get inverse transforms for weak augmentations.

    Args:
        mean (float): mean of the dataset.
        std (float): std of the dataset.

    Returns:
        torchvision.transforms.Compose: inverse transforms.
    """
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean_inv, std_inv)]
    )


class SSL_Dataset:
    """
    SSL_Dataset class separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(
        self,
        name="eurosat_rgb",
        train=True,
        data_dir="./data",
        seed=42,
        alpha=1.0,
        nodes=1,
        node_indx=1,
    ):
        """
        Args
            name: name of dataset
            train: True means the dataset is training dataset (default=True)
            data_dir: path of directory, where data is downloaed or stored.
            seed: seed to use for the train / test split. Not available for cifar which is presplit
        """

        self.name = name
        self.seed = seed
        self.train = train
        self.data_dir = data_dir
        self.transform = get_transform(mean[name], std[name], train)
        self.inv_transform = get_inverse_transform(mean[name], std[name])
        self.alpha = alpha
        self.nodes = nodes
        self.node_indx = node_indx

        self.use_ms_augmentations = False
        # need to use different augmentations for multispectral
        if self.name == "eurosat_ms":
            self.use_ms_augmentations = True

    def get_data(self, num_labels):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.name == "eurosat_rgb":
            dset = EurosatRGBDataset(
                train=self.train,
                seed=self.seed,
                num_labels=num_labels,
                alpha=self.alpha,
                nodes=self.nodes,
                node_indx=self.node_indx,
            )
        # elif self.name == "eurosat_ms":
        #     dset = EurosatDataset(train=self.train, seed=self.seed)
        else:
            raise NotImplementedError("Dataset {} is not available".format(self.name))

        self.label_encoding = dset.label_encoding
        self.num_classes = dset.num_classes
        self.num_channels = dset.num_channels

        lb_data, lb_targets = dset.lb_data, dset.lb_targets
        ulb_data, ulb_targets = dset.ul_data, dset.ul_targets
        return lb_data, lb_targets, ulb_data, ulb_targets

    def get_test_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.name == "eurosat_rgb":
            dset = EurosatRGBDataset(train=False, node_indx=self.node_indx)
        # elif self.name == "eurosat_ms":
        #     dset = EurosatDataset(train=self.train, seed=self.seed)
        else:
            raise NotImplementedError("Dataset {} is not available".format(self.name))

        self.label_encoding = dset.label_encoding
        self.num_classes = dset.num_classes
        self.num_channels = dset.num_channels

        return dset.test_data, dset.test_targets

    def get_dset(self, use_strong_transform=False, strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.

        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """

        data, targets = self.get_test_data()

        return BasicDataset(
            data,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
            onehot,
            self.use_ms_augmentations,
        )

    def get_ssl_dset(
        self,
        num_labels,
        use_strong_transform=True,
        strong_transform=None,
        onehot=False,
    ):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.

        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair.
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.

        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """

        lb_data, lb_targets, ulb_data, ulb_targets = self.get_data(num_labels)

        lb_dset = BasicDataset(
            lb_data,
            lb_targets,
            self.num_classes,
            self.transform,
            use_strong_transform=False,
            strong_transform=None,
            onehot=onehot,
            use_ms_augmentations=self.use_ms_augmentations,
        )

        ulb_dset = BasicDataset(
            ulb_data,
            ulb_targets,
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
            onehot,
            self.use_ms_augmentations,
        )

        return lb_dset, ulb_dset
