import torch

from .BasicDataset import BasicDataset
from .EurosatDataset import EurosatDataset

from torchvision import transforms

mean, std = {}, {}
mean["eurosat_rgb"] = [x / 255 for x in [87.78644464, 96.96653968, 103.99007906]]
std["eurosat_rgb"] = [x / 255 for x in [51.92045453, 34.82338243, 29.26981551]]

# fmt: off
mean["eurosat_ms"] = [x / 255 for x in [91.94472713,74.57486138,67.39810048,58.46731632,72.24985416,114.44099918,134.4489474,129.75758655,41.61089189,0.86983654,101.75149263,62.3835689,145.87144681,]]
std["eurosat_ms"] = [x / 255 for x in [52.42854549,41.13263869,35.29470731,35.12547202,32.75119418,39.77189372,50.80983189,53.91031257,21.51845906,0.54159901,56.63841871,42.25028442,60.01180004,]]
# fmt: on


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
        data_dir="./data",
        num_labels=100,
        seed=42,
        alpha=100,
        nodes=1,
        node_indx=0,
    ):
        """Setup for creating datasets

        Args:
            name (str, optional): name of dataset. Defaults to "eurosat_rgb".
            data_dir (str, optional): path of directory, where data is downloaed or stored.. Defaults to "./data".
            num_labels (int, optional): Number of labels to use. Defaults to 100.
            seed (int, optional): seed to use for the train / test split. Defaults to 42.
            alpha (int, optional): Data heterogeneity parameter in (0,inf]). The smaller the more heterogeneous partitions. Defaults to 100.
            nodes (int, optional): Number of nodes to partition among. Defaults to 1.
            node_indx (int, optional): Node index of calling process. Defaults to 0.
        """

        self.name = name
        self.seed = seed
        self.data_dir = data_dir
        self.num_labels = num_labels
        self.train_transform = get_transform(mean[name], std[name], train=True)
        self.test_transform = get_transform(mean[name], std[name], train=False)
        self.inv_transform = get_inverse_transform(mean[name], std[name])
        self.alpha = alpha
        self.nodes = nodes
        self.node_indx = node_indx

        # need to use different augmentations for multispectral
        if self.name == "eurosat_rgb":
            self.use_ms_augmentations = False
            self.root_dir = "./data/EuroSAT_RGB/"
        elif self.name == "eurosat_ms":
            self.use_ms_augmentations = True
            self.root_dir = "./data/EuroSATallBands/"
        else:
            print("Dataset not recognized", flush=True)
            return

    def create_node_partitions(self):
        """Prepare the data partitioning for the nodes. Should only be called from root process."""
        # Create EuroSat dataset object and check/add folder for partitions
        dset = EurosatDataset(
            seed=self.seed,
            root_dir=self.root_dir,
            num_labels=self.num_labels,
            alpha=self.alpha,
            nodes=self.nodes,
            node_indx=self.node_indx,
        )
        dset.prepare_data()  # Create partitions

    def get_ssl_dset(
        self,
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
            BasicDataset (for labeled data), BasicDataset (for unlabeld data), BasicDataset (for test data)
        """
        # Create the eurosat object
        if self.name == "eurosat_rgb" or self.name == "eurosat_ms":
            eurosat_dset = EurosatDataset(
                seed=self.seed,
                root_dir=self.root_dir,
                num_labels=self.num_labels,
                alpha=self.alpha,
                nodes=self.nodes,
                node_indx=self.node_indx,
            )
        else:
            raise NotImplementedError("Dataset {} is not available".format(self.name))

        # Prepare training data
        self.num_classes = eurosat_dset.num_classes
        self.num_channels = eurosat_dset.num_channels

        lb_data, lb_targets = eurosat_dset.lb_data, eurosat_dset.lb_targets
        ulb_data, ulb_targets = eurosat_dset.ul_data, eurosat_dset.ul_targets

        lb_dset = BasicDataset(
            lb_data,
            lb_targets,
            self.num_classes,
            self.train_transform,
            use_strong_transform=False,
            strong_transform=None,
            onehot=onehot,
            use_ms_augmentations=self.use_ms_augmentations,
        )

        ulb_dset = BasicDataset(
            ulb_data,
            ulb_targets,
            self.num_classes,
            self.train_transform,
            use_strong_transform,
            strong_transform,
            onehot,
            self.use_ms_augmentations,
        )

        # Prepare test data
        test_data = eurosat_dset.test_data
        test_targets = eurosat_dset.test_targets
        test_dset = BasicDataset(
            test_data,
            test_targets,
            self.num_classes,
            self.test_transform,
            use_strong_transform=False,
            strong_transform=None,
            onehot=False,
            use_ms_augmentations=self.use_ms_augmentations,
        )

        return lb_dset, ulb_dset, test_dset
