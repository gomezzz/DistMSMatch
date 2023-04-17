import imageio
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from datetime import datetime
import json
from dotmap import DotMap
from .data_utils import split_ssl_data


class EurosatDataset(torch.utils.data.Dataset):
    """Eurosat dataset"""

    def __init__(
        self,
        root_dir: str,
        num_labels: int,
        nodes: int,
        node_indx: int,
        transform=None,
        seed=42,
        alpha=np.inf,
    ):
        """_summary_

        Args:
            train (bool): If true returns training set, else test
            root_dir (str): path to data
            num_labels (int): number of labeled samples to include
            nodes (int): total number of nodes
            node_indx (int): node index of current node
            transform (_type_, optional): Defaults to None.
            seed (int, optional):  Defaults to 42.
            alpha (_type_, optional): Decides data heterogeneity, value in (0,inf]. Defaults to np.inf (homogeneous partitions).
        """
        self.root_dir = root_dir
        self.num_labels = num_labels
        self.nodes = nodes
        self.node_indx = node_indx
        self.transform = transform
        self.seed = seed
        self.alpha = alpha

        self.size = [64, 64]

        if "EuroSAT_RGB" in self.root_dir:
            self.num_channels = 3
            self.multispectral = False
        elif "EuroSATallBands" in self.root_dir:
            self.num_channels = 13
            self.multispectral = True
        else:
            print("root directory not recognized", flush=True)

        self.num_classes = 10
        self.test_ratio = 0.1
        self.N = 27000

        self.prepare_data()

    def _normalize_to_0_to_1(self, img):
        """Normalizes the passed image to 0 to 1

        Args:
            img (np.array): image to normalize

        Returns:
            np.array: normalized image
        """
        img = img + np.minimum(0, np.min(img))  # move min to 0
        img = img / np.max(img)  # scale to 0 to 1
        return img

    def prepare_data(self):
        """If the data partitioning does not exist:
        1. Load all the data from the passed root directory
        2. Convert label strings to a class index
        3. Create training and test sets
        4. Split training data into labeled and unlabeled parts
        5. Partition the unlabeled part into number of nodes parts
        6. Save partitions to data_folder
        The data is by default resized to self.size
        """

        # load all images
        images = np.zeros(
            [self.N, self.size[0], self.size[1], self.num_channels], dtype="uint8"
        )
        labels = []
        filenames = []

        i = 0
        # read all the files from the image folder
        for item in tqdm(os.listdir(self.root_dir)):
            f = os.path.join(self.root_dir, item)
            if os.path.isfile(f):
                continue
            for subitem in os.listdir(f):
                sub_f = os.path.join(f, subitem)
                filenames.append(sub_f)
                # a few images are a few pixels off, we will resize them
                image = imageio.imread(sub_f)
                if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
                    # print("Resizing image...")
                    image = img_as_ubyte(
                        resize(image, (self.size[0], self.size[1]), anti_aliasing=True)
                    )
                if self.multispectral:
                    images[i] = img_as_ubyte(self._normalize_to_0_to_1(image))
                else:
                    images[i] = img_as_ubyte(image)
                i += 1
                labels.append(item)

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # sort by filenames
        images = images[filenames.argsort()]
        labels = labels[filenames.argsort()]

        # convert to integer labels
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)
        self.label_encoding = list(le.classes_)  # remember label encoding

        # split into a train and test set as provided data is not presplit
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=labels,
        )

        # divide training data into labeled and unlabeled data
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
            X_train,
            y_train,
            self.num_labels,
            self.num_classes,
            index=None,
            include_lb_to_ulb=False,
        )

        # Partition unlabeled data between clients,
        # returns a list of indices that goes into each partition from ulb data
        node_dataidx_map = self._partition_data(ulb_targets)

        # save unlabeled training data for each node
        data_idxs = node_dataidx_map[self.node_indx]
        self.ul_data = ulb_data[data_idxs, :, :, :]
        self.ul_targets = ulb_targets[data_idxs]

        # save labeled data
        self.lb_data = lb_data
        self.lb_targets = lb_targets

        self.test_data = X_test
        self.test_targets = y_test

        self.ul_cls_counts = self._node_cls_count(self.ul_targets)
        self.lb_cls_counts = self._node_cls_count(self.lb_targets)
        self.test_cls_counts = self._node_cls_count(self.test_targets)

        print(
            f"Node {self.node_indx} \n"
            + f"unlabeled class distribution:{self.ul_cls_counts} \n"
            + f"labeled class distribution:{self.lb_cls_counts} \n"
            + f"test label distribution:{self.test_cls_counts} \n",
            flush=True,
        )

    def _partition_data(self, labels):
        """Partition the dataset over the nodes.
        Alpha = np.inf randomly splits the data into partitions whereas
        alpha in (0,inf) assigns samples using latent dirichlet allocation

        Args:
            labels (_type_): labels in the original dataset

        Returns:
            _type_: list of data indices for each partition
        """
        n_labels = labels.shape[0]
        class_num = np.unique(labels)

        if self.alpha == np.inf:
            idxs = np.random.permutation(n_labels)
            node_idxs = np.array_split(idxs, self.nodes)
            node_dataidx_map = {i: node_idxs[i] for i in range(self.nodes)}

        else:
            min_size = 0
            node_dataidx_map = {}

            while min_size < 10:
                idx_batch = [[] for _ in range(self.nodes)]

                # divide each label among the different nodes
                for k in class_num:
                    idx_k = np.where(labels == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.alpha, self.nodes))
                    ## Balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < n_labels / self.nodes)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self.nodes):
                np.random.shuffle(idx_batch[j])
                node_dataidx_map[j] = idx_batch[j]

        return node_dataidx_map

    def _node_cls_count(self, labels):
        """Count the number of data points in each classes for each node

        Args:
            labels (_type_): labels of entire set

        Returns:
            _type_: count of occurences of each labels
        """
        unq, unq_cnt = np.unique(labels, return_counts=True)
        cls_count = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        return cls_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not self.multispectral:
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]
