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


class EurosatRGBDataset(torch.utils.data.Dataset):
    """EurosatRGB dataset"""

    def __init__(
        self,
        train,
        root_dir="./data/EuroSAT_RGB/",
        transform=None,
        seed=42,
        num_labels=100,
        nodes=2,
        alpha=0.2,
        node_indx=1,
    ):
        """_summary_

        Args:
            train (bool): If true returns training set, else test
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): seed used for train/test split
            nodes (int, optional): number of nodes to split data across. Defaults to 1.
            alpha (_type_, optional): parameter in [0,1] to decide heterogeneity in data partitioning. Heterogeneity increases towards 0. Defaults to None.
        """
        self.seed = seed
        self.size = [64, 64]
        self.num_channels = 3
        self.num_classes = 10
        self.root_dir = root_dir
        self.transform = transform
        self.test_ratio = 0.1
        self.train = train
        self.N = 27000
        self.nodes = nodes
        self.alpha = alpha
        self.node_indx = node_indx
        self.data_exist = False
        self.num_labels = num_labels
        self._load_data()

    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256"""

        # if folder exists and config matches, load from folder.
        # otherwise, create partitioning
        data_folder = self._look_for_data()

        if not self.data_exist:
            # load all images
            images = np.zeros([self.N, self.size[0], self.size[1], 3], dtype="uint8")
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
                            resize(
                                image, (self.size[0], self.size[1]), anti_aliasing=True
                            )
                        )
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

            # save data configuration
            data_config = DotMap(_dynamic=False)
            data_config.test_ratio = self.test_ratio
            data_config.label_encoding = self.label_encoding
            data_config.seed = self.seed
            data_config.datashape = images.shape
            data_config.num_labels = self.num_labels
            with open(data_folder + "/data_config.json", "w+") as f:
                json.dump(data_config, f)

            # divide training data into labeled and unlabeled data
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
                X_train,
                y_train,
                self.num_labels,
                self.num_classes,
                index=None,
                include_lb_to_ulb=False,
            )

            # Partition unlabeled data between clients
            node_dataidx_map = self._partition_data(ulb_targets)

            # save unlabeled training data for each node
            for node_indx in node_dataidx_map:
                file_name = data_folder + f"/node_{node_indx}"
                data_idxs = node_dataidx_map[node_indx]
                np.save(file_name + "ul-data", ulb_data[data_idxs, :, :, :])
                np.save(file_name + "ul-targets", ulb_targets[data_idxs])
            # save labeled data
            np.save(data_folder + "/lb-data", lb_data)
            np.save(data_folder + "/lb-targets", lb_targets)
            # save test data
            np.save(data_folder + "/test-data", X_test)
            np.save(data_folder + "/test-targets", y_test)

        # load data from folder
        if self.train:
            client_data_folder = data_folder + f"/node_{self.node_indx}"
            self.ul_data = np.load(client_data_folder + "ul-data.npy")
            self.ul_targets = np.load(client_data_folder + "ul-targets.npy")  # not used

            self.ul_cls_counts = self._node_cls_count(self.ul_targets)
            print(
                f"Node {self.node_indx} unlabeled class distribution:{self.ul_cls_counts}"
            )

            self.lb_data = np.load(data_folder + "/lb-data.npy")
            self.lb_targets = np.load(data_folder + "/lb-targets.npy")
            self.lb_cls_counts = self._node_cls_count(self.lb_targets)
            print(
                f"Node {self.node_indx} labeled class distribution:{self.lb_cls_counts}"
            )
        else:
            self.test_data = np.load(data_folder + "/test-data.npy")
            self.test_targets = np.load(data_folder + "/test-targets.npy")
            self.test_cls_counts = self._node_cls_count(self.test_targets)
            print(
                f"Node {self.node_indx} test label distribution:{self.test_cls_counts}"
            )

    def _look_for_data(self):
        data_folder = None
        if not self.data_exist:
            data_dir = (
                f"./data/alpha_"
                + str(self.alpha).replace(".", "")
                + f"-nodes_{self.nodes}"
            )
            folder_exists = os.path.exists(data_dir)
            if folder_exists:
                for _, dirs, files in os.walk(data_dir):
                    for subdir in dirs:
                        cur_folder = data_dir + "/" + subdir
                        data_config_path = cur_folder + "/data_config.json"
                        config_exists = os.path.exists(data_config_path)
                        if config_exists:
                            f = open(data_config_path)
                            data_config = json.load(f)
                            if (
                                data_config["test_ratio"] == self.test_ratio
                                and data_config["seed"] == self.seed
                                and data_config["num_labels"] == self.num_labels
                            ):
                                self.label_encoding = data_config["label_encoding"]
                                self.data_exist = True
                                data_folder = cur_folder

        if data_folder is None:
            data_folder = os.path.join(
                data_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            os.makedirs(data_folder)

        return data_folder

    def _partition_data(self, labels):
        """Partition the dataset over the nodes

        Args:
            labels (_type_): labels in the original dataset

        Returns:
            _type_: _description_
        """
        n_labels = labels.shape[0]
        class_num = np.unique(labels)

        if self.alpha is None:
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

    def _node_cls_count(self, targets):
        # count the number of data points in each classes for each node
        unq, unq_cnt = np.unique(targets, return_counts=True)
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
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]
