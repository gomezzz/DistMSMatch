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


class EurosatRGBDataset(torch.utils.data.Dataset):
    """EurosatRGB dataset"""

    def __init__(
        self,
        train,
        root_dir="../data/EuroSAT_RGB/",
        transform=None,
        seed=42,
        nodes=1,
        alpha=None,
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
        self._load_data()

    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256"""
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
                        resize(image, (self.size[0], self.size[1]), anti_aliasing=True)
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

        # THIS IS WHERE PARTITIONING SHOULD GO
        if self.nodes > 1:
            node_dataidx_map, _ = self._partition_data(labels)

        else:
            # split into a train and test set as provided data is not presplit
            X_train, X_test, y_train, y_test = train_test_split(
                images,
                labels,
                test_size=self.test_ratio,
                random_state=self.seed,
                stratify=labels,
            )

        if self.train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test

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
                for k in range(class_num):
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

        # count the number of data points in each classes for each node
        node_cls_counts = {}
        for node_i, dataidx in node_dataidx_map.items():
            unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            node_cls_counts[node_i] = tmp

        return node_dataidx_map, node_cls_counts

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
