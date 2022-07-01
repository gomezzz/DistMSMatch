import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch


def set_seeds(seed: int):
    """Sets the seeds for the random number generators in torch, numpy and random.

    Args:
        seed (int): seed for the random number generators
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
