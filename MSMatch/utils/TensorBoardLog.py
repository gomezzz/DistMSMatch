from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardLog:
    """
    Construct tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None):
        """Update the tensorboard with the given dictionary.

        Args:
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ""

        for key, value in tb_dict.items():
            self.writer.add_scalar(suffix + key, value, it)
