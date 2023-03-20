import torch
from ..utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from ..utils.get_optimizer import get_optimizer
from ..utils.get_net_builder import get_net_builder
from ..models.fixmatch.FixMatch import FixMatch


class BaseNode:
    """Class to form the foundation of ServerNode and SpacecraftNode by initializing the neural networks to be trained."""

    def __init__(self, rank, cfg, dataloader=None, is_server=False):
        self.rank = rank
        self.accuracy = []

        # Read out parameters from cfg
        self.cfg = cfg

        # server nodes are treated differently as no training is needed
        if is_server:
            self.device = "cpu"
            self.model = self._create_model()  # Create model
        else:
            self.device = (
                "cuda:{}".format(self.rank % torch.cuda.device_count())
                if torch.cuda.is_available()
                else "cpu"
            )
            self.model = self._create_model()  # Create model
            self.n_gpus = torch.cuda.device_count()

            self.model.set_data_loader(dataloader)  # create data iterators for training

    def _create_model(self):
        """Create the models to be trained. Two equal models are created, one for training
        and one for testing where the latter is updated from the trained model via exponential averaging.

        Returns:
            FixMatch: object including training and evaluation model
        """
        net_builder = get_net_builder(
            self.cfg.net,
            pretrained=self.cfg.pretrained,
            in_channels=self.cfg.num_channels,
            scale=self.cfg.scale,
        )

        model = FixMatch(
            net_builder,
            self.cfg.num_classes,
            self.cfg.num_channels,
            self.cfg.ema_m,
            T=self.cfg.T,
            p_cutoff=self.cfg.p_cutoff,
            lambda_u=self.cfg.ulb_loss_ratio,
            hard_label=True,
            device=self.device,
            rank=self.rank,
        )

        # get optimizer, ADAM and SGD are supported.
        optimizer = get_optimizer(
            model.train_model,
            self.cfg.opt,
            self.cfg.lr,
            self.cfg.momentum,
            self.cfg.weight_decay,
        )
        # We use a learning rate schedule to control the learning rate during training.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            self.cfg.num_train_iter,
            num_warmup_steps=self.cfg.num_train_iter * 0,
        )
        model.set_optimizer(optimizer, scheduler)

        return model

    def save_model(self, name):
        """Save the training model to folder"""
        torch.save(
            self.model.train_model, f"{self.cfg.sim_path}/{name}.pt"
        )  # save trained model
