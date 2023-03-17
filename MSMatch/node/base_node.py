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
        self.sim_path = cfg.sim_path
        self.scale = cfg.scale
        self.num_classes = cfg.num_classes
        self.num_channels = cfg.num_channels
        self.ema_m = cfg.ema_m
        self.T = cfg.T
        self.p_cutoff = cfg.p_cutoff
        self.ulb_loss_ratio = cfg.ulb_loss_ratio
        self.pretrained = cfg.pretrained
        self.opt = cfg.opt
        self.lr = cfg.lr
        self.momentum = cfg.momentum
        self.weight_decay = cfg.weight_decay
        self.num_train_iter = cfg.num_train_iter
        self.net = cfg.net
        self.mode = cfg.mode

        # server nodes are treated differently as no training is needed
        if is_server:
            self.device = "cpu"  
            self.model = self._create_model() # Create model
        else:
            self.device = (
                "cuda:{}".format(self.rank % torch.cuda.device_count())
                if torch.cuda.is_available()
                else "cpu"
            )
            self.model = self._create_model() # Create model
            self.n_gpus = torch.cuda.device_count()

            self.model.set_data_loader(dataloader)  # create data iterators for training

    def _create_model(self):
        """Create the models to be trained. Two equal models are created, one for training 
        and one for testing where the latter is updated from the trained model via exponential averaging.

        Returns:
            _type_: _description_
        """
        net_builder = get_net_builder(
            self.net,
            pretrained=self.pretrained,
            in_channels=self.num_channels,
            scale=self.scale,
        )

        model = FixMatch(
            net_builder,
            self.num_classes,
            self.num_channels,
            self.ema_m,
            T=self.T,
            p_cutoff=self.p_cutoff,
            lambda_u=self.ulb_loss_ratio,
            hard_label=True,
            device=self.device,
            rank=self.rank,
        )

        # get optimizer, ADAM and SGD are supported.
        optimizer = get_optimizer(
            model.train_model,
            self.opt,
            self.lr,
            self.momentum,
            self.weight_decay,
        )
        # We use a learning rate schedule to control the learning rate during training.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            self.num_train_iter,
            num_warmup_steps=self.num_train_iter * 0,
        )
        model.set_optimizer(optimizer, scheduler)

        return model

    def save_model(self,name):
        """Save the training model to folder
        """        
        torch.save(
            self.model.train_model, f"{self.sim_path}/{name}.pt"
        )  # save trained model
