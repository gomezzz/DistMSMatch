import torch
from ..utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from ..utils.get_optimizer import get_optimizer
from ..utils.get_net_builder import get_net_builder
from ..models.fixmatch.FixMatch import FixMatch


class BaseNode:
    def __init__(self, rank, cfg=None, dataloader=None, logger=None):
        self.cfg = cfg
        self.rank = rank
        self.logger = logger
        self.save_path = cfg.save_path
        self.sim_path = cfg.sim_path
        self.accuracy = []
        
        # Servernode will not have a rank
        if rank is not None:
            self.device = (
                "cuda:{}".format(self.rank % torch.cuda.device_count())
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = "cpu"
            
        # Create model
        self.model = self._create_model()
        self.n_gpus = torch.cuda.device_count()
        
        if rank is not None:
            self.model.set_data_loader(dataloader)

    def _create_model(self):
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

    def save_model(self):
        if self.cfg.mode == "Swarm":
            torch.save(
                self.model.train_model, f"{self.save_path}/model.pt"
            )  # save trained model
        else:
            torch.save(
                self.model.train_model, f"{self.sim_path}/node{self.rank}_model.pt"
            )  # save trained model
