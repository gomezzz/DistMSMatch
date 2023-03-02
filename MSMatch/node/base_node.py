import torch
from ..utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from ..utils.get_optimizer import get_optimizer
from ..utils.get_net_builder import get_net_builder
from ..utils.TensorBoardLog import TensorBoardLog
from ..models.fixmatch.FixMatch import FixMatch


class BaseNode:
    def __init__(self, rank, cfg=None, dataloader=None, logger=None):
        self.cfg = cfg
        self.rank = rank
        self.logger = logger
        self.save_path = cfg.save_path
        self.sim_path = cfg.sim_path
        self.tb_log = TensorBoardLog(self.save_path, "")
        self.accuracy = []

        # Create model
        if rank is not None:
            self.device = (
                "cuda:{}".format(self.rank % torch.cuda.device_count())
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = "cpu"
        
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

        # self.logger.info(
        #     f"Number of Trainable Params: {sum(p.numel() for p in model.train_model.parameters() if p.requires_grad)}"
        # )

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

        # self.logger.info(f"model_arch: {model}")
        # self.logger.info(f"Arguments: {self.cfg}")

        return model

    def aggregate(self):
        # self.logger.info(f"Node {self.rank}: aggregating neighbor models")
        # TODO: change this to weighted average based on samples
        # cw = 1/(len(neighbor_sd)+1)
        self.model.train_model.to(self.device)
        local_sd = self.model.train_model.state_dict()
        # neighbor_sd = [m.state_dict() for m in rx_models]
        cw = 1 / (len(self.ranks_in_lineofsight) + 1)

        for key in local_sd:
            local_sd[key] = cw * local_sd[key].to(self.device)

        for i in self.ranks_in_lineofsight:
            new_sd = torch.load(f"{self.sim_path}/node{i}/model.pt").state_dict()
            for key in local_sd:
                local_sd[key] += new_sd[key].to(self.device) * cw

        # update server model with aggregated models
        self.model.train_model.load_state_dict(local_sd)
        self.model.eval_model.to(self.device)
        self.model._eval_model_update()
        
        self.model.eval_model.cpu()
        self.model.train_model.cpu()

        self.do_training = True

    def save_model(self):
        if self.cfg.mode == "Swarm":
            torch.save(self.model.train_model, f"{self.save_path}/model.pt") # save trained model
        else:
            torch.save(self.model.train_model, f"{self.sim_path}/node{self.rank}_model.pt") # save trained model