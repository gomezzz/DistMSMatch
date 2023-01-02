import torch
import os

from ..utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from ..utils.get_optimizer import get_optimizer
from ..utils.get_net_builder import get_net_builder
from ..utils.accurarcy import accuracy
from ..utils.TensorBoardLog import TensorBoardLog
from ..models.fixmatch.FixMatch import FixMatch

class Node:
    def __init__(
        self,
        node_indx, 
        cfg,
        dataloader,
        logger,
        tb_log
    ):
    
        super(Node, self).__init__()
        
        self.cfg = cfg
        self.node_indx = node_indx
        self.logger = logger
        
        save_path = os.path.join(cfg.save_dir, f"node {self.node_indx}")
        self.tb_log = TensorBoardLog(save_path, "")
        
        # Create model
        self.model = self._create_model()
        self.model.set_data_loader(dataloader)
        
        # Set up PASEOS on node
        
        # Set up network layer on node
    
    def _create_model(self):
        net_builder = get_net_builder(
            self.cfg.net,
            pretrained=self.cfg.pretrained,
            in_channels=self.cfg.num_channels,
            scale=self.cfg.scale
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
                num_eval_iter=self.cfg.num_eval_iter
            )
        self.logger.info(f"Number of Trainable Params: {sum(p.numel() for p in model.train_model.parameters() if p.requires_grad)}")

        # get optimizer, ADAM and SGD are supported.
        optimizer = get_optimizer(
            model.train_model, self.cfg.opt, self.cfg.lr, self.cfg.momentum, self.cfg.weight_decay
        )
        # We use a learning rate schedule to control the learning rate during training.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.cfg.num_train_iter, num_warmup_steps=self.cfg.num_train_iter * 0
        )
        model.set_optimizer(optimizer, scheduler)
        
        # If a CUDA capable GPU is used, we move everything to the GPU now
        if torch.cuda.is_available():
            torch.cuda.set_device(self.cfg.gpu)
            model.train_model = model.train_model.cuda(self.cfg.gpu)
            model.eval_model = model.eval_model.cuda(self.cfg.gpu)

            self.logger.info(f"model_arch: {model}")
            self.logger.info(f"Arguments: {self.cfg}")
        return model
    
    def train(self):
        self.logger.info(f"Node {self.node_indx}")
        result = self.evaluate()
        self.logger.info(f"post aggregated acc: {result['eval/top-1-acc']}")
        result = self.model.train(self.cfg)
        self.logger.info(f"post training acc: {result['eval/top-1-acc']}")
        
    def evaluate(self):
        return self.model.evaluate(cfg=self.cfg)

    def save_model(self):
        self.model.save_run("latest_model.pth", self.cfg.save_path, self.cfg)
        
    def aggregate(self, rx_models):
        self.logger.info(f"Aggregating neighbor models")
        # TODO: change this to weighted average based on samples
        cw = 1/(len(rx_models)+1)
        
        local_sd = self.model.train_model.state_dict()
        neighbor_sd = [m.state_dict() for m in rx_models]
        for key in local_sd:
            local_sd[key] = cw * local_sd[key] + sum([sd[key] * cw for i, sd in enumerate(neighbor_sd)])
        
        # update server model with aggregated models
        self.model.train_model.load_state_dict(local_sd)
        self.model._eval_model_update()

            


