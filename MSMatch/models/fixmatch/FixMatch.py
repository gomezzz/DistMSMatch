import torch
import torch.nn.functional as F

import os
import sys
from pathlib import Path

from ...utils.consistency_loss import consistency_loss
from ...utils.cross_entropy_loss import cross_entropy_loss
from ...utils.accurarcy import accuracy
from ...utils.save_cfg import save_cfg


class FixMatch:
    def __init__(
        self,
        net_builder,
        num_classes,
        in_channels,
        ema_m,
        T,
        p_cutoff,
        lambda_u,
        hard_label=True,
        tb_log=None,
        logger=None,
        device="cpu",
        rank=None,
    ):
        """_summary_

        Args:
            net_builder: backbone network class (see get_net_builder in utils)
            num_classes: # of label classes
            in_channels: number of image channels
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see get_logger.py)
            rank (_type_, optional): node index. Defaults to None.
        """
        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # note there is a separate eval model due to exponential moving average of the weights
        self.train_model = net_builder(num_classes=num_classes, in_channels=in_channels)
        self.eval_model = net_builder(num_classes=num_classes, in_channels=in_channels)
        self.T = T  # temperature params function
        self.p_cutoff = p_cutoff  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        self.device = device
        self.rank = rank

        self.optimizer = None
        self.scheduler = None
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        for param_q, param_k in zip(
            self.train_model.parameters(), self.eval_model.parameters()
        ):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net

        self.eval_model.eval()

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """

        train_model_params = (
            self.train_model.module.parameters()
            if hasattr(self.train_model, "module")
            else self.train_model.parameters()
        )
        for param_train, param_eval in zip(
            train_model_params, self.eval_model.parameters()
        ):
            param_eval.copy_(
                param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m)
            )

        for buffer_train, buffer_eval in zip(
            self.train_model.buffers(), self.eval_model.buffers()
        ):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        """Create iterators from dataloaders for training

        Args:
            loader_dict (_type_): _description_
        """
        self.loader_dict = loader_dict
        self.print_fn(f"[!] data loader keys: {self.loader_dict.keys()}")
        self.lb_iterator = iter(self.loader_dict["train_lb"])
        self.ulb_iterator = iter(self.loader_dict["train_ulb"])

    def set_optimizer(self, optimizer, scheduler=None):
        """Assign optimizer and learning rate scheduling

        Args:
            optimizer (_type_): optimizer object
            scheduler (_type_, optional): scheduling object. Defaults to None.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_batch(self, cfg):
        """Train function of FixMatch for one batch.

        Args:
            cfg (_type_): config file

        Returns:
            train_accuracy: accuracy evaluated on test data
        """
        self.train_model.to(self.device)
        self.eval_model.to(self.device)
        self.train_model.train()

        # sample a batch of unlabeled data
        try:
            (x_ulb_w, x_ulb_s, _) = next(self.ulb_iterator)
        except StopIteration:
            self.ulb_iterator = iter(self.loader_dict["train_ulb"])
            (x_ulb_w, x_ulb_s, _) = next(self.ulb_iterator)

        # sample a batch of labeled data
        try:
            (x_lb, y_lb) = next(self.lb_iterator)
        except StopIteration:
            self.lb_iterator = iter(self.loader_dict["train_lb"])
            (x_lb, y_lb) = next(self.lb_iterator)

        self.train_model.zero_grad()

        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        # Move data to gpu
        if torch.cuda.is_available():
            x_lb, x_ulb_w, x_ulb_s = (
                x_lb.to(self.device),
                x_ulb_w.to(self.device),
                x_ulb_s.to(self.device),
            )
            y_lb = y_lb.to(self.device)

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        logits = self.train_model(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        del logits, inputs, x_lb, x_ulb_s, x_ulb_w

        sup_loss = cross_entropy_loss(logits_x_lb, y_lb, reduction="mean")
        unsup_loss, mask = consistency_loss(
            logits_x_ulb_w,
            logits_x_ulb_s,
            "ce",
            self.T,
            self.p_cutoff,
            use_hard_labels=cfg.hard_label,
        )
        total_loss = sup_loss + self.lambda_u * unsup_loss

        # parameter updates
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Update the evaluation model from training model
        with torch.no_grad():
            self._eval_model_update()
            train_accuracy = accuracy(logits_x_lb, y_lb)
            train_accuracy = train_accuracy[0].cpu()

        return train_accuracy

    @torch.no_grad()
    def evaluate(self, eval_loader=None):
        """Evaluate the model performance.

        Args:
            eval_loader (_type_, optional): Arbitrary data set to be used, if none, the test set is used. Defaults to None.

        Returns:
            _type_: _description_
        """
        # empty cache (without entering context of current gpu, gpu0 may be initialized)
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        use_ema = hasattr(self, "eval_model")

        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.to(self.device)
        eval_model.eval()

        if eval_loader is None:
            eval_loader = self.loader_dict["eval"]

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        for x, y in eval_loader:
            if torch.cuda.is_available():
                x, y = x.to(self.device), y.to(self.device)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)

            total_loss += loss.detach() * num_batch
            total_acc += acc.detach()

        if not use_ema:
            eval_model.train()

        loss = total_loss.detach().cpu() / total_num
        acc = total_acc.detach().cpu() / total_num

        return loss, acc
