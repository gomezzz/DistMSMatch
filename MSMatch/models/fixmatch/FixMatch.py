import torch
import torch.nn.functional as F

import os
from tqdm import tqdm
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
        num_eval_iter=1000,
        tb_log=None,
        logger=None,
    ):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.

        Args:
            net_builder: backbone network class (see get_net_builder in utils)
            num_classes: # of label classes
            in_channels: number of image channels
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see get_logger.py)
        """
        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # note there is a separate eval model due to exponential moving average of the weights
        self.train_model = net_builder(num_classes=num_classes, in_channels=in_channels)
        self.eval_model = net_builder(num_classes=num_classes, in_channels=in_channels)
        self.num_eval_iter = num_eval_iter
        self.T = T  # temperature params function
        self.p_cutoff = p_cutoff  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
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
        for param_train, param_eval in zip(train_model_params, self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f"[!] data loader keys: {self.loader_dict.keys()}")

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, cfg, progressbar=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        # lb: labeled, ulb: unlabeled
        self.train_model.train()

        total_epochs = cfg.num_train_iter // cfg.num_eval_iter
        curr_epoch = 0
        progressbar = tqdm(desc=f"Epoch {curr_epoch}/{total_epochs}", total=cfg.num_eval_iter)

        best_eval_acc, best_it = 0.0, 0

        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, _) in zip(
            self.loader_dict["train_lb"], self.loader_dict["train_ulb"]
        ):
            # prevent the training iterations exceed cfg.num_train_iter
            if self.it > cfg.num_train_iter:
                break

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            if torch.cuda.is_available():
                x_lb, x_ulb_w, x_ulb_s = (
                    x_lb.cuda(cfg.gpu),
                    x_ulb_w.cuda(cfg.gpu),
                    x_ulb_s.cuda(cfg.gpu),
                )
                y_lb = y_lb.cuda(cfg.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            logits = self.train_model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            del logits

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
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()
                train_accuracy = accuracy(logits_x_lb, y_lb)
                train_accuracy = train_accuracy[0]

            # tensorboard_dict update
            tb_dict = {}
            tb_dict["train/sup_loss"] = sup_loss.detach()
            tb_dict["train/unsup_loss"] = unsup_loss.detach()
            tb_dict["train/total_loss"] = total_loss.detach()
            tb_dict["train/mask_ratio"] = 1.0 - mask.detach()
            tb_dict["lr"] = self.optimizer.param_groups[0]["lr"]
            tb_dict["train/top-1-acc"] = train_accuracy

            progressbar.set_postfix_str(f"Total Loss={total_loss.detach():.3e}")
            progressbar.update(1)

            if self.it % self.num_eval_iter == 0:
                progressbar.close()
                curr_epoch += 1

                eval_dict = self.evaluate(cfg=cfg)
                tb_dict.update(eval_dict)

                if tb_dict["eval/top-1-acc"] > best_eval_acc:
                    best_eval_acc = tb_dict["eval/top-1-acc"]
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters"
                )

                progressbar = tqdm(
                    desc=f"Epoch {curr_epoch}/{total_epochs}", total=cfg.num_eval_iter
                )

            if not cfg.multiprocessing_distributed or (
                cfg.multiprocessing_distributed and cfg.rank % ngpus_per_node == 0
            ):
                if self.it == best_it:
                    self.save_run("model_best.pth", cfg.save_path, cfg=None)

                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            if self.it > 2**19:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(cfg=cfg)
        eval_dict.update({"eval/best_acc": best_eval_acc, "eval/best_it": best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, cfg=None):
        torch.cuda.empty_cache()
        use_ema = hasattr(self, "eval_model")

        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict["eval"]

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        for x, y in eval_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(cfg.gpu), y.cuda(cfg.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            loss = F.cross_entropy(logits, y.long(), reduction="mean")
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)

            total_loss += loss.detach() * num_batch
            total_acc += acc.detach()

        if not use_ema:
            eval_model.train()

        return {
            "eval/loss": total_loss / total_num,
            "eval/top-1-acc": total_acc / total_num,
        }

    def save_run(self, save_name, save_path, cfg=None):
        save_filename = os.path.join(save_path, save_name)

        # Create subfolder if it does not exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        train_model = (
            self.train_model.module if hasattr(self.train_model, "module") else self.train_model
        )
        eval_model = (
            self.eval_model.module if hasattr(self.eval_model, "module") else self.eval_model
        )
        torch.save(
            {
                "train_model": train_model.state_dict(),
                "eval_model": eval_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "it": self.it,
            },
            save_filename,
        )

        if cfg is not None:
            save_cfg(cfg)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        train_model = (
            self.train_model.module if hasattr(self.train_model, "module") else self.train_model
        )
        eval_model = (
            self.eval_model.module if hasattr(self.eval_model, "module") else self.eval_model
        )

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if "train_model" in key:
                    train_model.load_state_dict(checkpoint[key])
                elif "eval_model" in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == "it":
                    self.it = checkpoint[key]
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
