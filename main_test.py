import sys
sys.path.append("..")

# Main imports
import torch
import MSMatch as mm
from termcolor import colored

cfg_path=None
# We use a cfg DotMap (a dictionary with dot accessors) to store the configuration for the run
cfg=mm.load_cfg(cfg_path)

# Set seeds for reproducibility and enable loggers
mm.set_seeds(cfg.seed)
logger_level = "INFO"
logger = mm.get_logger(cfg.save_path, logger_level)
tb_log = mm.TensorBoardLog(cfg.save_path, "")

# Construct Dataset
print("Loading "+colored("train", "red")+ " dataset...")

# Create SSL object
train_dset = mm.SSL_Dataset(
    name=cfg.dataset, train=True, data_dir=None, seed=cfg.seed, alpha=cfg.alpha, nodes=cfg.nodes
)


lb_dset, ulb_dset = train_dset.get_ssl_dset(cfg.num_labels)

cfg.num_classes = train_dset.num_classes
cfg.num_channels = train_dset.num_channels

print("Loading "+colored("eval", "blue")+ " dataset...")
_eval_dset = mm.SSL_Dataset(
    name=cfg.dataset, train=False, data_dir=None, seed=cfg.seed,
)
eval_dset = _eval_dset.get_dset()

print("Initializing ", cfg.net)
net_builder = mm.get_net_builder(
    cfg.net,
    pretrained=cfg.pretrained,
    in_channels=cfg.num_channels,
    scale=cfg.scale
)

model = mm.FixMatch(
        net_builder,
        cfg.num_classes,
        cfg.num_channels,
        cfg.ema_m,
        T=cfg.T,
        p_cutoff=cfg.p_cutoff,
        lambda_u=cfg.ulb_loss_ratio,
        hard_label=True,
        num_eval_iter=cfg.num_eval_iter,
        tb_log=tb_log,
        logger=logger,
    )
logger.info(f"Number of Trainable Params: {sum(p.numel() for p in model.train_model.parameters() if p.requires_grad)}")

# Number of training iterations is based on that and how regularly we evaluate the model.
# Note that batch size here only refers to the supervised part, so the real batch size
# is cfg.batch_size * (1 + cfg.ulb_ratio)
cfg.num_train_iter = cfg.epoch * cfg.num_eval_iter * 32 // cfg.batch_size

# get optimizer, ADAM and SGD are supported.
optimizer = mm.get_optimizer(
    model.train_model, cfg.opt, cfg.lr, cfg.momentum, cfg.weight_decay
)
# We use a learning rate schedule to control the learning rate during training.
scheduler = mm.get_cosine_schedule_with_warmup(
    optimizer, cfg.num_train_iter, num_warmup_steps=cfg.num_train_iter * 0
)
model.set_optimizer(optimizer, scheduler)

# If a CUDA capable GPU is used, we move everything to the GPU now
if torch.cuda.is_available():
    cfg.gpu = 0
    torch.cuda.set_device(cfg.gpu)
    model.train_model = model.train_model.cuda(cfg.gpu)
    model.eval_model = model.eval_model.cuda(cfg.gpu)

logger.info(f"model_arch: {model}")
logger.info(f"Arguments: {cfg}")

loader_dict = {}
dset_dict = {"train_lb": lb_dset, "train_ulb": ulb_dset, "eval": eval_dset}

loader_dict["train_lb"] = mm.get_data_loader(
    dset_dict["train_lb"],
    cfg.batch_size,
    data_sampler="RandomSampler",
    num_iters=cfg.num_train_iter,
    num_workers=1,
    distributed=False,
)

loader_dict["train_ulb"] = mm.get_data_loader(
    dset_dict["train_ulb"],
    cfg.batch_size * cfg.uratio,
    data_sampler="RandomSampler",
    num_iters=cfg.num_train_iter,
    num_workers=4,
    distributed=False,
)

loader_dict["eval"] = mm.get_data_loader(
    dset_dict["eval"], cfg.eval_batch_size, num_workers=1
)

## set DataLoader on FixMatch
model.set_data_loader(loader_dict)


trainer = model.train
trainer(cfg)

# Evaluate the final model
model.evaluate(cfg=cfg)

model.save_run("latest_model.pth", cfg.save_path, cfg)