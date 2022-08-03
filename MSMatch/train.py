
# Main imports
import sys 
sys.path.append("..")
import torch
import MSMatch as mm
from utils.create_dir_string import create_dir_str
import os
import argparse
from termcolor import colored

# We use a cfg DotMap (a dictionary with dot accessors) to store the configuration for the run

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, help="Number of labels", default=None)
    parser.add_argument('--wd', type=float, help='Weight decay value', default=None)
    parser.add_argument('--lr', type=float, help='Learning rate', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=None)
    parser.add_argument('--scale', type=float, help='Params scale factor', default=None)
    parser.add_argument('--epoch', type=int, help='Number of epochs.', default=None)
    parser.add_argument('--save_dir', type=str, help='Path to the save directory', default="../notebooks/saved_models")
    parser.add_argument('--net', type=str, help='Network. Supported ""efficientnet"", ""unet"", ""efficientnet-lite0""', default=None)
    parser.add_argument('--seed', type=int, help='Simulation seed.', default=None)
 
    pargs=parser.parse_args()

    cfg = mm.get_default_cfg()
    if pargs.num_labels is not None:
        print("Using number of labels per class: "+colored(pargs.num_labels, "red"))
        cfg.num_labels=pargs.num_labels

    if pargs.wd is not None:
        print("Using weight decay: "+colored(pargs.wd, "red"))
        cfg.weight_decay=pargs.wd

    if pargs.lr is not None:
        print("Using learnig rate: "+colored(pargs.lr, "red"))
        cfg.lr=pargs.lr

    if pargs.batch_size is not None:
        print("Using batch size: "+colored(pargs.batch_size, "red"))
        cfg.batch_size =pargs.batch_size

    if pargs.scale is not None:
        print("Using scale factor: "+colored(pargs.scale, "red"))
        cfg.scale=pargs.scale

    if pargs.net is not None:
        print("Using network: "+colored(pargs.net, "red"))
        cfg.net=pargs.net

    if pargs.seed is not None:
        print("Using seed: "+colored(pargs.seed, "red"))
        cfg.seed=pargs.seed

    cfg.save_dir=pargs.save_dir
    #Update save name and save path
    dir_name = create_dir_str(cfg)
    
    cfg.save_name=os.path.join(cfg.save_name_root, dir_name)
    cfg.save_path = os.path.join(cfg.save_dir, cfg.save_name)

    # Set seeds for reproducibility and enable loggers
    mm.set_seeds(cfg.seed)
    logger_level = "INFO"
    logger = mm.get_logger(cfg.save_name, cfg.save_path, logger_level)
    tb_log = mm.TensorBoardLog(cfg.save_path, "")

    # Construct Dataset
    train_dset = mm.SSL_Dataset(name=cfg.dataset, train=True, data_dir=None, seed=cfg.seed,)
    lb_dset, ulb_dset = train_dset.get_ssl_dset(cfg.num_labels)

    cfg.num_classes = train_dset.num_classes
    cfg.num_channels = train_dset.num_channels

    _eval_dset = mm.SSL_Dataset(name=cfg.dataset, train=False, data_dir=None, seed=cfg.seed,)
    eval_dset = _eval_dset.get_dset()

    # Network initialization
    print("Initializing ", cfg.net)
    net_builder = mm.get_net_builder(
    cfg.net,
    pretrained=cfg.pretrained,
    in_channels=cfg.num_channels,
    scale=cfg.scale)
    model = mm.FixMatch(
        net_builder,
        cfg.num_classes,
        cfg.num_channels,
        cfg.ema_m,
        T=0.5,
        p_cutoff=cfg.p_cutoff,
        lambda_u=cfg.ulb_loss_ratio,
        hard_label=True,
        num_eval_iter=cfg.num_eval_iter,
        tb_log=tb_log,
        logger=logger,
    )
    logger.info(f"Number of Trainable Params: {sum(p.numel() for p in model.train_model.parameters() if p.requires_grad)}")

    # Specify number of epochs to train, for convergence a value like 100 may be sensible. 
    cfg.epoch =pargs.epoch

    # Number of training iterations is based on that and how regularly we evaluate the model.
    # Note that batch size here only refers to the supervised part, so the real batch size
    # is cfg.batch_size * (1 + cfg.ulb_ratio)
    cfg.num_train_iter = cfg.epoch * cfg.num_eval_iter * 32 // cfg.batch_size

    # get optimizer, ADAM and SGD are supported.
    optimizer = mm.get_optimizer(model.train_model, cfg.opt, cfg.lr, cfg.momentum, cfg.weight_decay)
    # We use a learning rate schedule to control the learning rate during training.
    scheduler = mm.get_cosine_schedule_with_warmup(optimizer, cfg.num_train_iter, num_warmup_steps=cfg.num_train_iter * 0)
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

    # Start training.
    trainer = model.train
    trainer(cfg)

    # Evaluate the final model
    model.evaluate(cfg=cfg)

    # Save final model. 
    model.save_model("latest_model.pth", cfg.save_path)


if __name__ == "__main__":
    main()
   