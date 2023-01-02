import sys
sys.path.append("..")

# Main imports
import asyncio
import torch
import MSMatch as mm
import numpy as np

def sync_nodes(nodes:list[mm.Node]):
    t_n = [node.local_time() for node in nodes] # get local time for each node
    extra_time = [np.maximum(t_n) - t for t in t_n] # find out how much each node must proceed to sync
    map(lambda n,t: n.advance_time(t), zip(nodes, extra_time)) # advance time of each node

async def main_loop(nodes: list[mm.Node], cfg):
    
    # Start by training all models locally
    for node in nodes:
        node.paseos.perform_activity("Train")
        await node.wait_for_activity()

        node.paseos.perform_activity("Evaluate")
        await node.wait_for_activity()
    
    sync_nodes(nodes)
    
       
        
        

if __name__ == '__main__':
    cfg_path=None
    # We use a cfg DotMap (a dictionary with dot accessors) to store the configuration for the run
    cfg=mm.load_cfg(cfg_path)
    if torch.cuda.is_available():
        cfg.gpu = 0

    # Set seeds for reproducibility and enable loggers
    mm.set_seeds(cfg.seed)
    logger_level = "INFO"
    logger = mm.get_logger(cfg.save_path, logger_level)
    
    # Number of training iterations per round is based on local epochs and how regularly we evaluate the model.
    # Note that batch size here only refers to the supervised part, so the real batch size
    # is cfg.batch_size * (1 + cfg.ulb_ratio)
    cfg.num_train_iter = cfg.local_epochs * cfg.num_eval_iter * 32 // cfg.batch_size
    
    # Create dataloaders for all nodes
    node_dls, cfg = mm.create_node_dataloaders(cfg)
    
    # Create nodes
    nodes = []
    for i in range(cfg.nodes):
        node = mm.Node(i, 
            cfg,
            node_dls[i],
            logger)
        nodes.append(node)

    # do training
    asyncio.run(main_loop(nodes))
    