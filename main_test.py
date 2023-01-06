import sys
sys.path.append("..")

# Main imports
import asyncio
import torch
import MSMatch as mm
import numpy as np
import pykep as pk
from loguru import logger

def sync_nodes(nodes:list[mm.BaseNode]):
    t_n = [node.local_time() for node in nodes] # get local time for each node
    extra_time = [np.max(t_n) - t for t in t_n] # find out how much each node must proceed to sync
    for i in range(len(nodes)):
        nodes[i].advance_time(extra_time[i])
        
    # all nodes must step in time before we update connections for the nodes    
    for node in nodes:
        node.update_los_connections()


async def main_loop(nodes: list[mm.BaseNode], cfg, logger):
    
    num_kb_to_tx = len(mm.model_to_bytestream(nodes[0].model.train_model))*8
    
    # Make sure nodes are aware of each other
    for node in nodes:
        nodes_exl_self = list(set(nodes) - set([node]))
        node.add_actors(nodes_exl_self)
    
    
    for r in range(cfg.training_rounds):
        logger.info(f"Training round {r}")
        # Train models locally
        local_models = {}
        for node in nodes:
            node.paseos.perform_activity("Train", [cfg])
            await node.wait_for_activity()
            local_models[node.name] = node.model.train_model
        
        sync_nodes(nodes)
        
        # Aggregate models between neighbors
        for i, node in enumerate(nodes):
            tx_rate_bps = 1000 * node.paseos.local_actor.communication_devices['link'].bandwidth_in_kbps
            tx_duration = num_kb_to_tx / tx_rate_bps
            
            nodes_exl_self = list(set(nodes) - set([node]))
            connected_nodes = node.get_los_connections()
            
            neighbor_models = []
            neighbors = []
            for n in nodes_exl_self:
                if (n.name in connected_nodes) and connected_nodes[n.name]: # there is a line-of-sight
                    if node.is_tx_feasible(n, tx_duration):
                        neighbor_models.append(local_models[n.name])
                        neighbors.append(n.name)
            if len(neighbor_models)>0:
                node.aggregate(neighbor_models)
                node.advance_time(tx_duration)
            logger.info(f"Node {node.node_indx} merged with {neighbors}")
        sync_nodes(nodes)
    
    for node in nodes:
        node.save_history()
    

if __name__ == '__main__':
    
    # import matplotlib.pyplot as plt
    # acc = np.zeros((16,100))
    # for node in range(16):
    #     path = f"./results/node {node}.npy"
    #     acc[node,:] = np.load(path)
    #     plt.plot(range(acc.shape[1]), acc[node,:], label=f"sat{node}")
    
    # plt.legend()
    # plt.xlabel("Training round")
    # plt.ylabel("Test accuracy")
    # plt.title("Walker (16 sats, 4 planes, 30 deg incl), 100 iterations/round")
    # plt.grid()
    
    
    cfg_path=None
    # We use a cfg DotMap (a dictionary with dot accessors) to store the configuration for the run
    cfg=mm.load_cfg(cfg_path)
    if torch.cuda.is_available():
        cfg.gpu = 0

    # Set seeds for reproducibility and enable loggers
    logger.remove()
    mm.set_seeds(cfg.seed)
    logger_level = "INFO"
    logger = mm.get_logger(cfg.save_path, logger_level)

    # Number of training iterations per round is based on local epochs and how regularly we evaluate the model.
    # Note that batch size here only refers to the supervised part, so the real batch size
    # is cfg.batch_size * (1 + cfg.ulb_ratio)
    cfg.num_train_iter = cfg.local_epochs * cfg.num_eval_iter * 32 // cfg.batch_size
    
    # Create dataloaders for all nodes
    node_dls, cfg = mm.create_node_dataloaders(cfg)
    
    # get constellations params
    altitude = 400 * 1000 # Constellation attitude above the Earth's ground
    inclination = 30.0 # inclination of each orbital plane
    nPlanes = cfg.planes # the number of orbital planes (see linked wiki article)
    nSats = cfg.nodes // nPlanes # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2022-Oct-27 21:00:00") # the starting date of our simulation
    
    # Compute orbits of LEO satellites
    planet_list,sats_pos_and_v = mm.get_constellation(altitude,inclination,nSats,nPlanes,t0)

    # Create nodes
    nodes = []
    for i in range(cfg.nodes):
        node = mm.PaseosNode(i, sats_pos_and_v[i],
            cfg,
            node_dls[i],
            logger)
        nodes.append(node)

    # do training
    asyncio.run(main_loop(nodes, cfg, logger))
    