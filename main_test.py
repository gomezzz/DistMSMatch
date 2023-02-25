import sys
sys.path.append("..")

# Main imports
import asyncio
import torch
import MSMatch as mm
import numpy as np
import pykep as pk
from loguru import logger
from mpi4py import MPI


async def main_loop(node, cfg, comm, other_ranks, logger):
    
    simulation_time = 24.0 * 3600  # simulation interval in seconds
    t = 0 # starting time in seconds
    timestep = 1800  # how often we synchronize actors' trajectories
    comm.Barrier()
    while t <= simulation_time:
        
        if node.do_training:      
            node.paseos.perform_activity("Train", [cfg])
            await node.paseos.wait_for_activity()      
            node.do_training = False 
        
        comm.Barrier()
        node.exchange_actors(comm, other_ranks, verbose=True)
        node.exchange_models_and_aggregate(comm, verbose=False)
        comm.Barrier()
        
        node.advance_time(timestep)
        t += timestep
        
        if node.rank == 0: 
            logger.info(f"-----------------------------------")
        
        
if __name__ == '__main__':
    
    # import matplotlib.pyplot as plt
    # acc = np.zeros((16,100))
    # for node in range(16):
    #     path = f"./results/node {node}.npy"
    #     acc[node,:] = np.load(path)[:100]
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
        
    # Get MPI object
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    
    assert (size == cfg.nodes), "number of satellites should equal number of processes"
    
    rank = comm.Get_rank()
    other_ranks = [x for x in range(size) if x != rank]
    print(f"Started rank {rank}, other ranks are {other_ranks}")
    


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
    altitude = 600 * 1000 # Constellation attitude above the Earth's ground
    inclination = 30.0 # inclination of each orbital plane
    nPlanes = cfg.planes # the number of orbital planes (see linked wiki article)
    nSats = cfg.nodes // nPlanes # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2022-Oct-27 21:00:00") # the starting date of our simulation
    
    # Compute orbits of LEO satellites
    planet_list,sats_pos_and_v = mm.get_constellation(altitude,inclination,nSats,nPlanes,t0)

    # Create node
    node = mm.PaseosNode(rank, sats_pos_and_v[rank], cfg, node_dls[rank], logger)

    # do training
    asyncio.run(main_loop(node, cfg, comm, other_ranks, logger))
    