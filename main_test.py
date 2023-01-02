import sys
sys.path.append("..")

# Main imports
import asyncio
import torch
import copy
import MSMatch as mm

async def space_main(node: mm.Node):
    """Main loop. The actor is updating its connections in preset intervals and attempts transmit/receive each time.
    If reception successful, the client will perform federated averaging with equal weights.

    Args:
        client (SpaceClient): The local client
    """
    STEPS = 100
    t = 0

    # start by training once locally
    node.paseos.perform_activity("Train")
    await node.wait_for_activity()

    node.paseos.perform_activity("Evaluate")
    await node.wait_for_activity()

    # Start network engines to listen for new actors
    async with node.network_mngr as node:
        async with node.namespace(name="swarm") as ns:
            while t < STEPS:
                # update connections within the namespace
                node.update_connections(ns.name)

                logger.debug(f"Iteration {t}")

                # Check if there are spacecrafts within LOS, if so transmit and receive
                node.paseos.perform_activity(
                    "Transmit_Receive", activity_func_args=[ns]
                )
                await node.wait_for_activity()

                if node.model_updated is True:
                    node.paseos.perform_activity("Train")
                    await node.wait_for_activity()
                    node.paseos.perform_activity("Evaluate")
                    await node.wait_for_activity()

                await asyncio.sleep(0.1)

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
    for r in range(cfg.training_rounds):
        logger.info(f"Training round {r}")

        for node in nodes:
            node.train()

        models_before_aggr = [copy.deepcopy(n.model.train_model) for n in nodes]
        index_list = list(range(cfg.nodes))
        for i in index_list:
            loc_index_list = list(set(index_list) - set([i]))
            neighbor_models = [models_before_aggr[i] for i in loc_index_list]
            nodes[i].aggregate(neighbor_models)
        del(models_before_aggr)