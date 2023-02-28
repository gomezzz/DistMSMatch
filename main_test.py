import sys
sys.path.append("..")

# Main imports
import MSMatch as mm
import numpy as np
import pykep as pk
from loguru import logger
from mpi4py import MPI
import paseos
import time


def main_loop():
    
    # Load config file
    cfg_path=None
    cfg=mm.load_cfg(cfg_path)
    mm.set_seeds(cfg.seed) # set seed
        
    # Get MPI object
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # get process number of current instance
    size = comm.Get_size()
    
    # make sure that each satellite has a process
#    assert (size == cfg.nodes), "number of satellites should equal number of processes"
    print(f"Starting rank {rank}", flush=True)
    
    # PASEOS setup
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62  # inclination of the orbit
    nPlanes = cfg.planes # the number of orbital planes (see linked wiki article)
    nSats = cfg.nodes // nPlanes # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2023-Dec-17 14:42:42")  # starting date of our simulation
    
    # Number of training iterations per round (number of times we sample a batch) is based on local epochs and how regularly we evaluate the model.
    # Note that batch size here only refers to the supervised part, so the real batch size
    # is cfg.batch_size * (1 + cfg.ulb_ratio)
    cfg.num_train_iter = (cfg.lb_epochs * cfg.num_labels) // cfg.batch_size
    # Create dataloaders for all nodes
    node_dls, cfg = mm.create_node_dataloaders(cfg)
    
    # Compute orbits of LEO satellites
    planet_list,sats_pos_and_v = mm.get_constellation(altitude, inclination, nSats, nPlanes, t0)

    # Create node
    node = mm.SpaceCraftNode(sats_pos_and_v[rank], cfg, node_dls[rank], comm, logger)

    # Ground stations
    stations = [
        ["Maspalomas", 27.7629, -15.6338, 205.1],
        ["Matera", 40.6486, 16.7046, 536.9],
        ["Svalbard", 78.9067, 11.8883, 474.0],
    ]
    
    #------------------------------------
    # Enter main loop
    #------------------------------------
    time_in_standby = 0
    time_since_last_update = 0
    time_per_batch = 0.3
    time_for_comms = node.comm_duration
    standby_period = 900  # how long to standby if necessary
    model_update_countdown = 0
    test_losses = []
    test_accuracy = []
    local_time_at_test = []
    communication_times = []
    communication_over_times = []
    
    total_batches = 50 # total number of training rounds
    batch_idx = 0 # starting round
    sim_time = 0
    paseos.set_log_level("INFO")
    while batch_idx <= total_batches:
        sim_time += time_per_batch
        if batch_idx % 10 == 0:
            print(
                f"Rank {rank} - Temperature[C]: "
                + f"{node.local_actor.temperature_in_K - 273.15:.2f},"
                + f"Battery SoC: {node.local_actor.state_of_charge:.2f}", flush=True
            )
        
        
        if model_update_countdown > 0:
            # if we have shared models, we make sure to do the 
            activity = "Model"
            power_consumption = 10
        else:    
            # Find out what kind of activity to perform
            activity, power_consumption, time_in_standby = node.decide_on_activity(
                time_per_batch,
                time_in_standby,
                standby_period,
                time_since_last_update,
            )
        
        # run the current activity (Model update, training, or standby)
        if activity == "Model_update":
            if model_update_countdown == 0:
                print(
                    f"Node{rank} will update with {node.paseos.known_actor_names()} at {node.local_actor.local_time}"
                )
                
                # sync on actor positions and update on line-of-sights
                node.exchange_actors() 
                node.aggregate()
                communication_times.append(sim_time)
                
                loss, acc = node.evaluate()
                test_losses.append(loss)
                test_accuracy.append(acc)
                local_time_at_test.append(node.local_actor.local_time)
                node.logger.info(f"Rank {node.rank}, post aggregation: eval acc: {acc}")
                
                time_since_last_update = 0
                model_update_countdown = time_for_comms // time_per_batch
            else:
                model_update_countdown = np.maximum(0, model_update_countdown-1)
                if model_update_countdown == 0:
                    communication_over_times.append(sim_time)
            
            # increase time for communication
            node.perform_activity(activity, power_consumption, time_per_batch)
        
        elif activity == "Training":
            # record activity in paseos
            node.perform_activity(activity, power_consumption, time_per_batch)
            time_since_last_update += time_per_batch
            
            # perform training for one iteration only
            node.train_one_batch()
            batch_idx += 1
            
            if batch_idx % 10 == 0:
                loss, acc = node.evaluate()
                test_losses.append(loss)
                test_accuracy.append(acc)
                local_time_at_test.append(sim_time)
                node.logger.info(f"Rank {node.rank}, batch_idx {batch_idx}: eval acc: {acc}")
                
                node.save_model() # save model to folder
        else:
            # Standby
            node.perform_activity(activity, power_consumption, time_per_batch)
        
        comm.Barrier() # sync all models in time
    
    # Save things to become a happy camper
    node.paseos.save_status_log_csv(f"{cfg.save_dir}/paseos_rank{rank}.csv")
    np.savetxt(f"{cfg.save_dir}/loss_rank{rank}.csv", np.array(test_losses), delimiter=",")
    np.savetxt(f"{cfg.save_dir}/acc_rank{rank}.csv", np.array(test_accuracy), delimiter=",")
    np.savetxt(f"{cfg.save_dir}/test_time_rank{rank}.csv", np.array(local_time_at_test), delimiter=",")
    np.savetxt(f"{cfg.save_dir}/comm_time_rank{rank}.csv", np.array(communication_times), delimiter=",")
    np.savetxt(f"{cfg.save_dir}/post_comm_time_rank{rank}.csv", np.array(communication_over_times), delimiter=",")

        
if __name__ == '__main__':
    main_loop()
