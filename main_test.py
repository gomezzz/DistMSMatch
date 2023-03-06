import sys
sys.path.append("..")

# Main imports
import MSMatch as mm
import numpy as np
import pykep as pk
from loguru import logger
import time
import paseos
from mpi4py import MPI



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
    # Number of training iterations per round (number of times we sample a batch) is based on local epochs and how regularly we evaluate the model.
    # Note that batch size here only refers to the supervised part, so the real batch size
    # is cfg.batch_size * (1 + cfg.ulb_ratio)
    cfg.num_train_iter = (cfg.lb_epochs * cfg.num_labels) // cfg.batch_size
    # Create dataloaders for all nodes
    node_dls, cfg = mm.create_node_dataloaders(cfg)
    
        # Create node
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if cfg.mode == "Swarm":
        cfg.sim_path = cfg.save_dir + f"ISL/{timestr}"
    elif cfg.mode == "FL_ground":
        cfg.sim_path = cfg.save_dir + f"FL_ground/{timestr}"
    elif cfg.mode == "FL_geostat":
        cfg.sim_path = cfg.save_dir + f"FL_geostat/{timestr}"
        
    cfg.save_path = cfg.sim_path + f"/node{rank}"
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62  # inclination of the orbit
    nPlanes = cfg.planes # the number of orbital planes (see linked wiki article)
    nSats = cfg.nodes // nPlanes # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2023-Dec-17 14:42:42")  # starting date of our simulation
    # Compute orbits of LEO satellites
    planet_list,sats_pos_and_v = mm.get_constellation(altitude, inclination, nSats, nPlanes, t0)
    

    
    # create parameter server if needed
    if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
        node = mm.SpaceCraftNode(sats_pos_and_v[2*rank], cfg, node_dls[2*rank], comm, logger)
        # orbit_duration = planet_list[0].compute_period(t0)
        
        planet_list, sats_pos_and_v = mm.get_constellation(altitude=35786 * 1000,inclination=0,nSats=1,nPlanes=1,t0=t0,startingW=31)
        server_node = mm.ServerNode(cfg, list(range(size)), sats_pos_and_v)
        if rank == 0:
            server_node.broadcast_global_model()
        comm.Barrier() # make sure all nodes wait for the global model to be stored
        node.set_server_node(server_node)
        node.get_global_model() # receive the global model to start from
        
    else:
        node = mm.SpaceCraftNode(sats_pos_and_v[rank], cfg, node_dls[rank], comm, logger)
        server_node = None
        
    #------------------------------------
    # Enter main loop
    #------------------------------------
    time_in_standby = 0
    time_since_last_update = 0
    time_per_batch = 16
    time_for_comms = node.comm_duration
    standby_period = 1e3  # how long to standby if necessary
    model_update_countdown = -1
    test_losses = []
    test_accuracy = []
    train_accuracy = []
    local_time_at_train = []
    local_time_at_test = []
    communication_times = []
    communication_over_times = []
    Verbose = False
    
    total_time = 15*24*3600 # total number of training rounds
    batch_idx = 0 # starting round
    sim_time = 0
    paseos.set_log_level("INFO")
    while sim_time <= total_time:
        sim_time = node.paseos._state.time
        
        if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
            server_node.time_since_last_global_update += time_per_batch
        
        if rank == 0:
            print(f"batch: {batch_idx}, time: {sim_time}", flush=True, end="\r")
        
        if model_update_countdown > 0:
            # if we have shared models, we make sure to do the 
            activity = "Model_update"
            power_consumption = 10
        else:    
            # Find out what kind of activity to perform
            activity, power_consumption, time_in_standby = node.decide_on_activity(
                time_per_batch,
                time_in_standby,
                standby_period,
                time_since_last_update,
            )
            
            comm.Barrier() # sync all models in time
            if cfg.mode == "Swarm":
                node.exchange_actors(verbose=False)  # Find out what actors can be seen and what the other actors are doing
            else:
                node.comm_to_server() # find out if we can communicate to a server node
                # aggregate global model from buffered models
                if  node.rank == 0:
                    server_node.update_global_model()
                    
        # run the current activity (Model update, training, or standby)
        if activity == "Model_update":
            if model_update_countdown == -1:
                if Verbose == True:
                    print(
                        f"Node{rank} will update with {node.paseos.known_actor_names} at {sim_time}", flush=True
                    )
                    
                # update the node model
                if cfg.mode == "Swarm":
                    node.aggregate()
                    communication_times.append(sim_time)
                    model_update_countdown = 2*time_for_comms // time_per_batch # transmit to two neighbors
                else:
                    # upload current local model and receive new global
                    communication_times.append(sim_time)
                    model_update_countdown = 2*time_for_comms // time_per_batch # transmit and receive form server

            else:
                model_update_countdown -= 1
                # the communications for updating is over
                if model_update_countdown < 0:
                    if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
                        model_loaded = node.get_global_model() # get current global model
                        if model_loaded:
                            time_since_last_update = 0
                        else:
                            # make sure to get back in here next iteration
                            model_update_countdown += 1
                    else:
                        time_since_last_update = 0
                    communication_over_times.append(sim_time)
                    loss, acc = node.evaluate()
                    test_losses.append(loss)
                    test_accuracy.append(acc)
                    local_time_at_test.append(sim_time)
                    
                    
                    print(f"Rank {node.rank}, post aggregation: eval acc: {acc}", flush=True)
                # the client model reached the server node
                elif model_update_countdown == time_for_comms // time_per_batch + 1:
                    if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
                        node.save_model() # "communicate" the local model to the server
                    
            # increase time for communication
            node.perform_activity(activity, power_consumption, time_per_batch)
        
        elif activity == "Training":
            # record activity in paseos
            node.perform_activity(activity, power_consumption, time_per_batch)
            time_since_last_update += time_per_batch
            
            # perform training for one iteration only
            train_acc = node.train_one_batch()
            train_accuracy.append(train_acc)
            local_time_at_train.append(sim_time)
            batch_idx += 1
            
            if batch_idx % 50 == 0 or batch_idx == 0:
                loss, acc = node.evaluate()
                test_losses.append(loss)
                test_accuracy.append(acc)
                local_time_at_test.append(sim_time)
                if rank == 0:
                    print(f"Simulation time: {sim_time} "
                        + f"Rank {node.rank}, batch_idx {batch_idx}: eval acc: {acc} "
                        + f"Temp: {node.local_actor.temperature_in_K - 273.15:.2f} "
                        + f"Battery SoC: {node.local_actor.state_of_charge:.2f}", flush=True)
            
                if cfg.mode == "Swarm":
                    node.save_model() # save model to folder
        else:
            # Standby
            node.perform_activity(activity, power_consumption, time_per_batch)
        
    # Save things to become a happy camper
    np.savetxt(f"{cfg.save_path}/test_loss.csv", np.array(test_losses), delimiter=",")
    np.savetxt(f"{cfg.save_path}/test_acc.csv", np.array(test_accuracy), delimiter=",")
    np.savetxt(f"{cfg.save_path}/test_time.csv", np.array(local_time_at_test), delimiter=",")
    np.savetxt(f"{cfg.save_path}/train_acc.csv", np.array(train_accuracy), delimiter=",")
    np.savetxt(f"{cfg.save_path}/train_time.csv", np.array(local_time_at_train), delimiter=",")
    np.savetxt(f"{cfg.save_path}/comm_time.csv", np.array(communication_times), delimiter=",")
    np.savetxt(f"{cfg.save_path}/post_comm_time.csv", np.array(communication_over_times), delimiter=",")
    node.paseos.save_status_log_csv(f"{cfg.save_path}/paseos_data.csv")
        
if __name__ == '__main__':
    main_loop()
