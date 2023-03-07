import sys
import os
sys.path.append("..")

# Main imports
import MSMatch as mm
import numpy as np
import pykep as pk
from loguru import logger
import paseos
from mpi4py import MPI


def main_loop():

    # Get MPI object
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # get process number of current instance
    size = comm.Get_size()

    # Load config file
    cfg = mm.load_cfg(cfg_path=None)
    cfg.save_path = cfg.sim_path + f"/node{rank}"   
    mm.set_seeds(cfg.seed)  # set seed

    # make sure that each satellite has a process
    #    assert (size == cfg.nodes), "number of satellites should equal number of processes"
    print(f"Starting rank {rank}", flush=True)

    # -------------------------------
    # PASEOS setup
    # -------------------------------
    # Create dataloaders for all nodes
    node_dls, cfg = mm.create_node_dataloaders(cfg)
    print(f"HERE rank{rank}", flush=True)
    # Create coordinate/position for each node
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62  # inclination of the orbit
    nPlanes = cfg.planes  # the number of orbital planes (see linked wiki article)
    nSats = cfg.nodes // nPlanes  # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2023-Dec-17 14:42:42")  # starting date of our simulation
    # Compute orbits of LEO satellites
    planet_list, sats_pos_and_v = mm.get_constellation(
        altitude, inclination, nSats, nPlanes, t0
    )

    # create parameter server if needed
    if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
        node = mm.SpaceCraftNode(
            sats_pos_and_v[rank], cfg, node_dls[rank], comm, logger
        )
        # orbit_duration = planet_list[0].compute_period(t0)
        
        # Get position and velocity for geostationary satellite
        planet_list, sats_pos_and_v = mm.get_constellation(
            altitude=35786 * 1000,
            inclination=0,
            nSats=1,
            nPlanes=1,
            t0=t0,
            startingW=31,
        )
        # create server node
        server_node = mm.ServerNode(cfg, list(range(size)), sats_pos_and_v)
        
        # For FL, the server operates on rank = 0
        if rank == 0:
            os.makedirs(cfg.sim_path)
            server_node.broadcast_global_model()
            
        comm.Barrier()  # make sure all nodes wait for the global model to be stored
        node.set_server_node(server_node) # make each know about the server
        node.get_global_model()  # load the global model into the node

    else:
        os.makedirs(cfg.save_path)
        # Create spacecraft node
        node = mm.SpaceCraftNode(
            sats_pos_and_v[rank], cfg, node_dls[rank], comm, logger
        )
        server_node = None

    # ------------------------------------
    # Enter main loop
    # ------------------------------------
    time_in_standby = 0 # how long has the node been in standby
    time_since_last_update = 0 # how long ago since last model received
    time_per_batch = 16 # time duration to train a single batch
    time_for_comms = node.comm_duration # duration to share a model 
    standby_period = 1e3  # how long to standby if necessary
    model_update_countdown = -1 # this is a counter used to remain in "Model update" 
    
    # Allocate for storing results
    test_losses = []
    test_accuracy = []
    train_accuracy = []
    local_time_at_train = []
    local_time_at_test = []
    communication_times = []
    communication_over_times = []
    
    verbose = False # Print what is going on

    total_time = 15 * 24 * 3600  # total simulation time
    batch_idx = 0  # starting batch index
    sim_time = 0 # simulation time
    
    paseos.set_log_level("INFO")
    while sim_time <= total_time:
        
        # print current step
        if rank == 0:
            print(f"batch: {batch_idx}, time: {sim_time}", flush=True, end="\r")
            
        sim_time = node.paseos._state.time # keep track of simulation time

        # if the model countdown is positive, we are still communicating and should remain in model update
        if model_update_countdown > 0:
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

            comm.Barrier()  # sync all models in time
            if cfg.mode == "Swarm":
                node.exchange_actors(
                    verbose=False
                )  # Find out what actors can be seen and what the other actors are doing
            else:
                node.check_if_sever_available()  # check if server is visible and add to known actors
                if node.rank == 0:
                    server_node.update_global_model() # attempt to update global model (only if buffer contains > 2 unique models)

        # run the current activity (Model update, training, or standby)
        if activity == "Model_update":
            # if counter is not counting down
            if model_update_countdown == -1:
                if verbose == True:
                    print(
                        f"Node{rank} will update with {node.paseos.known_actor_names} at {sim_time}",
                        flush=True,
                    )

                # initiate communications
                communication_times.append(sim_time) # store time for communication
                model_update_countdown = (
                    2 * time_for_comms // time_per_batch
                )  # transmit to two neighbors (swarm) or uplink/downlink (FL)
 
            else:
                model_update_countdown -= 1 # decrease counter
                # the communications is over
                if model_update_countdown < 0:
                    if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
                        model_loaded = (
                            node.get_global_model()
                        )  # get current global model
                        if model_loaded:
                            time_since_last_update = 0
                        else:
                            # make sure to get back in here next iteration because something went wrong loading
                            model_update_countdown += 1
                    else:
                        node.aggregate() # aggregate received models (loaded from folders)
                        time_since_last_update = 0
                    loss, acc = node.evaluate() # evaluate updated model
                    
                    # Store values
                    communication_over_times.append(sim_time)
                    test_losses.append(loss)
                    test_accuracy.append(acc)
                    local_time_at_test.append(sim_time)

                    print(
                        f"Rank {node.rank}, post aggregation: eval acc: {acc}",
                        flush=True,
                    )
                # the client model reached the server node (FL)
                elif model_update_countdown == time_for_comms // time_per_batch + 1:
                    if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
                        node.save_model()  # "communicate" the local model to the server by storing it to a folder

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

            # evaluate the model and store results
            if batch_idx % 50 == 0 or batch_idx == 0:
                loss, acc = node.evaluate()
                test_losses.append(loss)
                test_accuracy.append(acc)
                local_time_at_test.append(sim_time)
                if rank == 0:
                    print(
                        f"Simulation time: {sim_time} "
                        + f"Rank {node.rank}, batch_idx {batch_idx}: eval acc: {acc} "
                        + f"Temp: {node.local_actor.temperature_in_K - 273.15:.2f} "
                        + f"Battery SoC: {node.local_actor.state_of_charge:.2f}",
                        flush=True,
                    )

                if cfg.mode == "Swarm":
                    node.save_model()  # save model to folder
        else:
            # Standby
            node.perform_activity(activity, power_consumption, time_per_batch)
        
        # increase time since the server node updated global model
        if cfg.mode == "FL_ground" or cfg.mode == "FL_geostat":
            server_node.update_time_since_update(time_per_batch)

    # -------------------------------------------------
    # Save things to become a happy camper
    np.savetxt(f"{cfg.save_path}/test_loss.csv", np.array(test_losses), delimiter=",")
    np.savetxt(f"{cfg.save_path}/test_acc.csv", np.array(test_accuracy), delimiter=",")
    np.savetxt(
        f"{cfg.save_path}/test_time.csv", np.array(local_time_at_test), delimiter=","
    )
    np.savetxt(
        f"{cfg.save_path}/train_acc.csv", np.array(train_accuracy), delimiter=","
    )
    np.savetxt(
        f"{cfg.save_path}/train_time.csv", np.array(local_time_at_train), delimiter=","
    )
    np.savetxt(
        f"{cfg.save_path}/comm_time.csv", np.array(communication_times), delimiter=","
    )
    np.savetxt(
        f"{cfg.save_path}/post_comm_time.csv",
        np.array(communication_over_times),
        delimiter=",",
    )
    node.paseos.save_status_log_csv(f"{cfg.save_path}/paseos_data.csv")


if __name__ == "__main__":
    main_loop()
