import pykep as pk
import paseos
from .base_node import BaseNode
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor


class ServerNode(BaseNode):
    def __init__(
        self,
        stations,
        t0,
        comm,
        logger
    ):
        super(BaseNode, self).__init__(comm.Get_rank(), cfg=None, dataloader=None, logger=logger)
        self.comm = comm

        self.groundstation_actors = []
        for station in stations:
            gs_actor = ActorBuilder.get_actor_scaffold(
            name=station[0], actor_type=GroundstationActor, epoch=t0 )
            ActorBuilder.set_ground_station_location(
                gs_actor,
                latitude=station[1],
                longitude=station[2],
                elevation=station[3],
                minimum_altitude_angle=5,
            )
            # paseos_instance.add_known_actor(gs_actor)
            self.groundstation_actors.append(gs_actor)            
            
        
        
class SpaceCraftNode(BaseNode):
    def __init__(
        self,
        pos_and_vel,
        cfg,
        dataloader,
        comm,
        logger
    ):
        super(SpaceCraftNode, self).__init__(comm.Get_rank(), cfg, dataloader, logger)
        
        # Set up PASEOS instance
        self.earth = pk.planet.jpl_lp("earth")
        id = f"sat{self.rank}"
        sat = ActorBuilder.get_actor_scaffold(
            id, SpacecraftActor, pk.epoch(0)
        )
        ActorBuilder.set_orbit(
            sat, pos_and_vel[0], pos_and_vel[1], pk.epoch(0), self.earth
        )
        # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
        # 87Ah * 28 Volt = 8.7696e9Ws
        ActorBuilder.set_power_devices(
            actor=sat,
            battery_level_in_Ws=277200 * 0.5,
            max_battery_level_in_Ws=277200,
            charging_rate_in_W=20,
        )
        ActorBuilder.set_thermal_model(
            actor=sat,
            actor_mass=6.0,
            actor_initial_temperature_in_K=283.15,
            actor_sun_absorptance=0.9,
            actor_infrared_absorptance=0.5,
            actor_sun_facing_area=0.012,
            actor_central_body_facing_area=0.01,
            actor_emissive_area=0.1,
            actor_thermal_capacity=6000,
        )
        
        ActorBuilder.add_comm_device(sat, device_name="link", bandwidth_in_kbps=1000)
        
        paseos_cfg = paseos.load_default_cfg()  # loading paseos cfg to modify defaults        
        self.paseos = paseos.init_sim(sat, paseos_cfg)
        self.local_actor = self.paseos.local_actor
        
        self.comm = comm
        self.other_ranks = [x for x in range(self.comm.Get_size()) if x != self.rank] # get all other process numbers
        
        self.ranks_with_same_gpu = [x for x in range(self.comm.Get_size() ) if x % self.n_gpus ==  self.rank % self.n_gpus]
        
        model_size_kb = self._get_model_size_bytes() * 8 / 1e3
        self.comm_duration = model_size_kb / self.local_actor.communication_devices['link'].bandwidth_in_kbps
        print(f'Comm duration: {self.comm_duration} s', flush=True)

    def _get_model_size_bytes(self):
        param_size = 0
        for param in self.model.train_model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.train_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_bytes = (param_size + buffer_size) 
        print('model size: {:.3f}MB'.format(size_all_bytes // 1024**2), flush=True)
        return size_all_bytes

    def _encode_actor(self):
        """Encode an actor in a list.

        Args:
            actor (SpaceActor): Actor to encode

        Returns:
            actor_data: [name,epoch,pos,velocity]
        """
        data = []
        data.append(self.local_actor.name)
        data.append(self.local_actor.local_time)
        r, v = self.local_actor.get_position_velocity(self.local_actor.local_time)
        data.append(r)
        data.append(v)
        return data
    
    def _parse_actor_data(self, actor_data):
        """Decode an actor from a data list

        Args:
            actor_data (list): [name,epoch,pos,velocity]

        Returns:
            actor: Created actor
        """
        actor = ActorBuilder.get_actor_scaffold(
            name=actor_data[0], actor_type=SpacecraftActor, epoch=actor_data[1]
        )
        ActorBuilder.set_orbit(
            actor=actor,
            position=actor_data[2],
            velocity=actor_data[3],
            epoch=actor_data[1],
            central_body=self.earth,
        )
        return actor
    
    def exchange_actors(self, verbose=False):
        """This function exchanges the states of various actors among all MPI ranks.

        Args:
            comm (MPI_COMM_WORLD): The MPI comm world.
            paseos_instance (PASEOS): The local paseos instance.
            local_actor (SpacecraftActor): The rank's local actor.
            other_ranks (list of int): The indices of the other ranks.
            rank (int): Rank's index.
        """
        if verbose:
            print(f"Rank {self.rank} starting actor exchange.")
        send_requests = []  # track our send requests
        recv_requests = []  # track our receive request
        self.paseos.emtpy_known_actors()  # forget about previously known actors

        # Send local actor to other ranks
        for i in self.other_ranks:
            actor_data = self._encode_actor()
            send_requests.append(
                self.comm.isend(actor_data, dest=i, tag=int(str(self.rank) + str(i)))
            )

        # Receive from other ranks
        for i in self.other_ranks:
            recv_requests.append(self.comm.irecv(source=i, tag=int(str(i) + str(self.rank))))

        # Wait for data to arrive
        window_end = pk.epoch(
            self.local_actor.local_time.mjd2000 + self.comm_duration * pk.SEC2DAY
        )
        
        self.ranks_in_lineofsight = []
        t_now = self.local_actor.local_time.mjd2000
        t_after_comm = t_now + + self.comm_duration * pk.SEC2DAY
        for i, recv_request in enumerate(recv_requests):
            other_actor_data = recv_request.wait()
            other_actor = self._parse_actor_data(other_actor_data)
            if self.local_actor.is_in_line_of_sight(other_actor, t_now) and self.local_actor.is_in_line_of_sight(other_actor, t_after_comm):
                self.paseos.add_known_actor(other_actor)
                self.ranks_in_lineofsight.append(self.other_ranks[i])

        # Wait until all other ranks have received everything.
        for send_request in send_requests:
            send_request.wait()

        if verbose:
            print(
                f"Rank {self.rank} completed actor exchange. Knows {self.paseos.known_actor_names} now.", flush=True
            )
    
    def queue_for_gpu(self, verbose=False):
        
        # Wait for ranks ahead to announce they are done
        recv_requests = []
        for i in self.ranks_with_same_gpu:
            if i < self.rank:
                recv_requests.append(self.comm.irecv(source=i, tag=int(str(i) + str(self.rank))))
                
        for recv_request in recv_requests:
            data = recv_request.wait()
        
    def step_out_of_queue_for_gpu(self):
        # announce to all devices sharing the GPU that GPU is released
        send_requests = []
        for i in self.ranks_with_same_gpu:
            if i > self.rank:
                send_requests.append(
                    self.comm.isend(1, dest=i, tag=int(str(self.rank) + str(i)))
                )
        for send_request in send_requests:
            send_request.wait()
    
    def decide_on_activity(
        self,
        timestep,
        time_in_standby,
        standby_period,
        time_since_last_update,
    ):
        """Heuristic to decide activitiy for the actor. Initiates a standby period of passed
        length when going into standby.

        Args:
            paseos_instance (paseos): Local instance

        Returns:
            activity,power_consumption
        """
        if (
            time_since_last_update > 900
            and len(self.paseos.known_actors) > 0
            and self.local_actor.state_of_charge > 0.1
            and self.local_actor.temperature_in_K < 273.15 + 45
        ):
            return "Model_update", 10, 0
        elif (
            self.local_actor.temperature_in_K > (273.15 + 40)
            or self.local_actor.state_of_charge < 0.2
            or (time_in_standby > 0 and time_in_standby < standby_period)
        ):
            return "Standby", 5, time_in_standby + timestep
        else:
            # Wattage from 1605B https://www.amd.com/en/products/embedded-ryzen-v1000-series
            # https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf
            return "Training", 30, 0
    
    def perform_activity(self, activity, power_consumption, time_to_run):
        self.local_actor._current_activity = activity
        return_code = self.paseos.advance_time(
            time_to_run,
            current_power_consumption_in_W=power_consumption,
            constraint_function=self.constraint_func
        )
        if return_code > 0:
            raise ("Activity was interrupted. Constraints no longer true?")
    
    def train_one_batch(self):
        # wait for nodes ahead in queue to use GPU
     #   self.queue_for_gpu()
        # Train the moedl
        self.model.train_one_batch(self.cfg)
        # Announce that gpu is released
     #   self.step_out_of_queue_for_gpu()

    
    def evaluate(self):
        loss, acc = self.model.evaluate()
        return loss, acc

    def constraint_func(self):
        """Constraint function for activitiy

        Args:
            paseos_instance (paseos): Local instance
            actors_to_track (Actor): Actors we want to communicate with

        Returns:
            Whether constraint is still met
        """
        # Check constraints
        if self.paseos.local_actor.temperature_in_K > (273.15 + 65):
            return False
        if self.paseos.local_actor.state_of_charge < 0.1:
            return False

        return True

