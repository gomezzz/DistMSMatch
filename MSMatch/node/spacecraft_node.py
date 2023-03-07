import pykep as pk
import paseos
from .base_node import BaseNode
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor
import torch
import os


class ServerNode(BaseNode):
    def __init__(self, cfg, node_ranks, sats_pos_and_v=None):
        super(ServerNode, self).__init__(
            rank=None, cfg=cfg, dataloader=None, logger=None
        )

        # There may be more than one parameter server
        self.actors = []
        if cfg.mode == "FL_ground":
            # Ground stations
            stations = [
                ["Maspalomas", 27.7629, -15.6338, 205.1],
                ["Matera", 40.6486, 16.7046, 536.9],
                ["Svalbard", 78.9067, 11.8883, 474.0],
            ]

            for station in stations:
                gs_actor = ActorBuilder.get_actor_scaffold(
                    name=station[0], actor_type=GroundstationActor, epoch=pk.epoch(0)
                )
                ActorBuilder.set_ground_station_location(
                    gs_actor,
                    latitude=station[1],
                    longitude=station[2],
                    elevation=station[3],
                    minimum_altitude_angle=5,
                )
                # paseos_instance.add_known_actor(gs_actor)
                self.actors.append(gs_actor)
        elif cfg.mode == "FL_geostat":
            geosat = ActorBuilder.get_actor_scaffold(
                "EDRS-C", SpacecraftActor, epoch=pk.epoch(0)
            )
            t0 = pk.epoch_from_string(
                "2023-Dec-17 14:42:42"
            )  # starting date of our simulation

            # Compute orbits of geostationary satellite
            earth = pk.planet.jpl_lp("earth")
            ActorBuilder.set_orbit(
                geosat, sats_pos_and_v[0][0], sats_pos_and_v[0][1], t0, earth
            )
            self.actors.append(geosat)

        self.node_ranks = node_ranks
        self.local_updates_incomplete = node_ranks
        self.time_since_last_global_update = 0

    def broadcast_global_model(self):
        torch.save(
            self.model.train_model, f"{self.sim_path}/global_model.pt"
        )  # save global model

    def update_time_since_update(self, dt):
        self.time_since_last_global_update += dt
        
    def update_global_model(self):
        if self.time_since_last_global_update > 1e3:
            f = []
            for (dirpath, dirnames, filenames) in os.walk(self.sim_path):
                f.extend(filenames)
                break

            local_model_paths = []
            for filename in f:
                if "node" in filename:
                    local_model_paths.append(filename)

            n_models = len(local_model_paths)
            # make sure that we have at least 3 models to aggregate
            if n_models > 2:
                local_sd = self.model.train_model.state_dict()
                cw = 1 / (n_models)

                local_models = []
                for filename in local_model_paths:
                    print(f"Updating global model with {filename}", flush=True)
                    path = f"{self.sim_path}/{filename}"
                    try:
                        local_models.append(torch.load(path).state_dict())
                        os.remove(path)
                    except:
                        print("Load not successful", flush=True)

                for key in local_sd:
                    local_sd[key] = sum([sd[key] * cw for sd in local_models])

                self.time_since_last_global_update = 0

                # update server model with aggregated models
                self.model.train_model.load_state_dict(local_sd)
                torch.save(
                    self.model.train_model, f"{self.sim_path}/global_model.pt"
                )  # save trained model


class SpaceCraftNode(BaseNode):
    def __init__(
        self,
        pos_and_vel,
        cfg,
        dataloader,
        comm=None,
        logger=None,
    ):
        if comm is not None:
            rank = comm.Get_rank()
        else:
            rank = 0
        super(SpaceCraftNode, self).__init__(rank, cfg, dataloader, logger)

        # Set up PASEOS instance
        self.earth = pk.planet.jpl_lp("earth")
        id = f"sat{self.rank}"
        sat = ActorBuilder.get_actor_scaffold(id, SpacecraftActor, pk.epoch(0))
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
        if cfg.mode == "Swarm" or cfg.mode == "FL_geostat":
            bw = 100000 # 100 Mbps (optical link)
        else:
            bw = 1000 # 1 Mbps (RF link)
        ActorBuilder.add_comm_device(sat, device_name="link", bandwidth_in_kbps=bw)

        paseos_cfg = paseos.load_default_cfg()  # loading paseos cfg to modify defaults
        self.paseos = paseos.init_sim(sat, paseos_cfg)
        self.local_actor = self.paseos.local_actor

        if comm is not None:
            self.comm = comm
            self.other_ranks = [
                x for x in range(self.comm.Get_size()) if x != self.rank
            ]  # get all other process numbers

        model_size_kb = self._get_model_size_bytes() * 8 / 1e3
        self.comm_duration = (
            model_size_kb
            / self.local_actor.communication_devices["link"].bandwidth_in_kbps
        )
        print(f"Comm duration: {self.comm_duration} s", flush=True)

        self.update_time = 1e3

    def set_server_node(self, server_node):
        self.server_node = server_node

    def _get_model_size_bytes(self):
        param_size = 0
        for param in self.model.train_model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.train_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_bytes = param_size + buffer_size
        print("model size: {:.3f}MB".format(size_all_bytes // 1024**2), flush=True)
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
        data.append(self.local_actor.current_activity)
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
        actor._current_activity = actor_data[4]
        return actor

    def check_if_sever_available(self):

        window_start = self.local_actor.local_time
        # we need a window to go back and forth
        window_end = pk.epoch(
            self.local_actor.local_time.mjd2000 + 2 * self.comm_duration * pk.SEC2DAY
        )
        self.paseos.emtpy_known_actors()  # forget about previously known actors
        for s in self.server_node.actors:
            if self.local_actor.is_in_line_of_sight(
                s, epoch=window_start
            ) and self.local_actor.is_in_line_of_sight(s, epoch=window_end):
                self.paseos.add_known_actor(s)
                return

    def get_global_model(self):
        # We might attempt to load at the same time as model is being saved.
        try:
            global_model = torch.load(f"{self.sim_path}/global_model.pt").state_dict()
            self.model.train_model.load_state_dict(global_model)
            self.model.train_model.to("cpu")
            self.model.eval_model.to("cpu")
            self.model._eval_model_update()
            return True
        except:
            print(f"Node{self.rank} could not load global model")

        return False

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
            print(f"Rank {self.rank} starting actor exchange.", flush=True)
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
            recv_requests.append(
                self.comm.irecv(source=i, tag=int(str(i) + str(self.rank)))
            )

        # Wait for data to arrive
        window_end = pk.epoch(
            self.local_actor.local_time.mjd2000 + self.comm_duration * pk.SEC2DAY
        )

        self.ranks_in_lineofsight = []
        self.other_actors_training = []
        window_start = self.local_actor.local_time
        window_end = pk.epoch(
            self.local_actor.local_time.mjd2000 + self.comm_duration * pk.SEC2DAY
        )
        for i, recv_request in enumerate(recv_requests):
            other_actor_data = recv_request.wait()
            other_actor = self._parse_actor_data(other_actor_data)
            is_training = other_actor.current_activity == "Training"
            self.other_actors_training.append(is_training)

            if self.local_actor.is_in_line_of_sight(
                other_actor, epoch=window_start
            ) and self.local_actor.is_in_line_of_sight(other_actor, epoch=window_end):
                self.paseos.add_known_actor(other_actor)
                self.ranks_in_lineofsight.append(self.other_ranks[i])

        # Wait until all other ranks have received everything.
        for send_request in send_requests:
            send_request.wait()

        if verbose:
            print(
                f"Rank {self.rank} completed actor exchange. Knows {self.paseos.known_actor_names} now.",
                flush=True,
            )

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
            time_since_last_update > self.update_time
            and len(self.paseos.known_actors) > 0
            and self.local_actor.state_of_charge > 0.1
            and self.local_actor.temperature_in_K < 273.15 + 45
        ):
            self.local_actor._current_activity = "Model_update"
            if self.cfg.mode == "Swarm":
                return "Model_update", 13.5, 0
            else:
                return "Model_update", 10, 0
        elif (
            self.local_actor.temperature_in_K > (273.15 + 40)
            or self.local_actor.state_of_charge < 0.2
            or (time_in_standby > 0 and time_in_standby < standby_period)
        ):
            self.local_actor._current_activity = "Standby"
            return "Standby", 5, time_in_standby + timestep
        else:
            # Wattage from 1605B https://www.amd.com/en/products/embedded-ryzen-v1000-series
            # https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf
            self.local_actor._current_activity = "Training"
            return "Training", 30, 0

    def perform_activity(self, activity, power_consumption, time_to_run):
        return_code = self.paseos.advance_time(
            time_to_run,
            current_power_consumption_in_W=power_consumption,
            constraint_function=self.constraint_func,
        )
        if return_code > 0:
            raise ("Activity was interrupted. Constraints no longer true?")

    def train_one_batch(self):
        train_acc = self.model.train_one_batch(self.cfg)
        return train_acc.cpu().numpy()

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

    def aggregate(self):
        # self.logger.info(f"Node {self.rank}: aggregating neighbor models")
        self.model.train_model.to(self.device)
        local_sd = self.model.train_model.state_dict()
        # neighbor_sd = [m.state_dict() for m in rx_models]
        cw = 1 / (len(self.ranks_in_lineofsight) + 1)

        for key in local_sd:
            local_sd[key] = cw * local_sd[key].to(self.device)

        for i in self.ranks_in_lineofsight:
            new_sd = torch.load(f"{self.sim_path}/node{i}/model.pt").state_dict()
            for key in local_sd:
                local_sd[key] += new_sd[key].to(self.device) * cw

        # update server model with aggregated models
        self.model.train_model.load_state_dict(local_sd)
        self.model.eval_model.to(self.device)
        self.model._eval_model_update()

        self.model.eval_model.cpu()
        self.model.train_model.cpu()

        self.do_training = True
