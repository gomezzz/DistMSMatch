import pykep as pk
import paseos
from .base_node import BaseNode
from . import constants
from .aggregation import aggregate_models
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor
import torch

class SpaceCraftNode(BaseNode):
    """Class to create a spacecraft node

    Args:
        BaseNode (_type_): based on the BaseNode class to create a neural network
    """

    def __init__(
        self,
        pos_and_vel,
        cfg,
        dataloader,
        comm=None,
    ):
        rank = comm.Get_rank()
        super(SpaceCraftNode, self).__init__(rank, cfg, dataloader)

        # Set up PASEOS instance
        self.earth = pk.planet.jpl_lp("earth")
        id = f"sat{self.rank}"
        sat = ActorBuilder.get_actor_scaffold(id, SpacecraftActor, cfg.t0)
        ActorBuilder.set_orbit(sat, pos_and_vel[0], pos_and_vel[1], cfg.t0, self.earth)
        # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
        # 87Ah * 28 Volt = 8.7696e9Ws
        ActorBuilder.set_power_devices(
            actor=sat,
            battery_level_in_Ws=cfg.battery_level_in_Ws,
            max_battery_level_in_Ws=cfg.max_battery_level_in_Ws,
            charging_rate_in_W=cfg.charging_rate_in_W
        )
        ActorBuilder.set_thermal_model(
            actor=sat,
            actor_mass=cfg.actor_mass,
            actor_initial_temperature_in_K=cfg.actor_initial_temperature_in_K,
            actor_sun_absorptance=cfg.actor_sun_absorptance,
            actor_infrared_absorptance=cfg.actor_infrared_absorptance,
            actor_sun_facing_area=cfg.actor_sun_facing_area,
            actor_central_body_facing_area=cfg.actor_central_body_facing_area,
            actor_emissive_area=cfg.actor_emissive_area,
            actor_thermal_capacity=cfg.actor_thermal_capacity
        )

        ActorBuilder.add_comm_device(
            sat, device_name="link", bandwidth_in_kbps=cfg.bandwidth_in_kpbs
        )

        paseos_cfg = paseos.load_default_cfg()  # loading paseos cfg to modify defaults
        paseos_cfg.sim.start_time = cfg.t0.mjd2000 * pk.DAY2SEC # overwrite the starting time for PASEOS config
        self.paseos = paseos.init_sim(sat, paseos_cfg)
        self.local_actor = self.paseos.local_actor

        if comm is not None:
            self.comm = comm
            self.other_ranks = [
                x for x in range(self.comm.Get_size()) if x != self.rank
            ]  # get all other process numbers

        model_size_kb = cfg.compression_ratio * self._get_model_size_bytes() * 8 / 1e3
        self.comm_duration = (
            model_size_kb
            / self.local_actor.communication_devices["link"].bandwidth_in_kbps
        )
        self.comm_duration += self.comm_duration + cfg.ISL_setup_time # add setup time to the communication duration
        print(f"Comm duration: {self.comm_duration} s", flush=True)

        self.update_time = cfg.update_time

    def set_server_node(self, server_node: GroundstationActor):
        """Makes the server node known to the spacecraft

        Args:
            server_node (GroundstationActor): the servernode actor
        """
        self.server_node = server_node

    def _get_model_size_bytes(self):
        """Compute the size of the neural network in bytes

        Returns:
            _type_: size in bytes
        """
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
        """Encode the spacecraft as a list.

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

    def check_if_sever_available(self):
        """Go through the server actor(s) and check if there is line-of-sight.
        If so, add the server actor to the list of known actors
        """

        window_start = self.local_actor.local_time
        # we need a window to go back and forth
        window_end = pk.epoch(
            self.local_actor.local_time.mjd2000 + 2 * self.comm_duration * pk.SEC2DAY
        )
        self.paseos.empty_known_actors()  # forget about previously known actors
        for server in self.server_node.actors:
            if self.local_actor.is_in_line_of_sight(
                server, epoch=window_start
            ) and self.local_actor.is_in_line_of_sight(server, epoch=window_end):
                self.paseos.add_known_actor(server)
                return

    def get_global_model(self):
        """Replace the current local model with the global model

        Returns:
            bool: whether update was successful or not
        """
        # We might attempt to load at the same time as model is being saved.
        model_loaded = False
        update_attempts = 10
        while model_loaded == False and update_attempts > 0:
            try:
                global_model = torch.load(
                    f"{self.cfg.sim_path}/global_model.pt"
                ).state_dict()
                self.model.train_model.load_state_dict(global_model)
                self.model.eval_model.load_state_dict(global_model)
                model_loaded = True
                return model_loaded
            except:
                print(
                    f"Node{self.rank} failed loading global model, trying again",
                    flush=True,
                )
                update_attempts -= 1

        return model_loaded

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
            activity, power_consumption, time_in_standby
        """
        if (
            time_since_last_update > self.update_time
            and len(self.paseos.known_actors) > 0
            and self.local_actor.state_of_charge > constants.COMM_MIN_STATE_OF_CHARGE
            and self.local_actor.temperature_in_K < constants.COMM_MAX_TEMPERATURE
        ):
            self.local_actor._current_activity = "Model_update"
            if self.cfg.mode == "Swarm":
                return "Model_update", constants.SWARM_COMM_POWER, 0
            else:
                return "Model_update", constants.FL_GS_COMM_POWER, 0
        elif (
            self.local_actor.temperature_in_K > constants.STANDBY_TEMPERATURE
            or self.local_actor.state_of_charge < constants.STANDBY_STATE_OF_CHARGE
            or (time_in_standby > 0 and time_in_standby < standby_period)
        ):
            self.local_actor._current_activity = "Standby"
            return "Standby", constants.STANDBY_POWER, time_in_standby + timestep
        else:
            # Wattage from 1605B https://www.amd.com/en/products/embedded-ryzen-v1000-series
            # https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf
            self.local_actor._current_activity = "Training"
            return "Training", constants.TRAIN_POWER, 0

    def perform_activity(self, power_consumption, time_to_run):
        """Performs an activity by consuming power and advancing the time.
        During the time advancement, a constraint function is

        Args:
            power_consumption (_type_): power consumption in W
            time_to_run (_type_): time to advance simulation

        Returns:
            _type_: return code stating if we were able to complete activity
        """
        return_code = self.paseos.advance_time(
            time_to_run,
            current_power_consumption_in_W=power_consumption,
            constraint_function=self.constraint_func,
        )
        if return_code > 0:
            raise ("Activity was interrupted. Constraints no longer true?")
        return return_code

    def train_one_batch(self):
        """Train the training model over a single batch

        Returns:
            train_acc: test accuracy
        """
        train_acc = self.model.train_one_batch()
        return train_acc

    def evaluate(self):
        """Evaluate the eval model on the test set

        Returns:
            loss: accumulated loss on the test set
            acc: average accuracy on the test set
        """
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
        if (
            self.paseos.local_actor.temperature_in_K
            > constants.CONSTRAINT_MAX_TEMPERATURE
        ):
            return False
        if (
            self.paseos.local_actor.state_of_charge
            < constants.CONSTRAINT_MIN_STATE_OF_CHARGE
        ):
            return False

        return True

    def aggregate_neighbors(self):
        """Aggregate the neighboring models with the local model"""
        self.model.train_model.to("cpu")
        local_sd = self.model.train_model.state_dict()
        num_neighbors = len(self.ranks_in_lineofsight)
        weights = (1 / (num_neighbors + 1)) * torch.ones(num_neighbors+1)
        paths = []
        for i in self.ranks_in_lineofsight:
            paths.append(f"{self.cfg.sim_path}/node{i}_model.pt")

        local_sd = aggregate_models(local_sd, weights, paths) # aggregate models

        # update training and eval model with aggregated model
        self.model.train_model.load_state_dict(local_sd)
        self.model.eval_model.load_state_dict(local_sd)
        self.model.train_model.to(self.device)
        self.model.eval_model.to(self.device)
        

        self.do_training = True
