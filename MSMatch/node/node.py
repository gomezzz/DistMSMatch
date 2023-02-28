import os
import asyncio
import torch
import numpy as np

from ..utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from ..utils.get_optimizer import get_optimizer
from ..utils.get_net_builder import get_net_builder
from ..utils.TensorBoardLog import TensorBoardLog
from ..models.fixmatch.FixMatch import FixMatch
from ..node.node_utils import model_to_bytestream
from ..node.node_utils import bytestream_to_statedict

import pykep as pk
import paseos
from paseos import ActorBuilder, SpacecraftActor

SHOW_ALL_WINDOWS = True


class BaseNode:
    def __init__(self, rank, cfg, dataloader, logger):
        self.cfg = cfg
        self.rank = rank
        self.logger = logger
        self.save_path = os.path.join(cfg.save_dir, f"node {self.rank}")
        self.tb_log = TensorBoardLog(self.save_path, "")
        self.accuracy = []

        self.device = (
            "cuda:{}".format(self.rank % torch.cuda.device_count())
            if torch.cuda.is_available()
            else "cpu"
        )

        # Create model
        # self.model = self._create_model()
        # self.model.set_data_loader(dataloader)

    def _create_model(self):
        net_builder = get_net_builder(
            self.cfg.net,
            pretrained=self.cfg.pretrained,
            in_channels=self.cfg.num_channels,
            scale=self.cfg.scale,
        )

        model = FixMatch(
            net_builder,
            self.cfg.num_classes,
            self.cfg.num_channels,
            self.cfg.ema_m,
            T=self.cfg.T,
            p_cutoff=self.cfg.p_cutoff,
            lambda_u=self.cfg.ulb_loss_ratio,
            hard_label=True,
            num_eval_iter=self.cfg.num_eval_iter,
            device=self.device,
            rank=self.rank,
        )

        self.logger.info(
            f"Number of Trainable Params: {sum(p.numel() for p in model.train_model.parameters() if p.requires_grad)}"
        )

        # get optimizer, ADAM and SGD are supported.
        optimizer = get_optimizer(
            model.train_model,
            self.cfg.opt,
            self.cfg.lr,
            self.cfg.momentum,
            self.cfg.weight_decay,
        )
        # We use a learning rate schedule to control the learning rate during training.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            self.cfg.num_train_iter,
            num_warmup_steps=self.cfg.num_train_iter * 0,
        )
        model.set_optimizer(optimizer, scheduler)

        self.logger.info(f"model_arch: {model}")
        self.logger.info(f"Arguments: {self.cfg}")

        return model

    def save_model(self):
        self.model.save_run("latest_model.pth", self.cfg.save_path, self.cfg)

    def save_history(self):
        np.save(self.save_path, self.accuracy)

    def aggregate(self, neighbor_sd):
        self.logger.info(f"Node {self.rank}: aggregating neighbor models")
        # TODO: change this to weighted average based on samples
        cw = 1 / (len(neighbor_sd) + 1)

        local_sd = self.model.train_model.state_dict()
        # neighbor_sd = [m.state_dict() for m in rx_models]
        for key in local_sd:
            local_sd[key] = cw * local_sd[key] + sum(
                [sd[key] * cw for i, sd in enumerate(neighbor_sd)]
            )

        # update server model with aggregated models
        self.model.train_model.load_state_dict(local_sd)
        self.model._eval_model_update()
        self.do_training = True


class PaseosNode(BaseNode):
    def __init__(self, rank, pos_and_vel, cfg, dataloader, logger):
        super(PaseosNode, self).__init__(rank, cfg, dataloader, logger)

        # Set up PASEOS instance
        self.earth = pk.planet.jpl_lp("earth")
        id = f"sat{self.rank}"
        sat = ActorBuilder.get_actor_scaffold(id, SpacecraftActor, pk.epoch(0))
        ActorBuilder.set_orbit(
            sat, pos_and_vel[0], pos_and_vel[1], pk.epoch(0), self.earth
        )
        ActorBuilder.set_power_devices(sat, 500, 10000, 1)
        ActorBuilder.add_comm_device(sat, device_name="link", bandwidth_in_kbps=1000)

        paseos_cfg = paseos.load_default_cfg()  # loading paseos cfg to modify defaults
        paseos_cfg.sim.time_multiplier = 10.0

        self.paseos = paseos.init_sim(sat, paseos_cfg)
        self.local_actor = self.paseos.local_actor

        kBps2bps = 8 / 1e3
        # self.model_size = len(model_to_bytestream(self.model.train_model))
        # print(f"Model size = {self.model_size}", flush=True)
        # model_size_kb = self.model_size * kBps2bps
        # self.tx_duration = (
        #     model_size_kb
        #     / self.local_actor.communication_devices["link"].bandwidth_in_kbps
        # )

        # self.do_training = True
        # # Register activities
        # self.paseos.register_activity(
        #     "Train", activity_function=self.train, power_consumption_in_watt=0
        # )

        # self.paseos.register_activity(
        #     "Evaluate", activity_function=self.evaluate, power_consumption_in_watt=0
        # )

    @property
    def name(self):
        return self.paseos.local_actor.name

    def local_time(self) -> float:
        return self.paseos.local_actor.local_time.mjd2000 * pk.DAY2SEC

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

    def exchange_actors(self, comm, other_ranks, verbose=False):
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
        for i in other_ranks:
            actor_data = self._encode_actor()
            send_requests.append(
                comm.isend(actor_data, dest=i, tag=int(str(self.rank) + str(i)))
            )

        # Receive from other ranks
        for i in other_ranks:
            recv_requests.append(comm.irecv(source=i, tag=int(str(i) + str(self.rank))))

        # Wait for data to arrive
        self.ranks_in_lineofsight = []
        local_t = self.local_actor.local_time
        for i, recv_request in enumerate(recv_requests):
            other_actor_data = recv_request.wait()
            other_actor = self._parse_actor_data(other_actor_data)
            if self.local_actor.is_in_line_of_sight(other_actor, local_t):
                self.paseos.add_known_actor(other_actor)
                self.ranks_in_lineofsight.append(other_ranks[i])

        # Wait until all other ranks have received everything.
        for send_request in send_requests:
            send_request.wait()

        if verbose:
            print(
                f"Rank {self.rank} completed actor exchange. Knows {self.paseos.known_actor_names} now.",
                flush=True,
            )

    def exchange_models_and_aggregate(self, comm, verbose=False):
        # Share trained model with actors in line-of-sight
        send_requests = []  # track our send requests
        recv_requests = []  # track our receive request

        if len(self.ranks_in_lineofsight) > 0:
            if verbose:
                self.logger.info(f"Rank {self.rank} starting model exchange.")
            model_data = model_to_bytestream(self.model.train_model)

            for i in self.ranks_in_lineofsight:
                self.logger.info(f"Rank {self.rank} tx/rx from rank {i}.")
                recv_requests.append(
                    comm.irecv(
                        buf=(self.model_size + 50),
                        source=i,
                        tag=int(str(i) + str(self.rank)),
                    )
                )
                send_requests.append(
                    comm.isend(model_data, dest=i, tag=int(str(self.rank) + str(i)))
                )

            actor_models = []
            for recv_request in recv_requests:
                other_actor_model_bytearray = recv_request.wait()
                self.logger.info(f"Rank {self.rank} receive completed.")
                actor_models.append(
                    bytestream_to_statedict(
                        other_actor_model_bytearray, self.model.train_model
                    )
                )

            for send_request in send_requests:
                send_request.wait()

            self.aggregate(actor_models)

            if verbose:
                print(
                    f"Rank {self.rank} aggregated with {self.paseos.known_actor_names}."
                )
        else:
            if verbose:
                self.logger.info(f"Rank {self.rank} no LOS connections.")

    def advance_time(self, timestep):
        self.paseos.advance_time(timestep, current_power_consumption_in_W=0)

    async def wait_for_activity(self):
        self.paseos.wait_for_activity()

    # Activities
    async def train(self, args):
        result = self.model.train(self.cfg)
        self.logger.info(f"Rank {self.rank} eval acc: {result['eval/top-1-acc']}")
        self.accuracy.append(result["eval/top-1-acc"].cpu())

    async def evaluate(self, args):
        return self.model.evaluate(cfg=self.cfg)
