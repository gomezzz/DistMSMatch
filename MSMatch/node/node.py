import os
import sys
import asyncio
import torch
import numpy as np

from ..utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from ..utils.get_optimizer import get_optimizer
from ..utils.get_net_builder import get_net_builder
from ..utils.TensorBoardLog import TensorBoardLog
from ..models.fixmatch.FixMatch import FixMatch
from .node_utils import bytestream_to_statedict
from .node_utils import model_to_bytestream

import pykep as pk
import paseos
from paseos import ActorBuilder, SpacecraftActor


import scatterbrained as sb

class Node:
    
    _local_time = None
    
    def __init__(
        self,
        node_indx, 
        cfg,
        dataloader,
        logger,
        host_ip = "127.0.0.1", 
        tx_port = 4001,
        rx_port = 4002,
        heartbeat_intrvl = 2
    ):
    
        super(Node, self).__init__()
        
        self.cfg = cfg
        self.node_indx = node_indx
        self.logger = logger
        save_path = os.path.join(cfg.save_dir, f"node {self.node_indx}")
        self.tb_log = TensorBoardLog(save_path, "")
        
        # Create model
        self.model = self._create_model()
        self.model.set_data_loader(dataloader)
        
        # Set up PASEOS on node
        # Define central body
        earth = pk.planet.jpl_lp("earth")

        # Define local actor
        self.id = f"sat{self.node_indx}"
        self.sat = ActorBuilder.get_actor_scaffold(
            self.id, SpacecraftActor, pk.epoch(0)
        )
        ActorBuilder.set_orbit(
            self.sat, [10000000, 0, 0], [0, 8000.0, 0], pk.epoch(0), earth
        )
        ActorBuilder.set_power_devices(self.sat, 500, 10000, 1)


        self.paseos = paseos.init_sim(self.sat)
        self.los_connections = []

        # Instantiate metadata
        self.metadata = self.sat.get_position_velocity(pk.epoch(0))

        # create network layer
        self.host_ip = host_ip
        self.tx_port = tx_port
        self.rx_port = rx_port
        self.heartbeat_intrvl = heartbeat_intrvl
        self.tx_tasks = []
        self.rx_tasks = []
        self.discovered_peers = None
        self.network_mngr = self._initiate_network_engine()

        # Register activities
        self.paseos.register_activity(
            "Transmit_Receive",
            activity_function=self.transmit_receive,
            power_consumption_in_watt=0,
        )

        self.paseos.register_activity(
            "Train", activity_function=self.train, power_consumption_in_watt=0
        )

        self.paseos.register_activity(
            "Evaluate", activity_function=self.evaluate, power_consumption_in_watt=0
        )

        

    def local_time(self) -> float:
        return self.paseos._local_actor.local_time.mjd2000 * pk.DAY2SEC
    
    def _create_model(self):
        net_builder = get_net_builder(
            self.cfg.net,
            pretrained=self.cfg.pretrained,
            in_channels=self.cfg.num_channels,
            scale=self.cfg.scale
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
                num_eval_iter=self.cfg.num_eval_iter
            )
        self.logger.info(f"Number of Trainable Params: {sum(p.numel() for p in model.train_model.parameters() if p.requires_grad)}")

        # get optimizer, ADAM and SGD are supported.
        optimizer = get_optimizer(
            model.train_model, self.cfg.opt, self.cfg.lr, self.cfg.momentum, self.cfg.weight_decay
        )
        # We use a learning rate schedule to control the learning rate during training.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.cfg.num_train_iter, num_warmup_steps=self.cfg.num_train_iter * 0
        )
        model.set_optimizer(optimizer, scheduler)
        
        # If a CUDA capable GPU is used, we move everything to the GPU now
        if torch.cuda.is_available():
            torch.cuda.set_device(self.cfg.gpu)
            model.train_model = model.train_model.cuda(self.cfg.gpu)
            model.eval_model = model.eval_model.cuda(self.cfg.gpu)

            self.logger.info(f"model_arch: {model}")
            self.logger.info(f"Arguments: {self.cfg}")
        return model
    
    def _initiate_network_engine(self):
        # Create the discovery engine for the node
        de = sb.discovery.DiscoveryEngine(
            publisher=sb.discovery.UDPBroadcaster(self.host_ip, port=self.tx_port),
            subscriber=sb.discovery.UDPReceiver(self.host_ip, port=self.rx_port),
            heartbeat=self.heartbeat_intrvl,
        )

        # create the node
        return sb.Node(
            id=self.id,
            host=self.host_ip,
            discovery_engine=de,
            metadata=self.metadata
        )
    
    def save_model(self):
        self.model.save_run("latest_model.pth", self.cfg.save_path, self.cfg)

    def aggregate(self, rx_models):
        self.logger.info(f"Aggregating neighbor models")
        # TODO: change this to weighted average based on samples
        cw = 1/(len(rx_models)+1)
        
        local_sd = self.model.train_model.state_dict()
        neighbor_sd = [m.state_dict() for m in rx_models]
        for key in local_sd:
            local_sd[key] = cw * local_sd[key] + sum([sd[key] * cw for i, sd in enumerate(neighbor_sd)])
        
        # update server model with aggregated models
        self.model.train_model.load_state_dict(local_sd)
        self.model._eval_model_update()
        
    def advance_time(self):
        self.sim.advance_time(1000)
        self.time = self.sim._state.time

    def update_metadata(self, metadata):
        # if node only belongs to one namespace, we can easily get its identity
        local_ids = self.com_layer.discovery_engine._identities.items()
        identity = next(iter(local_ids))[1]  # get the first identity from list
        self.com_layer.discovery_engine.update_metadata(identity, metadata)
        
    def update_connections(self, ns_name: str):
        """Check the list of discovered nodes in network engine.
        If nodes are not known, add them.
        If nodes are not discovered but known, remove them.
        Save the peers where line-of-sight is established

        Args:
            ns_name (str): name of namespace
        """
        self.discovered_peers = self.network_mngr.get_connections(ns_name)

        # remove actors that have not sent heartbeat for some time
        discovered_peer_names = set([x.id for x in list(self.discovered_peers)])
        known_actors = set(self.paseos.known_actor_names)
        actors_to_remove = known_actors - discovered_peer_names
        for actor in actors_to_remove:
            self.paseos.remove_known_actor(actor)

        # find out what actors to add
        actors_to_add = discovered_peer_names - known_actors

        if len(self.discovered_peers) > 0:
            earth = pk.planet.jpl_lp("earth")
            time = self.sat.local_time
            for peer in self.discovered_peers:
                name = peer.id
                # add if not in list of known actors
                if name in actors_to_add:
                    # get the information from peer
                    location = peer.metadata[0]
                    velocity = peer.metadata[1]

                    # create an actor
                    peer_sat = ActorBuilder.get_actor_scaffold(
                        name, SpacecraftActor, location, time
                    )
                    ActorBuilder.set_orbit(peer_sat, location, velocity, time, earth)
                    self.paseos.add_known_actor(peer_sat)
                else:
                    peer_sat = self.paseos.known_actors[name]

                # check if in line of sight from local actor
                self.los_connections = []
                los = self.sat.is_in_line_of_sight(peer_sat, time)
                if los is True:
                    self.los_connections.append(peer)



    #-----------------------------------------------
    #               Define activities
    #-----------------------------------------------
    async def train(self, args):
        self.logger.info(f"Node {self.node_indx}")
        result = self.evaluate()
        self.logger.info(f"post aggregated acc: {result['eval/top-1-acc']}")
        result = self.model.train(self.cfg)
        self.logger.info(f"post training acc: {result['eval/top-1-acc']}")
        
    async def evaluate(self, args):
        return self.model.evaluate(cfg=self.cfg)
    
    async def wait_for_activity(self):
        while self.paseos._is_running_activity is True:
            await asyncio.sleep(0.1)

    async def transmit_receive(self, args):
        ns = args[0]

        # Create tasks for TX and RX
        if self.los_connections is not None:
            n_peers = len(self.los_connections)

            for peer in self.los_connections:
                msg = model_to_bytestream(self.model)
                self.tx_tasks.append(asyncio.create_task(ns.send_to(peer, msg)))
                self.rx_tasks.append(asyncio.create_task(ns.recv(timeout=1)))
            await asyncio.gather(*self.tx_tasks)

            result = await asyncio.gather(*self.tx_tasks, *self.rx_tasks)
            self.tx_tasks = []
            self.rx_tasks = []
            rx_data = result[n_peers:]

            self.aggregate(rx_data)
        else:
            self.logger.debug(f"No peers connected")

    # def aggregate(self, rx_data):
    #     # convert byte stream to dictionary
    #     peer_models = []
    #     peer_identities = []
    #     for indx, tuple in enumerate(rx_data):
    #         peer_identity = tuple[0]
    #         if peer_identity is not None:
    #             peer_identities.append(peer_identity)
    #             payload = tuple[1][0]
    #             peer_models.append(
    #                 bytestream_to_statedict(payload, self.model)
    #             )

    #     if len(peer_models) > 0:
    #         self.logger.debug(
    #             f"Aggregated {len(peer_models)} peer models from {peer_identities}"
    #         )
    #         # update client model by the mean of the received and local models
    #         self.model.load_state_dict(
    #             self.aggregate(peer_models)
    #         )
    #         self.model_updated = True
    #     else:
    #         self.logger.debug("No models received")