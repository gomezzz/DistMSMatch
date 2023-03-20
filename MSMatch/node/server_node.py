import pykep as pk
from .base_node import BaseNode
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor
import torch
from .aggregation import aggregate_models


class ServerNode(BaseNode):
    """Class for the server orchestrating the federation.
    The server is handled by the process with rank = 0.

    Args:
        BaseNode (_type_): Class initializing the neural networks.
    """

    def __init__(self, cfg, node_ranks, sats_pos_and_v=None):
        """
        Args:
            cfg: parameters to create neural network
            node_ranks (list): rank number of all the nodes
            sats_pos_and_v (list): position and velocity of geosat
        """
        super(ServerNode, self).__init__(
            rank=None, cfg=cfg, dataloader=None, is_server=True
        )

        # There may be more than one parameter server
        self.actors = []
        if cfg.mode == "FL_ground":
            for station in cfg.stations:
                gs_actor = ActorBuilder.get_actor_scaffold(
                    name=station[0], actor_type=GroundstationActor, epoch=cfg.t0
                )
                ActorBuilder.set_ground_station_location(
                    gs_actor,
                    latitude=station[1],
                    longitude=station[2],
                    elevation=station[3],
                    minimum_altitude_angle=cfg.minimum_altitude_angle,
                )
                self.actors.append(gs_actor)
        elif cfg.mode == "FL_geostat":
            geosat = ActorBuilder.get_actor_scaffold(
                "EDRS-C", SpacecraftActor, epoch=pk.epoch(0)
            )

            # Compute orbits of geostationary satellite
            earth = pk.planet.jpl_lp("earth")
            ActorBuilder.set_orbit(
                geosat, sats_pos_and_v[0][0], sats_pos_and_v[0][1], cfg.t0, earth
            )
            self.actors.append(geosat)

        self.node_ranks = node_ranks
        self.models_shared = [False for i in range(len(node_ranks))]

        self.model.train_model.to(self.device)

    def save_global_model(self):
        self.save_model("global_model")

    def update_global_model(self):
        """Updated the global model with models received by averaging weights and then save the new model to folder."""
        n_models = sum(self.models_shared)
        if n_models > 0:
            # make sure that we have at least 3 models to aggregate
            local_sd = self.model.train_model.state_dict()
            weight = 1 / (n_models + 1)

            # get the local models that have been shared
            paths = []
            for rank, model_shared in enumerate(self.models_shared):
                if model_shared:
                    print(f"Updating global model with node{rank}", flush=True)
                    paths.append(f"{self.sim_path}/node{rank}_model.pt")
                    self.models_shared[rank] = False

            aggregate_models(local_sd, weight, paths) # aggregate local models with global model

            # update server model with aggregated models
            self.model.train_model.load_state_dict(local_sd)
            self.save_global_model()
