import pykep as pk
from .base_node import BaseNode
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor
import torch
import os


class ServerNode(BaseNode):
    """Class for the server orchestrating the federation.

    Args:
        BaseNode (_type_): Class initializing the neural networks.
    """    
    def __init__(self, cfg, node_ranks, sats_pos_and_v=None):
        super(ServerNode, self).__init__(
            rank=None, cfg=cfg, dataloader=None, logger=None, is_server=True
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
                    name=station[0], actor_type=GroundstationActor, epoch=cfg.t0
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

            # Compute orbits of geostationary satellite
            earth = pk.planet.jpl_lp("earth")
            ActorBuilder.set_orbit(
                geosat, sats_pos_and_v[0][0], sats_pos_and_v[0][1], cfg.t0, earth
            )
            self.actors.append(geosat)

        self.node_ranks = node_ranks
        self.local_updates_incomplete = node_ranks
        self.time_since_last_global_update = 0

    def broadcast_global_model(self):
        """Broadcast is done by saving a global model.
        """        
        self.save_model(self.model.train_model, "global_model.pt")  

    def update_global_model(self):
        """Updated the global model with models received. Note that at least 3 models must be received to trigger the update.
        """        
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
                weight = 1 / (n_models)

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
                    local_sd[key] = sum([sd[key] * weight for sd in local_models])

                self.time_since_last_global_update = 0

                # update server model with aggregated models
                self.model.train_model.load_state_dict(local_sd)
                torch.save(
                    self.model.train_model, f"{self.sim_path}/global_model.pt"
                )  # save trained model
