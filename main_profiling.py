import sys

sys.path.append("..")

# Main imports
import MSMatch as mm
import numpy as np
import pykep as pk
from loguru import logger
import time


def main():
    cfg = mm.load_cfg(cfg_path=None)
    cfg.num_train_iter = (cfg.lb_epochs * cfg.num_labels) // cfg.batch_size
    # data will be split among cfg.nodes
    node_dls, cfg = mm.create_node_dataloaders(cfg)
    cfg.save_path = ""
    cfg.sim_path = ""

    # Make a PASEOS instance
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62  # inclination of the orbit
    nPlanes = cfg.planes  # the number of orbital planes (see linked wiki article)
    nSats = cfg.nodes // nPlanes  # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2023-Dec-17 14:42:42")  # starting date of our simulation
    # Compute orbits of LEO satellites
    planet_list, sats_pos_and_v = mm.get_constellation(
        altitude, inclination, nSats, nPlanes, t0
    )

    node = mm.SpaceCraftNode(
        sats_pos_and_v[0], cfg, node_dls[0], comm=None, logger=None
    )

    N = 10  # number of batches to train
    training_time = []
    for i in range(N):
        print(f"batch: {i}", flush=True, end="\r")
        start = time.time()
        node.train_one_batch()
        end = time.time()
        training_time.append(end - start)

    avg_time = np.mean(training_time)
    print(f"Training duration per batch is {avg_time}s")


if __name__ == "__main__":
    main()
