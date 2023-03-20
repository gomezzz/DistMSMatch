from dotmap import DotMap
import time

def get_default_cfg():
    """Returns the default configuration for MSMatch.

    Returns:
        DotMap: the default configuration
    """
    cfg = DotMap(_dynamic=False)

    cfg.mode = "Swarm"  # "FL_ground", "FL_geostat", "Swarm"
    cfg.save_dir = "./results/"
    cfg.nodes = 8  # number of spacecraft participating in colaborative learning

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if cfg.mode == "Swarm":
        cfg.sim_path = cfg.save_dir + f"ISL/{timestr}"
    elif cfg.mode == "FL_ground":
        cfg.sim_path = cfg.save_dir + f"FL_ground/{timestr}"
    elif cfg.mode == "FL_geostat":
        cfg.sim_path = cfg.save_dir + f"FL_geostat/{timestr}"

    # Configuration related to the dataset
    cfg.dataset = "eurosat_rgb"  # "eurosat_ms" # "eurosat_rgb"
    cfg.alpha = 100  # (0,inf), data heterogeneity (higher means more homogeneous)

    # Configuration related to the neural networks
    cfg.net = "efficientnet-lite0"
    cfg.batch_size = 32
    cfg.p_cutoff = 0.95
    cfg.lr = 0.03
    cfg.uratio = 3
    cfg.weight_decay = 7.5e-4
    cfg.ulb_loss_ratio = 1.0
    cfg.seed = 42
    cfg.num_labels = 50
    cfg.num_train_iter = (100 * cfg.num_labels) // cfg.batch_size
    cfg.opt = "SGD"
    cfg.pretrained = True
    cfg.ema_m = 0.99
    cfg.eval_batch_size = 64
    cfg.momentum = 0.9
    cfg.T = 0.5
    cfg.amp = False
    cfg.hard_label = True
    cfg.scale = 1

    # PASEOS specific configuration
    cfg.start_time = "2023-Dec-17 14:42:42"
    cfg.planes = 1
    cfg.constellation_altitude = 786 * 1000  # altitude above the Earth's ground [m]
    cfg.constellation_inclination = 98.62  # inclination of the orbit

    # onboard power device settings
    cfg.battery_level_in_Ws = 277200 * 0.5
    cfg.max_battery_level_in_Ws = 277200
    cfg.charging_rate_in_W = 20

    # onboard temperature settings
    cfg.actor_mass = 6.0
    cfg.actor_initial_temperature_in_K = 283.15
    cfg.actor_sun_absorptance = 0.9
    cfg.actor_infrared_absorptance = 0.5
    cfg.actor_sun_facing_area = 0.012
    cfg.actor_central_body_facing_area = 0.01
    cfg.actor_emissive_area = 0.1
    cfg.actor_thermal_capacity = 6000

    # onboard communication device
    if cfg.mode == "Swarm" or cfg.mode == "FL_geostat":
        cfg.bandwidth_in_kpbs = 100000  # [kbps] 100 Mbps (optical link)
    else:
        cfg.bandwidth_in_kpbs = 1000  # [kpbs] 1 Mbps (RF link)
    cfg.compression_ratio = 12.7/13.8 # 13.8 and 12.7 MB are the sizes of original and zipped version of the model
    
    # Groundstation coordinates
    if cfg.mode == "FL_ground":
        cfg.stations = [
            ["Maspalomas", 27.7629, -15.6338, 205.1],
            ["Matera", 40.6486, 16.7046, 536.9],
            ["Svalbard", 78.9067, 11.8883, 474.0],
        ]
        cfg.minimum_altitude_angle = 5

    cfg.update_time = (
        1e3  # [s] how much time should pass before attempting to share model
    )

    return cfg
