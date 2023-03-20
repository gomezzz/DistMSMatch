"""Define constants for the nodes
"""

SWARM_COMM_POWER = 13.5  # [W] power consumption using ISL
FL_GS_COMM_POWER = 10  # [W] power consumption from satellite to GS
TRAIN_POWER = 30  # [W] power consumption from training
STANDBY_POWER = 5  # [W] power consumption from stand by
COMM_MIN_STATE_OF_CHARGE = 0.1  # minimum state of charge to communicate
STANDBY_STATE_OF_CHARGE = 0.2  # state of charge before going into standby
COMM_MAX_TEMPERATURE = 273.15 + 45  # [K] maximum temperature to communicate
STANDBY_TEMPERATURE = 273.15 + 40  # [K] temperature for going into stand by
CONSTRAINT_MAX_TEMPERATURE = 273.15 + 65  # [K] maximum temperature to run activities
CONSTRAINT_MIN_STATE_OF_CHARGE = 0.1  # [W] minimum state of charge to run activities
