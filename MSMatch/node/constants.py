"""Define constants for the nodes
"""

SWARM_COMM_PWR = 13.5 # [W] power consumption of communicating using ISL
FL_GS_COMM_PWR = 10 # [W] power consumption to from satellite to GS
TRAIN_PWR = 30 # [W] power consumption from training
STANDBY_PWR = 5 # [W] power consumption from stand by
COMM_MIN_SOC = 0.1 # minimum state of charge to communicate
STANDBY_SOC = 0.2 # state of charge before going into standby
COMM_MAX_TEMP = 273.15 + 45 # [K] maximum temperature to communicate
STANDBY_TEMP = 273.15 + 40 # [K] temperature for going into stand by
