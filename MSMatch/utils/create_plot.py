import numpy as np
import matplotlib.pyplot as plt

MODE = 2 # 0 = FL_ground, 1 = FL_geostat, 2 = Swarm

time_per_batch = 16
if MODE == 0: 
    time_per_communication = 109.25
    W_for_comm = 10
    # path = "results/FL_ground/20230302-232823" # FL_ground old version
    path = "results/FL_ground/20230303-135240"
    n_nodes = 4
    data_multiplier = 1 # only send data once
    comm_multiplier = 2 # down and up
elif MODE == 1:
    pass
    #path = "results/FL_geostat/20230302-221148" #FL_geostat old version
elif MODE == 2:
    time_per_communication = 1.09
    W_for_comm = 13.5
    n_nodes = 8
    #path = 'results/ISL/20230302-184139' #swarm old version
    path = "results/ISL/20230303-174256"
    path = "results/ISL/20230306-102343"
    path= "results/ISL/20230306-211213" # 50 labels
    data_multiplier = 2 # one for each neighbor
    comm_multiplier = 2 # one for each neighbor
    
model_size = 13 #MB
W_for_training = 30
W_for_standby = 5
total_time = 72*3600

test_acc = []
test_time = []
train_time = []
comm_time = []
orbit_duration = 6025
for i in range(n_nodes):
    test_acc.append(np.genfromtxt(f"{path}/node{i}/test_acc.csv", delimiter=','))
    test_time.append(np.genfromtxt(f"{path}/node{i}/test_time.csv", delimiter=',')/orbit_duration)
    train_time.append(np.genfromtxt(f"{path}/node{i}/train_time.csv", delimiter=','))
    comm_time.append(np.genfromtxt(f"{path}/node{i}/comm_time.csv", delimiter=','))
    
    plt.plot(test_time[i], test_acc[i], label=f"Sat{i}")
plt.xlabel("Completed orbits")
plt.ylabel("Test accuracy")
plt.ylim(0.4,1.0)
plt.title("Swarm learning with 50 labels and pretrained EfficientNet-lite0")
plt.legend()
plt.grid()
plt.show()

# get test accuracy
acc = np.mean([x[-1] for x in test_acc])
n_train = np.mean( [len(x) for x in train_time] )
training_time = n_train * time_per_batch
n_comm = np.mean( [len(x) for x in comm_time] )
comm_time = comm_multiplier*n_comm * time_per_communication
comm_data = data_multiplier * n_comm * model_size
standby_time = total_time - training_time - comm_time
consumed_power_kWh = (training_time*W_for_training + comm_time/2*W_for_comm + standby_time*W_for_standby) / (1e3 * 3600)
print(
    f"Statistics of simulation: "
    + f"\nAccuracy: {acc}" 
    + f"\nTransmitted data [MB]: {comm_data}"
    + f"\nConsumed power [kWh]: {consumed_power_kWh}"
    + f"\nTraining time [h]: {training_time/3600}"
    + f"\nComm time [h]: {comm_time/3600}"
)
