import os
import sys
current_directory = os.getcwd()
if not os.path.exists(current_directory + '\\experimental_data'):
    os.makedirs(current_directory + '\\experimental_data')
    os.makedirs(current_directory + '\\experimental_data\\plots')
    os.makedirs(current_directory + '\\experimental_data\\results')
sys.path.append(current_directory.replace('experiments', ''))
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')
from run import execute_simulation, create_experiment_snapshots, execute_ai_training
from mtdnetwork.mtd.completetopologyshuffle import CompleteTopologyShuffle
from mtdnetwork.mtd.ipshuffle import IPShuffle
from mtdnetwork.mtd.hosttopologyshuffle import HostTopologyShuffle
from mtdnetwork.mtd.portshuffle import PortShuffle
from mtdnetwork.mtd.osdiversity import OSDiversity
from mtdnetwork.mtd.servicediversity import ServiceDiversity
from mtdnetwork.mtd.usershuffle import UserShuffle
from mtdnetwork.mtd.osdiversityassignment import OSDiversityAssignment
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)

# Environment and agent settings
static_features = ["host_compromise_ratio", "exposed_endpoints", "attack_path_exposure",  "overall_asr_avg", "roa", "shortest_path_variability", "risk", "attack_type"]
time_features = ["mtd_freq", "overall_mttc_avg", "time_since_last_mtd"]
features = {"static": static_features, "time": time_features}
state_size = 8
time_series_size = 3 # Time Since Last MTD, MTTC, mtd_freqency
custom_strategies = [
    CompleteTopologyShuffle,
    # HostTopologyShuffle,
    IPShuffle,
    OSDiversity,
    # PortShuffle,
    # OSDiversityAssignment,
    ServiceDiversity,
    # UserShuffle
]
action_size = len(custom_strategies) + 1  # Deploy(4 types, 1-4) or don't deploy(0) MTD technique  

# Learning Parameters
gamma = 0.95  # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
train_start = 1000

episodes = 100

# Simulator Settings
start_time = 0
finish_time = 5000
mtd_interval = 200
scheme = 'mtd_ai'
total_nodes = 100
new_network = True
file_name = "all_features"

static_degrade_factor = 2000

static_features = ["host_compromise_ratio"]
# Loop through time features
custom_strategies = [
    CompleteTopologyShuffle,
    # HostTopologyShuffle,
    IPShuffle,
    OSDiversity,
    # PortShuffle,
    # OSDiversityAssignment,
    ServiceDiversity,
    # UserShuffle
]

for static_feature in static_features:
    features = {"static": [static_feature], "time": []}
    for mtd_strategies in custom_strategies:
        action_size = 2
        file_name = static_feature
        file_name += f"_{[mtd_strategies][0]().get_name()}"
        print(file_name)
        # Train using all features and only deploy single MTD
        execute_ai_training(custom_strategies=[mtd_strategies], features = features, start_time=start_time, finish_time=finish_time, mtd_interval=mtd_interval, state_size=state_size, time_series_size=time_series_size, action_size=action_size, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, batch_size=batch_size, train_start=train_start, scheme=scheme, total_nodes=total_nodes, new_network=new_network, episodes=episodes, file_name=file_name )
    

