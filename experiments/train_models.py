import os
import sys

sys.path.append('/home/22489437/Documents/GitHub/MTDSim')

import concurrent.futures
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
import tensorflow as tf

# Ensure experimental_data directory exists
current_directory = os.getcwd()
if not os.path.exists(current_directory + '\\experimental_data'):
    os.makedirs(current_directory + '\\experimental_data\\plots')
    os.makedirs(current_directory + '\\experimental_data\\results')

sys.path.append(current_directory.replace('experiments', ''))
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')

logging.basicConfig(format='%(message)s', level=logging.INFO)

# Static simulator settings
state_size = 8
time_series_size = 3
action_size = 5
start_time = 0
finish_time = 5000
mtd_interval = 200
scheme = 'mtd_ai'
total_nodes = 100
new_network = True

static_features = ["host_compromise_ratio", "exposed_endpoints", "attack_path_exposure",  "overall_asr_avg", "roa", "shortest_path_variability", "risk", "attack_type"]
time_features = ["mtd_freq", "overall_mttc_avg", "time_since_last_mtd"]
features = {"static": static_features, "time": time_features}


def run_experiment(filename, gamma=None, epsilon=None, epsilon_decay=None, batch_size=None, train_start=None):
    create_experiment_snapshots([25, 50, 75, 100])
    execute_ai_training(features=features, start_time=start_time, finish_time=finish_time, mtd_interval=mtd_interval, 
                        state_size=state_size, time_series_size=time_series_size, action_size=action_size, 
                        gamma=gamma, epsilon=epsilon, epsilon_min=0.01, epsilon_decay=epsilon_decay, 
                        batch_size=batch_size, train_start=train_start, scheme=scheme, 
                        total_nodes=total_nodes, new_network=new_network, episodes=100, file_name=filename)
    
    tf.keras.backend.clear_session()


# Define experiments as (filename, gamma, epsilon, epsilon_decay, batch_size, train_start)

experiments = []

# Experiment 1: Impact of gamma
# gammas = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
# for gamma in gammas:
#     experiments.append((f'gamma_{gamma}', gamma, 1.0, 0.990, 64, 500))

# Experiment 2: Impact of epsilon and epsilon decay
# epsilons = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
epsilons = [0.8, 0.9, 1.0]
epsilon_decays = [0.980, 0.990, 0.995, 0.998]
for epsilon in epsilons:
    for epsilon_decay in epsilon_decays:
        experiments.append((f'epsilon_{epsilon}_decay_{epsilon_decay}', 0.95, epsilon, epsilon_decay, 32, 1000))


# Experiment 4: Impact of train start
train_starts = [500, 1000, 1500, 2000]
for train_start in train_starts:
    experiments.append((f'train_start_{train_start}', 0.95, 1.0, 0.990, 32, train_start))


print(len(experiments))
# Parallel execution of experiments
if __name__ == '__main__':

    # Disable all GPU devices and force TensorFlow to use CPU
    # tf.config.set_visible_devices([], 'GPU')

    # Limit to 4 processes at a time
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_experiment, *exp) for exp in experiments]
        concurrent.futures.wait(futures)
