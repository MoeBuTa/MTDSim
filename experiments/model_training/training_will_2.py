import os
import sys
import warnings
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool

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


# # Define your environment and agent settings
static_features = ["host_compromise_ratio",  "attack_path_exposure", "overall_asr_avg", "roa", "risk"]




time_features = ["mtd_freq", "overall_mttc_avg", "time_since_last_mtd", "shortest_path_variability", "ip_variability", "attack_type"]



# Define your parameters
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




# Define your parameters
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

# Custom strategies
custom_strategies = [
    CompleteTopologyShuffle,
    IPShuffle,
    OSDiversity,
    ServiceDiversity
]

# Parallel training function
def parallel_training(custom_strategy):
    # Define the specific features for training
    features = {"static": static_features, "time": time_features}

    
    # Action size is based on the number of strategies (1 strategy + no MTD option)
    action_size = 2
    
    # Create a unique filename for each strategy
    file_name = f"all_features_{custom_strategy.__name__}"
    
    # Execute the training function for the given strategy
    execute_ai_training(
        custom_strategies=[custom_strategy],  # Single strategy at a time
        features=features,
        start_time=start_time,
        finish_time=finish_time,
        mtd_interval=mtd_interval,
        state_size=5,
        time_series_size=6,
        action_size=action_size,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        train_start=train_start,
        scheme=scheme,
        total_nodes=total_nodes,
        new_network=new_network,
        episodes=episodes,
        file_name=file_name,
    )
    print(f"Finished training with {custom_strategy.__name__}")

# Pool of workers
if __name__ == '__main__':
    # Create a pool to parallelize the tasks
    pool = Pool()

    # Start parallel execution for each MTD strategy
    pool.map(parallel_training, custom_strategies)

    # Close and join the pool
    pool.close()
    pool.join()

    print("Parallel training complete!")