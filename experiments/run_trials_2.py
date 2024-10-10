import os
import sys
import warnings
import logging
import multiprocessing
from run_experiment import Experiment
from mtdnetwork.mtd.completetopologyshuffle import CompleteTopologyShuffle
from mtdnetwork.mtd.ipshuffle import IPShuffle
from mtdnetwork.mtd.osdiversity import OSDiversity
from mtdnetwork.mtd.servicediversity import ServiceDiversity

# Setting up logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Simulator settings
epsilon = 1.0
start_time = 0
finish_time = 15000
# mtd_interval = [200]
mtd_interval = [60]
network_size = [150]
total_nodes = 150
new_network = True
trial = 1500
result_head_path = '/Users/williamho/Documents/GitHub/MTDSim'

# Define strategies
mtd_strategies_dict = {
    'CompleteTopologyShuffle': CompleteTopologyShuffle,
    'IPShuffle': IPShuffle,
    'OSDiversity': OSDiversity,
    'ServiceDiversity': ServiceDiversity
}

# Metrics and models

static_features = ["host_compromise_ratio", "attack_path_exposure",  "overall_asr_avg", "roa",  "risk"]

other_features = ["all_features","hybrid","mtd_freq", "overall_mttc_avg", "time_since_last_mtd"]
time_features = ["mtd_freq", "overall_mttc_avg", "time_since_last_mtd"]


metrics = other_features

# Define the function to run each experiment
def run_experiment_in_process(model, metric, process_name):
    strategy_name = model.split('_')[-1]
    mtd_strategy = mtd_strategies_dict.get(strategy_name, None)

    # if not mtd_strategy:
    #     print(f"Strategy {strategy_name} not found in mtd_strategies_dict.")
    #     return

    # print(f"Running model: {model} with strategy: {strategy_name} in {process_name}")
    
    # Create and run the experiment
    experiment = Experiment(
        model_metric=metric, 
        epsilon=epsilon, 
        start_time=start_time, 
        finish_time=finish_time, 
        mtd_interval=mtd_interval, 
        network_size=network_size, 
        total_nodes=total_nodes, 
        new_network=new_network, 
        model=model, 
        trial=trial, 
        result_head_path=result_head_path, 
        mtd_strategies=[mtd_strategy] if mtd_strategy else list(mtd_strategies_dict.values()),
    
        # mtd_strategies=list(mtd_strategies_dict.values())
    )
    
    experiment.run_trials_ai_multi(process_name)

if __name__ == '__main__':
    for metric in metrics:
        models = [
            metric,
            #  f"{metric}_CompleteTopologyShuffle",
            # f"{metric}_IPShuffle",
            # f"{metric}_OSDiversity",
            # f"{metric}_ServiceDiversity"
        ]
        
        # Assign process names dynamically
        process_names = ["process_1", "process_2", "process_3", "process_4"]
        
        # Create multiprocessing pool
        with multiprocessing.Pool(processes=5) as pool:
            # Prepare arguments for each model
            args = [(model, metric, process_names[i]) for i, model in enumerate(models)]
            # Run models in parallel
            pool.starmap(run_experiment_in_process, args)
