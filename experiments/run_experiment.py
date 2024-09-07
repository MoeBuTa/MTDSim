import os
import sys
current_directory = os.getcwd()
if not os.path.exists(current_directory + '\\experimental_data'):
    os.makedirs(current_directory + '\\experimental_data')
    os.makedirs(current_directory + '\\experimental_data\\plots')
    os.makedirs(current_directory + '\\experimental_data\\results')
sys.path.append(current_directory.replace('experiments', ''))
import warnings
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
from run import execute_simulation, create_experiment_snapshots, execute_ai_model, single_mtd_simulation, mtd_ai_simulation, multiple_mtd_simulation, specific_multiple_mtd_simulation
from mtdnetwork.mtd.completetopologyshuffle import CompleteTopologyShuffle
from mtdnetwork.mtd.ipshuffle import IPShuffle
from mtdnetwork.mtd.hosttopologyshuffle import HostTopologyShuffle
from mtdnetwork.mtd.portshuffle import PortShuffle
from mtdnetwork.mtd.osdiversity import OSDiversity
from mtdnetwork.mtd.servicediversity import ServiceDiversity
from mtdnetwork.mtd.usershuffle import UserShuffle
from mtdnetwork.mtd.osdiversityassignment import OSDiversityAssignment
import logging
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt

# logging.basicConfig(format='%(message)s', level=logging.INFO)



class Experiment:
    def __init__(self, epsilon, start_time, finish_time, mtd_interval, network_size,total_nodes, new_network,  model, trial, result_head_path):
        # Learning Parameters
        self.epsilon = epsilon  # exploration rate

        # Simulator Settings
        self.start_time = start_time
        self.finish_time = finish_time

        self.total_nodes = total_nodes
        self.new_network = new_network
        self.model = model
        self.schemes = [ model, 'simultaneous', 'random', 'alternative']
        self.trial = trial
        self.model_path = f"AI_model/models_will/main_network_{model}.h5"
        self.mtd_strategies = [
            CompleteTopologyShuffle,
            # HostTopologyShuffle,
            IPShuffle,
            OSDiversity,
            # PortShuffle,
            # OSDiversityAssignment,
            ServiceDiversity,
            # UserShuffle
        ]
        self.mtd_interval = mtd_interval
        self.network_size = network_size
        self.result_head_path = result_head_path




    def run_trials(self, scheme):
        for i in range(self.trial):
            print("Trial_", i)
            print(scheme)
            if scheme == 'nomtd':
                mtd = single_mtd_simulation("nomtd", [None], 
                                                         mtd_interval=self.mtd_interval,network_size=self.network_size) 
            elif scheme == self.model:
      
                mtd = mtd_ai_simulation(self.model, self.model_path, self.start_time, self.finish_time, self.total_nodes, new_network = self.new_network, 
                                                            mtd_interval=self.mtd_interval,network_size=self.network_size )  

            else:
                mtd = specific_multiple_mtd_simulation(scheme, self.mtd_strategies, scheme, mtd_interval=self.mtd_interval,network_size=self.network_size)
        return 
    
    def get_result(self, path,model):
        path = f'{path}/experiments/experimental_data/results/{model}.csv'
        df = pd.read_csv(path)
        return df
    def get_result_checkpoint_median(self, model,checkpoints = 9):

        df = self.get_result(self.result_head_path, model).drop('Name', axis = 1)
        df['group'] = df.index // checkpoints
        # Group by the new column and calculate median
        df = df.groupby('group').median().reset_index(drop=True)
        # Drop the 'group' column if you don't need it in the result
        df = df.drop(columns='group', errors='ignore')
        return df

    def get_result_stats(self, checkpoint_medians,stats_type = 'median'):
        if stats_type == 'median':
            return checkpoint_medians.median()
        return checkpoint_medians.std()
    
    def raw_result_stats_pipeline(self, scheme,run_trial = False, stats_type = 'median', checkpoints = 9):
        if run_trial:
            self.run_trials(scheme)

        checkpoint_medians = self.get_result_checkpoint_median(scheme, checkpoints)
     
        result = self.get_result_stats(checkpoint_medians,stats_type = stats_type)
        return result
        
    def scale_metrics(self, metrics_dict, normalization_dict):
        # Define which metrics should be maximized and which should be minimized
        metrics_to_maximize = {"MEF", 'time_to_compromise'}  
        metrics_to_minimize = {'host_compromise_ratio', 'attack_path_exposure', 'ASR', 'ROA', 'exposed_endpoints', "risk"}  

        scaled_metrics = {}

        for key, value in metrics_dict.items():
            if key in normalization_dict:
                norm_value = normalization_dict[key]
                if norm_value != 0:
                    if key in metrics_to_maximize:
                        # Normalize by dividing the metric value by the normalization value
                        scaled_metrics[key] = value / norm_value
                        
                    elif key in metrics_to_minimize:
                        # Inverse the ratio for metrics to be minimized
                        scaled_metrics[key] = 1 / (value / norm_value)
                    else:
                        # Handle cases where the metric is not in either category
                        scaled_metrics[key] = value / norm_value
                else:
                    # Handle the case where norm_value is zero
                    scaled_metrics[key] = 1  # Or any other placeholder value as needed
            else:
                # Handle cases where normalization value is not defined
                scaled_metrics[key] = value  # Or handle differently as needed
        return scaled_metrics

    def scaled_pipeline(self, scheme,run_trial = False, stats_type = 'median'):
        nomtd_result = self.raw_result_stats_pipeline('nomtd',run_trial, stats_type)
        scheme_result = self.raw_result_stats_pipeline(scheme,run_trial, stats_type)
        scaled_scheme_result = self.scale_metrics(scheme_result.to_dict(), nomtd_result.to_dict())
        return {scheme:scaled_scheme_result}
    
    def multiple_scaled_pipeline(self,schemes,run_trial = False, stats_type = 'median'):
        multi_schemes = {}
        nomtd_result = self.raw_result_stats_pipeline('nomtd',run_trial, stats_type)
        for scheme in schemes:
            scheme_result = self.raw_result_stats_pipeline(scheme,run_trial, stats_type)
            scaled_scheme_result = self.scale_metrics(scheme_result.to_dict(), nomtd_result.to_dict())

            multi_schemes[scheme] = scaled_scheme_result
        return multi_schemes
