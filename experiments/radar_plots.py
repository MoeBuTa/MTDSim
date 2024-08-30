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
from run_experiment import Experiment


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class RadarPlot(Experiment):
    def __init__(self, epsilon, start_time, finish_time, mtd_interval, network_size,total_nodes, new_network,  model, trial, result_head_path):
        super().__init__(epsilon, start_time, finish_time, mtd_interval, network_size,total_nodes, new_network,  model, trial, result_head_path)

    def plot_n_schemes(self, schemes_data):
        """
        Plots multiple schemes on a radar chart.
        
        :param schemes_data: A dictionary where keys are scheme names and values are dictionaries of metrics.
        """
        scheme_names = list(schemes_data.keys())
        first_scheme = schemes_data[scheme_names[0]]
        labels = list(first_scheme.keys())
        num_vars = len(labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Use colormap for colors
        colors = plt.cm.tab20.colors
        legend_patches = []

        for i, scheme_name in enumerate(scheme_names):
            metrics_values = schemes_data[scheme_name]
            values = list(metrics_values.values())
            values += values[:1]  # Close the circle

            # Plot data
            ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)
            ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=scheme_name)

            # Add score labels
            for j in range(num_vars):
                angle = angles[j]
                value = values[j]
                ax.text(angle, value + 0.05, f'{value:.2f}', horizontalalignment='center', size=10, color='black')

        # Add labels
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Create legend
        legend_patches = [Patch(color=colors[i % len(colors)], label=scheme_name) for i, scheme_name in enumerate(scheme_names)]
        plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.1, 1.1))

        plt.show()

