import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from run_experiment import Experiment
import numpy as np
from mtdnetwork.mtd.completetopologyshuffle import CompleteTopologyShuffle
from mtdnetwork.mtd.ipshuffle import IPShuffle
from mtdnetwork.mtd.osdiversity import OSDiversity
from mtdnetwork.mtd.servicediversity import ServiceDiversity
from matplotlib.colors import to_rgb, to_hex
import matplotlib.ticker as ticker
class StackedBarChart(Experiment):
    def __init__(self, metric, epsilon, start_time, finish_time, mtd_interval, network_size, total_nodes, new_network, model, trial, result_head_path):
        super().__init__(metric, epsilon, start_time, finish_time, mtd_interval, network_size, total_nodes, new_network, model, trial, result_head_path)
        self.weighted_data = None
        # Define strategies
        self.mtd_strategies_dict = {
            'CompleteTopologyShuffle': CompleteTopologyShuffle,
            'IPShuffle': IPShuffle,
            'OSDiversity': OSDiversity,
            'ServiceDiversity': ServiceDiversity
        }




    def calculate_weighted_sum(self, weighted_df):
        """
        This method calculates the sum of weighted metrics for each scheme and performs
        additional calculations such as z-score and min-max normalization.
        """
        # Calculate the sum of the weighted metrics for each scheme
        weighted_df['sum'] = weighted_df.sum(axis=1)

        # Calculate mean, standard deviation, and z-score of the sums
        mean_sum = weighted_df['sum'].mean()
        std_sum = weighted_df['sum'].std()
        weighted_df['zscore'] = (weighted_df['sum'] - mean_sum) / std_sum

        # Perform min-max normalization on the sums
        min_sum = weighted_df['sum'].min()
        max_sum = weighted_df['sum'].max()
        weighted_df['minmax'] = (weighted_df['sum'] - min_sum) / (max_sum - min_sum)

        return weighted_df


    def plot_n_schemes(self, schemes_data, weights=None, title = 'Comparison of Schemes with Weighted Metrics'):
        """
        This method plots a stacked bar chart comparing schemes based on weighted metrics.
        """
        # Convert the schemes_data to a DataFrame and transpose so that rows are schemes and columns are metrics
        df = pd.DataFrame(schemes_data).T

        # Set default weights to 1 if no custom weights are provided
        if weights is None:
            weights = {metric: 1 for metric in df.columns}

        # Ensure that weights are a Pandas Series for easy multiplication with DataFrame columns
        weights = pd.Series(weights)

        # Check if all metrics in the DataFrame have corresponding weights
        missing_weights = set(df.columns) - set(weights.index)
        if missing_weights:
            raise ValueError(f"Missing weights for metrics: {missing_weights}")

        # Calculate weighted metrics
        weighted_df = df * weights

        # Use the newly created method to modify the weighted_df and calculate sum, z-score, and min-max normalization
        weighted_df = self.calculate_weighted_sum(weighted_df)

        print("Normalized Weighted Metrics for Each Scheme:")
        print(weighted_df['sum'])

        # Sort DataFrame by 'sum'
        weighted_df_sorted = weighted_df.sort_values(by='sum', ascending=True)

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(16, 12))

        # Colors for each metric
        colors = plt.cm.tab20.colors

        # Initialize the bottom position for each metric as 0
        bottom = np.zeros(len(weighted_df_sorted))
        metrics = df.columns  # Include all columns

        # Plot stacked bar chart where schemes are on the x-axis, and metrics are stacked bars
        for i, metric in enumerate(metrics):
            ax.bar(weighted_df_sorted.index, weighted_df_sorted[metric], label=metric, color=colors[i % len(colors)], bottom=bottom)
            bottom += weighted_df_sorted[metric]  # Update bottom to stack bars

        # Add labels and title
        ax.set_ylabel('Stacked Metric Value')

        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=8)

        # Set y-axis to have intervals of 0.5
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}.png")
        plt.show()

        # Store the weighted data
        self.weighted_data = weighted_df


    def normalized_chart(self, normalization='minmax'):
        result = self.weighted_data[normalization].sort_values()

        # Define colors for each combination
        colors = {
            'Static + Shuffling': 'lightblue',
            'Static + Diversity': 'lightgreen',
            'Dynamic + Shuffling': 'salmon',
            'Dynamic + Diversity': 'orange',
            'All Features + Shuffling': 'pink',  # New color for all features + Shuffling
            'All Features + Diversity': 'purple',  # New color for all features + Diversity
        }

        # Feature classification using substring matching
        static_substrings = ["host_compromise_ratio", "exposed_endpoints", "attack_path_exposure", "risk"]
        dynamic_substrings = ["overall_asr_avg", "roa", "shortest_path_variability", "mtd_freq", "overall_mttc_avg", "time_since_last_mtd"]

        # MTD strategy classification
        shuffling_strategies = ['CompleteTopologyShuffle', 'IPShuffle']
        diversity_strategies = ['OSDiversity', 'ServiceDiversity']

        # Determine feature and strategy types
        feature_colors = {}
        for feature in result.index:
            feature_lower = feature.lower()
            
            # Check if feature belongs to static or dynamic categories
            if any(substring in feature_lower for substring in static_substrings):
                feature_type = 'Static'
            elif any(substring in feature_lower for substring in dynamic_substrings):
                feature_type = 'Dynamic'
            else:
                feature_type = 'Unknown'

            # Identify the MTD strategy from the feature name
            strategy_type = 'Shuffling' if any(strategy in feature for strategy in shuffling_strategies) else 'Diversity'
            
            # Determine if the feature is categorized as 'All Features'
            if 'all_features' in feature.lower():
                combination = f"All Features + {strategy_type}"
            else:
                combination = f"{feature_type} + {strategy_type}"

            feature_colors[feature] = colors.get(combination, 'grey')  # Default color for unmatched combinations

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(12, 9))  # Increase the figure size

        # if normalization == 'sum':
        #     # Add horizontal line at y = 7.230848
        #     ax.axhline(y=7.230848, color='red', linestyle='--', label='Simultaneous Scheme')
        # Plot bars with colors based on feature and strategy type
        color_list = [feature_colors.get(feature, 'grey') for feature in result.index]
        sns.barplot(x=result.index, y=result, palette=color_list, ax=ax)

        # Add legend
        handles = [plt.Line2D([0], [0], color=color, lw=10) for color in colors.values()]
        labels = list(colors.keys())
        ax.legend(handles, labels, title='Feature + Strategy Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=8)

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}_normalized.png")
        plt.show()

        
    

    def all_combinations_chart(self, normalization='minmax'):
        result = self.weighted_data[normalization].sort_values()

        # Define colors for each combination
        colors = {
            'host_compromise_ratio + CompleteTopologyShuffle': 'lightblue',
            'host_compromise_ratio + IPShuffle': 'lightcoral',
            'host_compromise_ratio + OSDiversity': 'lightgreen',
            'host_compromise_ratio + ServiceDiversity': 'lightpink',
            'exposed_endpoints + CompleteTopologyShuffle': 'salmon',
            'exposed_endpoints + IPShuffle': 'orange',
            'exposed_endpoints + OSDiversity': 'yellowgreen',
            'exposed_endpoints + ServiceDiversity': 'gold',
            'attack_path_exposure + CompleteTopologyShuffle': 'skyblue',
            'attack_path_exposure + IPShuffle': 'red',
            'attack_path_exposure + OSDiversity': 'lightseagreen',
            'attack_path_exposure + ServiceDiversity': 'peachpuff',
            'risk + CompleteTopologyShuffle': 'powderblue',
            'risk + IPShuffle': 'tomato',
            'risk + OSDiversity': 'darkseagreen',
            'risk + ServiceDiversity': 'lightgoldenrodyellow',
            'overall_asr_avg + CompleteTopologyShuffle': 'lightsteelblue',
            'overall_asr_avg + IPShuffle': 'darkorange',
            'overall_asr_avg + OSDiversity': 'lightyellow',
            'overall_asr_avg + ServiceDiversity': 'darkgoldenrod',
            'roa + CompleteTopologyShuffle': 'deepskyblue',
            'roa + IPShuffle': 'firebrick',
            'roa + OSDiversity': 'mediumseagreen',
            'roa + ServiceDiversity': 'khaki',
            'shortest_path_variability + CompleteTopologyShuffle': 'lightcyan',
            'shortest_path_variability + IPShuffle': 'orangered',
            'shortest_path_variability + OSDiversity': 'darkolivegreen',
            'shortest_path_variability + ServiceDiversity': 'moccasin',
            'mtd_freq + CompleteTopologyShuffle': 'lightgray',
            'mtd_freq + IPShuffle': 'darkviolet',
            'mtd_freq + OSDiversity': 'mediumslateblue',
            'mtd_freq + ServiceDiversity': 'lightcoral',
            'overall_mttc_avg + CompleteTopologyShuffle': 'lavender',
            'overall_mttc_avg + IPShuffle': 'crimson',
            'overall_mttc_avg + OSDiversity': 'mediumvioletred',
            'overall_mttc_avg + ServiceDiversity': 'lightpink',
            'time_since_last_mtd + CompleteTopologyShuffle': 'aliceblue',
            'time_since_last_mtd + IPShuffle': 'indigo',
            'time_since_last_mtd + OSDiversity': 'darkkhaki',
            'time_since_last_mtd + ServiceDiversity': 'lightyellow',
            'all_features + CompleteTopologyShuffle': 'grey',
            'all_features + IPShuffle': 'grey',
            'all_features + OSDiversity': 'grey',
            'all_features + ServiceDiversity': 'grey',
        }

        # Define static and dynamic feature lists
        static_features = ["host_compromise_ratio", "exposed_endpoints", "attack_path_exposure", "risk"]
        dynamic_features = ["overall_asr_avg", "roa", "shortest_path_variability", "mtd_freq", "overall_mttc_avg", "time_since_last_mtd"]

        # Define MTD strategy types
        shuffling_strategies = ['CompleteTopologyShuffle', 'IPShuffle']
        diversity_strategies = ['OSDiversity', 'ServiceDiversity']

        # Initialize DataFrame for plotting
        plot_data = pd.DataFrame(index=result.index, columns=['Value'])

        # Determine feature and strategy types for each feature in result.index
        feature_colors = {}
        for feature in result.index:
            feature_lower = feature.lower()
            # Determine the feature name
            feature_name = next((f for f in static_features + dynamic_features if f.lower() in feature_lower), 'Unknown')
            
            # Determine the strategy type
            if any(strategy in feature for strategy in shuffling_strategies):
                strategy_type = 'CompleteTopologyShuffle' if 'CompleteTopologyShuffle' in feature else 'IPShuffle'
            elif any(strategy in feature for strategy in diversity_strategies):
                strategy_type = 'OSDiversity' if 'OSDiversity' in feature else 'ServiceDiversity'
            else:
                strategy_type = 'Unknown'

            combination = f"{feature_name} + {strategy_type}"
            feature_colors[feature] = colors.get(combination, 'grey')  # Default color for unmatched combinations

        # Create a DataFrame for all combinations for the legend
        all_combinations = pd.DataFrame(index=[
            f"{feature} + CompleteTopologyShuffle" for feature in static_features + dynamic_features] +
            [f"{feature} + IPShuffle" for feature in static_features + dynamic_features] +
            [f"{feature} + OSDiversity" for feature in static_features + dynamic_features] +
            [f"{feature} + ServiceDiversity" for feature in static_features + dynamic_features] +
            [f"all_features + CompleteTopologyShuffle", f"all_features + IPShuffle", 
            f"all_features + OSDiversity", f"all_features + ServiceDiversity"],
            columns=['Value']
        )
        all_combinations.fillna(0, inplace=True)  # Fill NaN values with 0 for plotting

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(14, 10))  # Increase the figure size

        # Plot bars with colors based on feature and strategy type
        color_list = [feature_colors.get(feature, 'grey') for feature in result.index]
        sns.barplot(x=result.index, y=result, palette=color_list, ax=ax)

        # Create legend with all combinations
        handles = [plt.Line2D([0], [0], color=color, lw=10) for color in colors.values()]
        labels = list(colors.keys())
        ax.legend(handles, labels, title='Feature + Strategy Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=8)

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}_all_combinations.png")
        plt.show()



    def features_only_chart(self, normalization='minmax'):
        result = self.weighted_data[normalization].sort_values()

        # Define a set of distinguishable colors for each feature
        base_colors = {
            'host_compromise_ratio': '#1f77b4',  # Blue
            'exposed_endpoints': '#ff7f0e',  # Orange
            'attack_path_exposure': '#2ca02c',  # Green
            'risk': '#d62728',  # Red
            'overall_asr_avg': '#9467bd',  # Purple
            'roa': '#8c564b',  # Brown
            'shortest_path_variability': '#e377c2',  # Pink
            'mtd_freq': '#7f7f7f',  # Grey
            'overall_mttc_avg': '#bcbd22',  # Olive
            'time_since_last_mtd': '#17becf',  # Teal
            'all_features': '#d3d3d3',  # Light gray
        }

        # Initialize DataFrame for plotting
        plot_data = pd.DataFrame(index=result.index, columns=['Value'])

        # Determine feature colors for each feature in result.index
        feature_colors = {}
        for feature in result.index:
            # Determine the feature name
            feature_name = next((f for f in base_colors.keys() if f.lower() in feature.lower()), 'Unknown')
            feature_colors[feature] = base_colors.get(feature_name, '#d3d3d3')  # Default to light gray for unmatched

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(14, 10))  # Increase the figure size

        # Plot bars with colors based on feature
        color_list = [feature_colors.get(feature, '#d3d3d3') for feature in result.index]
        sns.barplot(x=result.index, y=result, palette=color_list, ax=ax)

        # Create legend with all features
        handles = [plt.Line2D([0], [0], color=color, lw=10) for color in base_colors.values()]
        labels = list(base_colors.keys())
        ax.legend(handles, labels, title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=8)

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}_features_only.png")
        plt.show()



    def feature_type_only_chart(self, normalization='minmax'):
        result = self.weighted_data[normalization].sort_values()

        # Define colors for static and dynamic features
        colors = {
            'Static': '#1f77b4',  # Blue for static features
            'Dynamic': '#ff7f0e',  # Orange for dynamic features
            "All Features": '#d3d3d3',
        }

        # Define lists of static and dynamic features
        static_features = [
            'host_compromise_ratio', 'exposed_endpoints', 'attack_path_exposure', 'risk'
        ]
        dynamic_features = [
            'overall_asr_avg', 'roa', 'shortest_path_variability', 'mtd_freq', 'overall_mttc_avg', 'time_since_last_mtd'
        ]

        # Initialize DataFrame for plotting
        plot_data = pd.DataFrame(index=result.index, columns=['Value'])

        # Determine feature colors for each feature in result.index
        feature_colors = {}
        for feature in result.index:
            # Determine the feature name
            feature_name = next((f for f in static_features + dynamic_features if f.lower() in feature.lower()), 'Unknown')
            if feature_name in static_features:
                color_key = 'Static'
            elif feature_name in dynamic_features:
                color_key = 'Dynamic'
            else:
                color_key = 'All Features'

            feature_colors[feature] = colors.get(color_key, '#d3d3d3')  # Default to light gray for unmatched

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(14, 10))  # Increase the figure size

        # Plot bars with colors based on feature category
        color_list = [feature_colors.get(feature, '#d3d3d3') for feature in result.index]
        sns.barplot(x=result.index, y=result, palette=color_list, ax=ax)

        # Create legend with static and dynamic categories
        handles = [plt.Line2D([0], [0], color=color, lw=10) for color in colors.values()]
        labels = list(colors.keys())
        ax.legend(handles, labels, title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=8)

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}_features_tyoe_only.png")
        plt.show()



   

    def mtd_techniques_chart(self, normalization='minmax'):
        result = self.weighted_data[normalization].sort_values()

        # Define colors for each MTD technique set
        base_colors = {
            'CompleteTopologyShuffle': '#ff9999',  # Light red
            'IPShuffle': '#ffcc99',  # Light orange
            'OSDiversity': '#99ff99',  # Light green
            'ServiceDiversity': '#99ccff',  # Light blue
            'All MTD Techniques': '#d3d3d3'  # Light gray for the all MTD set
        }

        # Initialize DataFrame for plotting
        plot_data = pd.DataFrame(index=result.index, columns=['Value'])

        # Determine MTD technique colors for each feature in result.index
        mtd_colors = {}
        for feature in result.index:
            # Determine the MTD technique name
            mtd_name = next((tech for tech in base_colors.keys() if tech in feature), 'Unknown')
            if mtd_name == 'Unknown':
                color_key = "All MTD Techniques"
            else:
                color_key = mtd_name
            
            mtd_colors[feature] = base_colors.get(color_key, '#d3d3d3')  # Default to light gray for unmatched

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(14, 10))  # Increase the figure size

        # Plot bars with colors based on MTD technique
        color_list = [mtd_colors.get(feature, '#d3d3d3') for feature in result.index]
        sns.barplot(x=result.index, y=result, palette=color_list, ax=ax)

        # Create legend with all MTD techniques
        handles = [plt.Line2D([0], [0], color=color, lw=10) for color in base_colors.values()]
        labels = list(base_colors.keys())
        ax.legend(handles, labels, title='MTD Technique', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=8)

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}_mtd_techniques.png")
        plt.show()


