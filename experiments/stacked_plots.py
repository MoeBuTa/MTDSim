import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from run_experiment import Experiment
import seaborn as sns

class StackedBarChart(Experiment):
    def __init__(self, epsilon, start_time, finish_time, mtd_interval, network_size, total_nodes, new_network, model, trial, result_head_path):
        super().__init__(epsilon, start_time, finish_time, mtd_interval, network_size, total_nodes, new_network, model, trial, result_head_path)
        self.weighted_data = None
    def plot_n_schemes(self, schemes_data, weights=None):
        """
        Plots multiple schemes on a stacked bar chart and calculates a weighted sum for each scheme.

        :param schemes_data: A dictionary where keys are scheme names and values are dictionaries of metrics.
        :param weights: A dictionary of weights for each metric. Defaults to 1 for all metrics if not provided.
        :param normalize: A method for normalization. Options are 'none', 'minmax', or 'zscore'. Defaults to 'none'.
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

        # Calculate the sum of the weighted metrics for each scheme
        weighted_df['sum'] = weighted_df.sum(axis=1)
        mean_sum = weighted_df['sum'].mean()
        std_sum = weighted_df['sum'].std()
        weighted_df['zscore'] = (weighted_df['sum'] - mean_sum) / std_sum

        min_sum = weighted_df['sum'].min()
        max_sum = weighted_df['sum'].max()
        weighted_df['minmax'] = (weighted_df['sum'] - min_sum) / (max_sum - min_sum)



        print("Normalized Weighted Metrics for Each Scheme:")
        print(weighted_df['sum'])

        # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Colors for each metric
        colors = plt.cm.tab20.colors

        # Initialize the bottom position for each metric as 0
        bottom = np.zeros(len(df))
        metrics = df.columns  # Include all columns

        # Plot stacked bar chart where schemes are on the x-axis, and metrics are stacked bars
        for i, metric in enumerate(metrics):
            ax.bar(df.index, weighted_df[metric], label=metric, color=colors[i % len(colors)], bottom=bottom)
            bottom += weighted_df[metric]  # Update bottom to stack bars



        # Add labels and title
        ax.set_ylabel('Stacked Metric Value' )
        ax.set_title('Comparison of Schemes with Weighted Metrics')
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Save the plot as a PNG file
        plt.savefig(f"{self.model}_{self.trial}.png")
        plt.show()

        self.weighted_data = weighted_df

    def normalized_chart(self,normalization = 'minmax'):
        result = self.weighted_data[normalization].sort_values()
         # Set up the figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(12, 9))  # Increase the figure size
        sns.barplot(x = result.index, y = result)
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.savefig(f"{self.model}_{self.trial}_normalized.png")
        plt.show()
