import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_metric_ratio(evaluations, metric_column):
    combined_df = pd.concat(evaluations)
    grouped_df = combined_df.groupby('times')[metric_column].agg(['mean', 'std']).reset_index()

    # Plot
    plt.errorbar(grouped_df['times'], grouped_df['mean'], yerr=grouped_df['std'], fmt='o', ecolor='r', capthick=2, capsize=5)
    plt.plot(grouped_df['times'], grouped_df['mean'], linestyle='-', color='b')  # Add line plot

    plt.xlabel('Times')
    plt.ylabel(f'Average {metric_column.replace("_", " ").title()}')
    plt.title(f'Average {metric_column.replace("_", " ").title()} vs Times with Error Bars')
    plt.grid(True)
    plt.show()