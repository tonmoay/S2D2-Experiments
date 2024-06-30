import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utils import load_results

def compute_statistics(data, data_index, add_runtime=False, runtime=0):

    processed_data = data[:, data_index]
    
    mean = np.mean(processed_data)
    std = np.std(processed_data)

    if add_runtime:
        return round((np.sum(processed_data) + runtime), 2), 0

    return mean, std

def create_excel_sheet(data, load_dir, excel_writer):
    """
    Create an Excel sheet for each data set, with grouped columns for 'S2D2 vs Baseline' and 'S2D2 vs S2D2'.
    """
    # Prepare data for DataFrame
    rows = []
    for city_name, city_data in data.items():
        num_nodes = city_data[6]['num_nodes']
        row = [city_name, num_nodes]

        # S2D2 vs Baseline data
        s2d2_vs_baseline_data = city_data[6]['sampled_s2d2_attacker_utility'][0][1]
        for data_index in range(3):  # Assuming 3 metrics: runtime, defender utility, attacker utility
            mean, std = compute_statistics(s2d2_vs_baseline_data, data_index, add_runtime=(data_index == 0), runtime=city_data[6]['sampled_s2d2_attacker_utility'][0][0])
            row.extend([mean, std])

        # S2D2 vs S2D2 data
        s2d2_vs_s2d2_data = city_data[6]['sampled_s2d2_attacker_utility'][1][1]
        for data_index in range(3):
            mean, std = compute_statistics(s2d2_vs_s2d2_data, data_index, add_runtime=(data_index == 0), runtime=city_data[6]['sampled_s2d2_attacker_utility'][1][0])
            row.extend([mean, std])

        rows.append(row)

    # Create DataFrame and save to Excel
    columns = ['City', 'Nodes',
               'RT. M.', 'A: S2D2 D:Baseline',
               'Def. U. M.', 'Def. U. Std.',
               'Att. U. M.', 'Att. U. Std.',
                'RTM', 'A: S2D2 D:S2D2',
                'Def. U. M.', 'Def. U. Std.',
                'Att. U. M.', 'Att. U. Std.']

    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(excel_writer, sheet_name=load_dir)  # Write with index

    # Adjust the Excel file to remove the index column
    workbook = excel_writer.book
    worksheet = workbook[load_dir]
    worksheet.delete_cols(1)  # Delete the first column which is the index

def plot_and_save_as_academic(data, ylabel, save_filename):
    """
    Plot data in an academic style and save the figure as a high-resolution PDF file.
    """
    # Set font details
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
    
    index_dict = {
        "Runtime (seconds)": 0,
        "Defender Utility": 1,
        "Attacker Utility": 2,
        "Baseline Defender Utility": 0  # New key for sampled_baseline_attacker_utility
    }
    
    colors = {
        's2d2_vs_baseline': 'blue',
        's2d2_vs_s2d2': 'green',
        'baseline_vs_baseline': 'red'  # Color for the new plot
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    x_values = []
    
    baseline_vs_baseline_means = []
    baseline_vs_baseline_stds = []
    
    s2d2_vs_baseline_means = []
    s2d2_vs_baseline_stds = []
    
    s2d2_vs_s2d2_means = []
    s2d2_vs_s2d2_stds = []

    for key in data.keys():
        num_nodes = data[key][6]['num_nodes']
        x_values.append(num_nodes)
        
        data_index = index_dict[ylabel]
        
        # Process S2D2 vs Baseline data
        s2d2_vs_baseline_data = data[key][6]['sampled_s2d2_attacker_utility'][0][1]
        s2d2_vs_baseline_mean, s2d2_vs_baseline_std = compute_statistics(
            s2d2_vs_baseline_data, data_index, 
            add_runtime=(ylabel == "Runtime (seconds)"), 
            runtime=data[key][6]['sampled_s2d2_attacker_utility'][0][0]
        )
        
        # Append the means and stds to the lists
        s2d2_vs_baseline_means.append(s2d2_vs_baseline_mean)
        s2d2_vs_baseline_stds.append(s2d2_vs_baseline_std)
        
        
        # Process S2D2 vs S2D2 data
        s2d2_vs_s2d2_data = data[key][6]['sampled_s2d2_attacker_utility'][1][1]
        s2d2_vs_s2d2_mean, s2d2_vs_s2d2_std = compute_statistics(
            s2d2_vs_s2d2_data, data_index, 
            add_runtime=(ylabel == "Runtime (seconds)"), 
            runtime=data[key][6]['sampled_s2d2_attacker_utility'][1][0]
        )
        
        # Append the means and stds to the lists
        s2d2_vs_s2d2_means.append(s2d2_vs_s2d2_mean)
        s2d2_vs_s2d2_stds.append(s2d2_vs_s2d2_std)
        
        # Process Baseline vs Baseline data
        # baseline_vs_baseline_data = data[key][6]['sampled_baseline_attacker_utility'][0][1]
        # baseline_vs_baseline_mean, baseline_vs_baseline_std = compute_statistics(
        #         baseline_vs_baseline_data, data_index,
        #         add_runtime=(ylabel == "Runtime (seconds)"),
        #         runtime=data[key][6]['sampled_baseline_attacker_utility'][1][0]
        #     )
        
        # Append the means and stds to the lists
        # baseline_vs_baseline_means.append(baseline_vs_baseline_mean)
        # baseline_vs_baseline_stds.append(baseline_vs_baseline_std)

    # Plot S2D2 vs Baseline
    # ax.scatter(x_values, s2d2_vs_baseline_means, color=colors['s2d2_vs_baseline'], label='A: S2D2 vs D: Baseline', marker='x')
    # ax.errorbar(x_values, s2d2_vs_baseline_means, yerr=s2d2_vs_baseline_stds, fmt='x', color=colors['s2d2_vs_baseline'], label='A: S2D2 vs D: Baseline')
    # ax.plot(x_values, baseline_means, color=colors['baseline'], label='S2D2 Attacker Baseline Defender', marker='o', linewidth=2)
    # for x, y, std in zip(x_values, baseline_means, baseline_stds):
    #     ax.text(x, y, f'{y:.2f}±{std:.2f}', color=colors['baseline'])

    # Plot S2D2 vs S2D2
    # ax.scatter(x_values, s2d2_vs_s2d2_means, color=colors['s2d2_vs_s2d2'], label='A: S2D2 vs D: S2D2', marker='x')
    # ax.errorbar(x_values, s2d2_vs_s2d2_means, yerr=s2d2_vs_s2d2_stds, fmt='x', color=colors['s2d2_vs_s2d2'], label='A: S2D2 vs D: S2D2')
    # ax.plot(x_values, our_method_means, color=colors['our_method'], label='S2D2 Attacker S2D2 Defender', marker='x', linewidth=2)
    # for x, y, std in zip(x_values, our_method_means, our_method_stds):
    #     ax.text(x, y, f'{y:.2f}±{std:.2f}', color=colors['our_method'])
    
    # Plot Baseline vs Baseline if applicable
    # ax.scatter(x_values, baseline_vs_baseline_means, color=colors['baseline_vs_baseline'], label='A: Baseline vs D: Baseline', marker='x')

    # Calculate the ratio
    value_ratio = [b / a for a, b in zip(s2d2_vs_s2d2_means, s2d2_vs_baseline_means)]
    
    # Scatter plot for the ratio
    ax.scatter(x_values, value_ratio, color='red', label='Ratio', marker='o', s=100)  # s is the marker size
    
    # Adding horizontal line at y=1
    ax.axhline(y=1, color='green', linestyle='--')
    
    ax.set_xlabel('Number of Nodes', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=20)
    
    fig.tight_layout()
    if save_filename:
        fig.savefig(save_filename, format='pdf', dpi=300, bbox_inches='tight')  # Save as high-resolution PDF
    plt.show()
    
if __name__ == "__main__":

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    adr_ratio = config['adr_ratio']
    experiment_name = config['experiment_name']
    perturbation = config['perturb_rewards']
    
    plots_directory = os.path.join('results', experiment_name, 'plots')
    os.makedirs(plots_directory, exist_ok=True)
    
    # Decide filename to save on rewards distribution
    comparison_results_filename = {
        # 'cities_manual_annotation': f"results/{experiment_name}/cross_compare_manual_adr_{adr_ratio}_perturb_{perturbation}.pkl",
        'cities_lognormal_dist': f"results/{experiment_name}/cross_compare_lognormal_adr_{adr_ratio}_perturb_{perturbation}.pkl",
        'cities_zipf_dist': f"results/{experiment_name}/cross_compare_zipf_adr_{adr_ratio}_perturb_{perturbation}.pkl"
    }
    
    excel_path = f"results/{experiment_name}/plots/comparison_data_adr_{adr_ratio}_perturb_{perturbation}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    
        for load_dir, file_name in comparison_results_filename.items():
            # Load the data
            cross_comparison_data = load_results(filename=file_name)
            
            y_labels = ['Runtime (seconds)', 'Defender Utility', 'Attacker Utility']
            
            for ylabel in y_labels:
                # Specify the save_filename
                save_filename = f"results/{experiment_name}/plots/{load_dir}_adr_{adr_ratio}_{ylabel.replace(' ', '_')}_perturb_{perturbation}.pdf"
                
                # Call the plot_and_save_as_academic function
                plot_and_save_as_academic(cross_comparison_data, ylabel, save_filename)
            
                # Create an Excel sheet for each set of data
                create_excel_sheet(cross_comparison_data, load_dir, writer)
