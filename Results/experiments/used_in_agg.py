#!/usr/bin/env python3
"""
used_in_agg.py - Create line graph showing clients used in aggregation over rounds for different timeout multipliers

Reads R_included_agg.log files from folders with format {dataset}_wait_{N} and creates a line graph
showing how the number of clients used in aggregation evolves over training rounds for each timeout multiplier.
The dataset (mnist or cifar) is automatically detected from the directory structure.

Usage:
    python used_in_agg.py
    python used_in_agg.py -b /path/to/custom/directory
    python used_in_agg.py -s moving_avg -w 15
    python used_in_agg.py -s exponential -w 5
    python used_in_agg.py -s subsample -w 10
    python used_in_agg.py -s none
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def extract_n_from_folder(folder_name: str, dataset: str) -> int:
    """Extract N from folder name like 'mnist_wait_5' or 'cifar_wait_10'"""
    pattern = rf'{dataset}_wait_(\d+)'
    match = re.search(pattern, folder_name)
    if match:
        return int(match.group(1))
    return None


def extract_dataset_from_path(base_dir: str) -> str:
    """Extract dataset name from the base directory path by looking for subdirectories starting with mnist or cifar"""
    if not os.path.exists(base_dir):
        return None
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if item.startswith('mnist'):
                return 'mnist'
            elif item.startswith('cifar'):
                return 'cifar'
    
    return None


def parse_included_agg_log(log_file_path: str, smoothing_method: str = 'moving_avg', window_size: int = 10) -> List[Tuple[int, float]]:
    """
    Parse R_included_agg.log file and return list of (round, avg_clients_used) tuples
    
    Args:
        log_file_path: Path to R_included_agg.log file
        
    Returns:
        List of (round, avg_clients_used) tuples averaged across all executions
    """
    executions = []  # List of executions, each execution is a list of clients_used_per_round
    current_execution = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('//'):
                    continue
                
                # Check for end of execution marker
                if line == '$ END OF EXECUTION $':
                    if current_execution:
                        executions.append(current_execution)
                        current_execution = []
                    continue
                
                # Parse number of clients used in this round
                try:
                    clients_used = int(line)
                    current_execution.append(clients_used)
                except ValueError:
                    continue
        
        # Add the last execution if it doesn't end with the marker
        if current_execution:
            executions.append(current_execution)
            
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    if not executions:
        return []
    
    # Find the maximum number of rounds across all executions
    max_rounds = max(len(execution) for execution in executions)
    
    # Calculate average clients used for each round across executions
    averaged_data = []
    for round_num in range(1, max_rounds + 1):
        clients_for_round = []
        for execution in executions:
            # Check if this execution has data for this round (0-indexed)
            if round_num - 1 < len(execution):
                clients_for_round.append(execution[round_num - 1])
        
        if clients_for_round:
            avg_clients = np.mean(clients_for_round)
            averaged_data.append((round_num, avg_clients))
    
    # Apply smoothing based on method
    if smoothing_method == 'moving_avg' and len(averaged_data) > window_size:
        smoothed_data = []
        for i in range(len(averaged_data)):
            if i < window_size - 1:
                # For the first few points, use expanding window
                window_start = 0
                window_end = i + 1
            else:
                # Use rolling window
                window_start = i - window_size + 1
                window_end = i + 1
            
            window_values = [averaged_data[j][1] for j in range(window_start, window_end)]
            smoothed_avg = np.mean(window_values)
            smoothed_data.append((averaged_data[i][0], smoothed_avg))
        
        averaged_data = smoothed_data
    
    elif smoothing_method == 'exponential':
        # Exponential moving average
        if averaged_data:
            alpha = 2.0 / (window_size + 1)  # Smoothing factor
            smoothed_data = [averaged_data[0]]  # Start with first point
            
            for i in range(1, len(averaged_data)):
                smoothed_value = alpha * averaged_data[i][1] + (1 - alpha) * smoothed_data[i-1][1]
                smoothed_data.append((averaged_data[i][0], smoothed_value))
            
            averaged_data = smoothed_data
    
    elif smoothing_method == 'subsample':
        # Subsample every N points
        step = max(1, window_size // 2)
        averaged_data = averaged_data[::step]
    
    return averaged_data


def collect_data(base_dir: str, smoothing_method: str = 'moving_avg', window_size: int = 10) -> Tuple[str, List[Tuple[int, List[Tuple[int, float]]]]]:
    """Collect aggregation data from all folders and determine dataset automatically"""
    # First, determine the dataset from the directory structure
    dataset = extract_dataset_from_path(base_dir)
    if not dataset:
        print(f"Error: Could not determine dataset (mnist or cifar) from directory structure in {base_dir}")
        return None, []
    
    print(f"Detected dataset: {dataset.upper()}")
    
    data_points = []
    
    # Iterate through all subdirectories that match the pattern
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Extract N from folder name
        n_multiplier = extract_n_from_folder(folder_name, dataset)
        if n_multiplier is None:
            continue
        
        # Look for R_included_agg.log
        log_file = os.path.join(folder_path, "R_included_agg.log")
        if not os.path.exists(log_file):
            print(f"Warning: R_included_agg.log not found in {folder_path}")
            continue
        
        print(f"Processing {folder_name} (timeout multiplier: {n_multiplier})")
        
        # Parse aggregation data
        agg_data = parse_included_agg_log(log_file, smoothing_method, window_size)
        
        if not agg_data:
            print(f"Warning: No aggregation data found in {log_file}")
            continue
        
        print(f"  Found {len(agg_data)} rounds of data")
        
        data_points.append((n_multiplier, agg_data))
    
    # Sort by timeout multiplier
    data_points.sort(key=lambda x: x[0])
    
    return dataset, data_points


def create_aggregation_graph(data_points: List[Tuple[int, List[Tuple[int, float]]]], dataset: str) -> None:
    """Create line graph showing clients used in aggregation over rounds for different timeout multipliers"""
    if not data_points:
        print("No data points to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for different timeout multipliers
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
              'magenta', 'yellow', 'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 
              'maroon', 'lightcoral', 'darkgray', 'darkolivegreen', 'darkcyan', 'indigo', 'gold',
              'navy', 'crimson', 'forestgreen', 'chocolate', 'mediumpurple', 'sienna', 'hotpink',
              'dimgray', 'yellowgreen', 'teal', 'plum', 'khaki']
    
    # Plot each timeout multiplier as a separate line
    for i, (n_multiplier, agg_data) in enumerate(data_points):
        if not agg_data:
            continue
            
        # Extract rounds and clients used
        rounds = [round_num for round_num, _ in agg_data]
        clients_used = [clients for _, clients in agg_data]
        
        # Plot the line
        color = colors[i % len(colors)]
        plt.plot(rounds, clients_used, 
                #marker='o', 
                linewidth=2, 
                #markersize=6, 
                label=f'Timeout x{n_multiplier}', 
                color=color)
        
        print(f"Timeout x{n_multiplier}: {len(rounds)} rounds, final clients used: {clients_used[-1]:.2f}")
    
    # Customize the plot
    plt.xlabel('Round', fontsize=22)
    plt.ylabel('Clients Used in Aggregation', fontsize=22)
    plt.title(f'RByz Clients Used in Aggregation - {dataset.upper()}', fontsize=22)
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Set Y-axis to show all integer values from 0 to max
    all_clients = []
    for _, agg_data in data_points:
        for round_num, clients in agg_data:
            all_clients.append(clients)
    
    if all_clients:
        min_clients = max(0, int(np.floor(min(all_clients))))
        max_clients = int(np.ceil(max(all_clients)))
        plt.yticks(range(min_clients, max_clients + 1))
        plt.ylim(min_clients - 0.2, max_clients + 0.2)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add legend inside the plot area at upper right
    plt.legend(loc='upper right', fontsize=22)
    
    # Set reasonable axis limits
    all_rounds = []
    
    for _, agg_data in data_points:
        for round_num, clients in agg_data:
            all_rounds.append(round_num)
    
    if all_rounds:
        # Set reasonable X-axis limits
        min_round = min(all_rounds)
        max_round = max(all_rounds)
        plt.xlim(min_round * 0.8, max_round * 1.2)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    output_path = os.path.join(output_dir, f"used_in_agg_{dataset}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create line graph showing clients used in aggregation over rounds for different timeout multipliers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python used_in_agg.py
    python used_in_agg.py -b /path/to/custom/directory
    python used_in_agg.py --base-dir /path/to/custom/directory
    python used_in_agg.py -s moving_avg -w 15
    python used_in_agg.py -s exponential -w 5
    python used_in_agg.py -s subsample -w 10
    python used_in_agg.py -s none  # No smoothing
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/timeouts_real",
                       help='Base directory containing dataset folders (default: /home/bustaman/rbyz/Results/keep/timeouts_real)')
    
    parser.add_argument('-s', '--smoothing', 
                       choices=['moving_avg', 'exponential', 'subsample', 'none'],
                       default='moving_avg',
                       help='Smoothing method to apply (default: moving_avg)')
    
    parser.add_argument('-w', '--window-size', 
                       type=int,
                       default=10,
                       help='Window size for smoothing (default: 10)')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    print(f"Smoothing method: {args.smoothing} (window size: {args.window_size})")
    print(f"Collecting aggregation data from log files...")
    
    # Collect data and automatically determine dataset
    dataset, data_points = collect_data(args.base_dir, args.smoothing, args.window_size)
    
    if not dataset:
        print("Could not determine dataset. Exiting.")
        sys.exit(1)
    
    if not data_points:
        print("No data found. Exiting.")
        sys.exit(1)
    
    print(f"\nFound data for {len(data_points)} configurations:")
    for n_multiplier, agg_data in data_points:
        print(f"  Timeout x{n_multiplier}: {len(agg_data)} rounds")
    
    print(f"\nCreating aggregation graph for {dataset.upper()}...")
    create_aggregation_graph(data_points, dataset)


if __name__ == "__main__":
    main()
