#!/usr/bin/env python3
"""
acc_evolution_vd_size.py - Create line graph showing accuracy evolution over rounds for different validation data percentages

Reads R_acc.log files from folders with format {dataset}_<percentage>%vd and creates a line graph
showing how accuracy evolves over training rounds for each validation data percentage.
The dataset (mnist or cifar) is automatically detected from the directory structure.

Usage:
    python acc_evolution_vd_size.py
    python acc_evolution_vd_size.py -b /path/to/custom/directory
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


def extract_percentage_from_folder(folder_name: str, dataset: str) -> float:
    """Extract percentage from folder name like 'mnist_25%vd' or 'cifar_25%vd'"""
    pattern = rf'{dataset}_(\d+(?:\.\d+)?)%vd'
    match = re.search(pattern, folder_name)
    if match:
        return float(match.group(1))
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


def parse_acc_log(log_file_path: str) -> List[Tuple[int, float]]:
    """
    Parse R_acc.log file and return list of (round, accuracy) tuples for each execution
    
    Args:
        log_file_path: Path to R_acc.log file
        
    Returns:
        List of (round, accuracy) tuples averaged across all executions
    """
    executions = []  # List of executions, each execution is a list of (round, accuracy)
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
                
                # Parse round and accuracy
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        round_num = int(parts[0])
                        accuracy = float(parts[1])
                        current_execution.append((round_num, accuracy))
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
    
    # Average accuracies across executions for each round
    # First, find all unique rounds across all executions
    all_rounds = set()
    for execution in executions:
        for round_num, _ in execution:
            all_rounds.add(round_num)
    
    # Sort rounds
    sorted_rounds = sorted(all_rounds)
    
    # Calculate average accuracy for each round
    averaged_data = []
    for round_num in sorted_rounds:
        accuracies_for_round = []
        for execution in executions:
            # Find accuracy for this round in this execution
            for r, acc in execution:
                if r == round_num:
                    accuracies_for_round.append(acc)
                    break
        
        if accuracies_for_round:
            avg_accuracy = np.mean(accuracies_for_round)
            averaged_data.append((round_num, avg_accuracy))
    
    return averaged_data


def collect_data(base_dir: str) -> Tuple[str, List[Tuple[float, List[Tuple[int, float]]]]]:
    """Collect accuracy evolution data from all folders and determine dataset automatically"""
    # First, determine the dataset from the directory structure
    dataset = extract_dataset_from_path(base_dir)
    if not dataset:
        print(f"Error: Could not determine dataset (mnist or cifar) from directory structure in {base_dir}")
        return None, []
    
    print(f"Detected dataset: {dataset.upper()}")
    
    base_path = os.path.join(base_dir, f"{dataset}*")  # This won't work with glob, let's fix it
    
    # Instead, let's look for directories that start with the dataset name
    data_points = []
    
    # Iterate through all subdirectories that start with the dataset name
    for folder_name in os.listdir(base_dir):
        if not folder_name.startswith(dataset):
            continue
            
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Extract percentage from folder name
        percentage = extract_percentage_from_folder(folder_name, dataset)
        if percentage is None:
            print(f"Warning: Could not extract percentage from folder '{folder_name}' for dataset '{dataset}'")
            continue
        
        # Look for R_acc.log
        log_file = os.path.join(folder_path, "R_acc.log")
        if not os.path.exists(log_file):
            print(f"Warning: R_acc.log not found in {folder_path}")
            continue
        
        print(f"Processing {folder_name} (percentage: {percentage}%)")
        
        # Parse accuracy evolution data
        acc_evolution = parse_acc_log(log_file)
        
        if not acc_evolution:
            print(f"Warning: No accuracy evolution data found in {log_file}")
            continue
        
        print(f"  Found {len(acc_evolution)} rounds of data")
        
        data_points.append((percentage, acc_evolution))
    
    # Sort by percentage
    data_points.sort(key=lambda x: x[0])
    
    return dataset, data_points


def create_evolution_graph(data_points: List[Tuple[float, List[Tuple[int, float]]]], dataset: str) -> None:
    """Create line graph showing accuracy evolution over rounds for different validation percentages"""
    if not data_points:
        print("No data points to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for different percentages
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
              'magenta', 'yellow', 'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 
              'maroon', 'lightcoral', 'darkgray', 'darkolivegreen', 'darkcyan', 'indigo', 'gold',
              'navy', 'crimson', 'forestgreen', 'chocolate', 'mediumpurple', 'sienna', 'hotpink',
              'dimgray', 'yellowgreen', 'teal', 'plum', 'khaki']
    
    # Plot each percentage as a separate line
    for i, (percentage, acc_evolution) in enumerate(data_points):
        if not acc_evolution:
            continue
            
        # Extract rounds and accuracies
        rounds = [round_num for round_num, _ in acc_evolution]
        accuracies = [accuracy for _, accuracy in acc_evolution]
        
        # Plot the line
        color = colors[i % len(colors)]
        plt.plot(rounds, accuracies, 
                marker='o', 
                linewidth=2, 
                markersize=4, 
                label=f'{percentage}%vd', 
                color=color)
        
        print(f"{percentage}%vd: {len(rounds)} rounds, final accuracy: {accuracies[-1]:.2f}")
    
    # Customize the plot
    plt.xlabel('Round', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.title(f'RByz Accuracy Evolution - {dataset.upper()}', fontsize=22)

    # Set X-axis to logarithmic scale
    plt.xscale('log')
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add legend inside the plot area at bottom right
    plt.legend(loc='lower right', fontsize=22)
    
    # Set reasonable axis limits
    all_rounds = []
    all_accuracies = []
    
    for _, acc_evolution in data_points:
        for round_num, accuracy in acc_evolution:
            all_rounds.append(round_num)
            all_accuracies.append(accuracy)
    
    if all_rounds and all_accuracies:
        # For log scale on X-axis, ensure minimum is at least 1
        min_round = max(min(all_rounds), 1)
        max_round = max(all_rounds)
        plt.xlim(min_round * 0.8, max_round * 1.2)
        plt.ylim(min(all_accuracies) - 2, max(all_accuracies) + 2)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    output_path = os.path.join(output_dir, f"acc_evolution_vd_size_{dataset}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create line graph showing accuracy evolution over rounds for different validation data percentages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python acc_evolution_vd_size.py
    python acc_evolution_vd_size.py -b /path/to/custom/directory
    python acc_evolution_vd_size.py --base-dir /path/to/custom/directory
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/acc_vs_test_size_nodev",
                       help='Base directory containing dataset folders (default: /home/bustaman/rbyz/Results/keep/acc_vs_test_size_nodev)')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    print(f"Collecting accuracy evolution data from log files...")
    
    # Collect data and automatically determine dataset
    dataset, data_points = collect_data(args.base_dir)
    
    if not dataset:
        print("Could not determine dataset. Exiting.")
        sys.exit(1)
    
    if not data_points:
        print("No data found. Exiting.")
        sys.exit(1)
    
    print(f"\nFound data for {len(data_points)} configurations:")
    for percentage, acc_evolution in data_points:
        print(f"  {percentage}%vd: {len(acc_evolution)} rounds")
    
    print(f"\nCreating accuracy evolution graph for {dataset.upper()}...")
    create_evolution_graph(data_points, dataset)


if __name__ == "__main__":
    main()