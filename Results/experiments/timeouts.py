#!/usr/bin/env python3
"""
timeouts.py - Create dual y-axis plot showing accuracy and rounds to converge vs number of clients

Reads R_final_data.log files from folders with format {dataset}_wait_{N} and creates a plot
with number of clients (N) on x-axis, accuracy on left y-axis, and rounds to converge on right y-axis.
The dataset is automatically detected from the directory structure. If the --time flag is used,
the right y-axis will show time in minutes instead of rounds to converge.

Usage:
    python timeouts.py -b /path/to/base/directory
    python timeouts.py --base-dir /path/to/custom/directory
    python timeouts.py -b /path/to/base/directory --time
    python timeouts.py --base-dir /path/to/custom/directory --time
    python timeouts.py -b /path/to/base/directory -t
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import subprocess
from typing import Dict, List, Tuple, Optional

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def extract_n_from_folder(folder_name: str, dataset: str) -> Optional[int]:
    """Extract N from folder name like 'cifar_wait_5' or 'mnist_wait_10'"""
    pattern = rf'{dataset}_wait_(\d+)'
    match = re.search(pattern, folder_name)
    if match:
        return int(match.group(1))
    return None


def extract_dataset_from_path(base_dir: str) -> Optional[str]:
    """Extract dataset name from the base directory path by looking for subdirectories with dataset patterns"""
    if not os.path.exists(base_dir):
        return None
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if '_wait_' in item:
                # Extract dataset from pattern like 'mnist_wait_5' or 'cifar_wait_10'
                parts = item.split('_wait_')
                if len(parts) == 2:
                    dataset = parts[0]
                    if dataset in ['mnist', 'cifar']:
                        return dataset
    
    return None


def parse_log_with_logToDic(log_file_path: str) -> Optional[Dict]:
    """
    Use logToDic.py to parse R_final_data.log file and return the JSON data
    
    Args:
        log_file_path: Path to R_final_data.log file
        
    Returns:
        Dictionary with parsed data or None if parsing fails
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logToDic_path = os.path.join(script_dir, "logToDic.py")
        
        # Check if logToDic.py exists
        if not os.path.exists(logToDic_path):
            print(f"Error: logToDic.py not found at {logToDic_path}")
            return None
        
        # Run logToDic.py with --json flag
        result = subprocess.run([
            sys.executable, logToDic_path, log_file_path, "--json"
        ], capture_output=True, text=True, check=True)
        
        # Parse JSON output
        data = json.loads(result.stdout)
        return data
        
    except subprocess.CalledProcessError as e:
        print(f"Error running logToDic.py: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        return None
    except Exception as e:
        print(f"Error processing file {log_file_path}: {e}")
        return None


def collect_data(base_dir: str, use_time: bool = False) -> Tuple[Optional[str], List[Tuple[int, float, float]]]:
    """Collect accuracy and rounds to converge (or time) data from all folders and determine dataset automatically"""
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
        n_clients = extract_n_from_folder(folder_name, dataset)
        if n_clients is None:
            continue
        
        # Look for R_final_data.log
        log_file = os.path.join(folder_path, "R_final_data.log")
        if not os.path.exists(log_file):
            print(f"Warning: R_final_data.log not found in {folder_path}")
            continue
        
        print(f"Processing {folder_name} (N={n_clients} clients)")
        
        # Parse log file using logToDic.py
        log_data = parse_log_with_logToDic(log_file)
        
        if not log_data:
            print(f"Warning: Could not parse data from {log_file}")
            continue
        
        # Extract accuracy values and calculate average
        if "Accuracy" not in log_data:
            print(f"Warning: 'Accuracy' field not found in {log_file}")
            continue
        
        accuracies = log_data["Accuracy"]
        if not accuracies:
            print(f"Warning: No accuracy values found in {log_file}")
            continue
        
        avg_accuracy = np.mean(accuracies)
        
        if use_time:
            # Extract time values and calculate average (convert to minutes)
            if "Time" not in log_data:
                print(f"Warning: 'Time' field not found in {log_file}")
                continue
            
            times = log_data["Time"]
            if not times:
                print(f"Warning: No time values found in {log_file}")
                continue
            
            avg_time_minutes = np.mean(times) / 60.0  # Convert seconds to minutes
            
            print(f"  Accuracy: {avg_accuracy:.2f}% (from {len(accuracies)} executions)")
            print(f"  Time: {avg_time_minutes:.1f} minutes (from {len(times)} executions)")
            
            data_points.append((n_clients, avg_accuracy, avg_time_minutes))
        else:
            # Extract rounds to converge values and calculate average
            if "rounds_to_converge" not in log_data:
                print(f"Warning: 'rounds_to_converge' field not found in {log_file}")
                continue
            
            rounds_to_converge = log_data["rounds_to_converge"]
            if not rounds_to_converge:
                print(f"Warning: No rounds_to_converge values found in {log_file}")
                continue
            
            avg_rounds = np.mean(rounds_to_converge)
            
            print(f"  Accuracy: {avg_accuracy:.2f}% (from {len(accuracies)} executions)")
            print(f"  Rounds to converge: {avg_rounds:.1f} (from {len(rounds_to_converge)} executions)")
            
            data_points.append((n_clients, avg_accuracy, avg_rounds))
    
    # Sort by number of clients
    data_points.sort(key=lambda x: x[0])
    
    return dataset, data_points


def create_timeout_graph(data_points: List[Tuple[int, float, float]], dataset: str, use_time: bool = False) -> None:
    """Create dual y-axis plot showing accuracy and rounds to converge (or time) vs number of clients"""
    if not data_points:
        print("No data points to plot")
        return
    
    # Extract data
    n_clients = [point[0] for point in data_points]
    accuracies = [point[1] for point in data_points]
    right_axis_data = [point[2] for point in data_points]
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot accuracy on left y-axis
    color = 'tab:blue'
    xlabel = 'Server timeout multiplier' if use_time else 'Server timeout multiplier'
    ax1.set_xlabel(xlabel, fontsize=22)
    ax1.set_ylabel('Accuracy (%)', color=color, fontsize=22)
    line1 = ax1.plot(n_clients, accuracies, 'o-', color=color, linewidth=2, markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Create secondary y-axis for rounds to converge or time
    ax2 = ax1.twinx()
    color = 'tab:red'
    if use_time:
        ax2.set_ylabel('Time (minutes)', color=color, fontsize=22)
        line2 = ax2.plot(n_clients, right_axis_data, 's-', color=color, linewidth=2, markersize=8, label='Time')
    else:
        ax2.set_ylabel('Rounds to Converge', color=color, fontsize=22)
        line2 = ax2.plot(n_clients, right_axis_data, 's-', color=color, linewidth=2, markersize=8, label='Rounds to Converge')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
    
    # Set title
    title_suffix = "Time" if use_time else "Timeout"
    plt.title(f'RByz {title_suffix} Analysis - {dataset.upper()}', fontsize=22, pad=20)

    # Show every x-axis value
    ax1.set_xticks(n_clients)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3)
    
    # Create combined legend with black text
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=22)
    # Set legend text color to black
    for text in legend.get_texts():
        text.set_color('black')
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    filename_suffix = "time" if use_time else "timeout"
    output_path = os.path.join(output_dir, f"{filename_suffix}s_{dataset}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create dual y-axis plot showing accuracy and rounds to converge vs number of clients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python timeouts.py -b /home/bustaman/rbyz/Results/keep/timeouts_real/cifar
    python timeouts.py -b /home/bustaman/rbyz/Results/keep/timeouts_real/cifar -t
    python timeouts.py --base-dir /path/to/custom/directory --time
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       required=True,
                       help='Base directory containing {dataset}_wait_{N} folders')
    
    parser.add_argument('-t', '--time', 
                       action='store_true',
                       help='Use time (in minutes) instead of rounds to converge for right y-axis')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    metric_type = "time" if args.time else "rounds to converge"
    print(f"Collecting timeout analysis data from log files (using {metric_type})...")
    
    # Collect data and automatically determine dataset
    dataset, data_points = collect_data(args.base_dir, args.time)
    
    if not dataset:
        print("Could not determine dataset. Exiting.")
        sys.exit(1)
    
    if not data_points:
        print("No data found. Exiting.")
        sys.exit(1)
    
    print(f"\nFound data for {len(data_points)} configurations:")
    if args.time:
        for n_clients, accuracy, time_mins in data_points:
            print(f"  N={n_clients}: Accuracy={accuracy:.2f}%, Time={time_mins:.1f} min")
    else:
        for n_clients, accuracy, rounds in data_points:
            print(f"  N={n_clients}: Accuracy={accuracy:.2f}%, Rounds={rounds:.1f}")
    
    analysis_type = "time" if args.time else "timeout"
    print(f"\nCreating {analysis_type} analysis graph for {dataset.upper()}...")
    create_timeout_graph(data_points, dataset, args.time)


if __name__ == "__main__":
    main()
