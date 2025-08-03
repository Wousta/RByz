#!/usr/bin/env python3
"""
acc_vs_test_sz.py - Create bar chart or line chart comparing validation data percentage vs accuracy

Iterates over folders in the specified base directory (or default /home/bustaman/rbyz/Results/keep/acc_vs_test_size)
with format {dataset}_<percentage>%vd, extracts data from R_final_data.log files,
and creates a bar chart (default) or line chart showing mean accuracy with standard deviation error bars.
The dataset (mnist or cifar) is automatically detected from the directory structure.

Usage:
    python acc_vs_test_sz.py
    python acc_vs_test_sz.py -b /path/to/custom/directory
    python acc_vs_test_sz.py -l  # Line chart
    python acc_vs_test_sz.py --line  # Line chart
"""

import os
import re
import sys
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add the current directory to path so we can import logToDic
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


def get_log_data_via_script(log_file_path: str) -> Dict:
    """Call logToDic.py script to extract data from log file"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logToDic_path = os.path.join(script_dir, 'logToDic.py')
        
        # Run logToDic.py with JSON output
        result = subprocess.run(
            ['python', logToDic_path, log_file_path, '--json'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse JSON output
        data = json.loads(result.stdout)
        return data
        
    except subprocess.CalledProcessError as e:
        print(f"Error running logToDic.py: {e}")
        print(f"stderr: {e.stderr}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from logToDic.py: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}


def collect_data(base_dir: str = "/home/bustaman/rbyz/Results/keep/acc_vs_test_size") -> Tuple[str, List[Tuple[float, List[float]]]]:
    """Collect accuracy data from all folders and determine dataset automatically"""
    # First, determine the dataset from the directory structure
    dataset = extract_dataset_from_path(base_dir)
    if not dataset:
        print(f"Error: Could not determine dataset (mnist or cifar) from directory structure in {base_dir}")
        return None, []
    
    print(f"Detected dataset: {dataset.upper()}")
    
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
        
        # Look for R_final_data.log
        log_file = os.path.join(folder_path, "R_final_data.log")
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found in {folder_path}")
            continue
        
        print(f"Processing {folder_name} (percentage: {percentage}%)")
        
        # Extract data using logToDic.py
        log_data = get_log_data_via_script(log_file)
        
        if not log_data or 'Accuracy' not in log_data:
            print(f"Warning: No accuracy data found in {log_file}")
            continue
        
        accuracies = log_data['Accuracy']
        print(f"  Found {len(accuracies)} accuracy values")
        
        data_points.append((percentage, accuracies))
    
    # Sort by percentage
    data_points.sort(key=lambda x: x[0])
    
    return dataset, data_points


def create_chart(data_points: List[Tuple[float, List[float]]], dataset: str, use_line_chart: bool = False) -> None:
    """Create bar chart or line chart with error bars"""
    if not data_points:
        print("No data points to plot")
        return
    
    # Extract percentages and calculate statistics
    percentages = []
    means = []
    stds = []
    
    for percentage, accuracies in data_points:
        percentages.append(percentage)
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
        
        print(f"{percentage}%: mean={np.mean(accuracies):.2f}, std={np.std(accuracies):.2f}")
    
    # Create the plot
    plt.figure(figsize=(7, 6))
    
    if use_line_chart:
        # Create line chart
        plt.plot(percentages, means, 
                marker='o', 
                linewidth=2, 
                markersize=6, 
                color='blue',
                markerfacecolor='blue',
                markeredgecolor='darkblue')
        
        # Add error bars
        plt.errorbar(percentages, means, yerr=stds, 
                    fmt='none', 
                    color='green', 
                    capsize=5, 
                    capthick=2,
                    linewidth=2)
        
        chart_type = "Line"
        output_suffix = "line"
    else:
        # Create bar chart
        bars = plt.bar(percentages, means, 
                       color='black', 
                       alpha=0.8,
                       width=2.0,  # Adjust width as needed
                       edgecolor='black',
                       linewidth=1)
        
        # Add error bars
        plt.errorbar(percentages, means, yerr=stds, 
                    fmt='none', 
                    color='green', 
                    capsize=5, 
                    capthick=2,
                    linewidth=2)
        
        chart_type = "Bar"
        output_suffix = "bars"
    
    # Customize the plot
    plt.xlabel('Max Validation Data per client %', fontsize=18)
    plt.ylabel('Model Accuracy', fontsize=18)
    plt.title(f'RByz Accuracy vs Validation Data Percentage - {dataset.upper()}', fontsize=18)

    # Set y-axis limits for better visualization
    min_acc = min(means) - max(stds) - 1
    max_acc = max(means) + max(stds) + 1
    plt.ylim(min_acc, max_acc)
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    plt.xticks(percentages)
    
    # Add horizontal line for reference (if needed)
    # plt.axhline(y=some_reference_value, color='red', linestyle='--', alpha=0.7, label='Reference')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    output_path_pdf = os.path.join(output_dir, f"acc_vs_test_sz_{output_suffix}_{dataset}.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF plot saved as {output_path_pdf}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create bar chart or line chart comparing validation data percentage vs accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python acc_vs_test_sz.py
    python acc_vs_test_sz.py -b /path/to/custom/directory
    python acc_vs_test_sz.py -l  # Line chart
    python acc_vs_test_sz.py --line  # Line chart
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/acc_vs_test_size",
                       help='Base directory containing dataset folders (default: /home/bustaman/rbyz/Results/keep/acc_vs_test_size)')
    
    parser.add_argument('-l', '--line', 
                       action='store_true',
                       help='Create line chart instead of bar chart')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    chart_type = "line" if args.line else "bar"
    print(f"Collecting data from log files...")
    
    # Collect data and automatically determine dataset
    dataset, data_points = collect_data(args.base_dir)
    
    if not dataset:
        print("Could not determine dataset. Exiting.")
        sys.exit(1)
    
    if not data_points:
        print("No data found. Exiting.")
        sys.exit(1)
    
    print(f"\nFound data for {len(data_points)} configurations:")
    for percentage, accuracies in data_points:
        print(f"  {percentage}%: {len(accuracies)} runs")
    
    print(f"\nCreating {chart_type} chart for {dataset.upper()}...")
    create_chart(data_points, dataset, args.line)


if __name__ == "__main__":
    main()