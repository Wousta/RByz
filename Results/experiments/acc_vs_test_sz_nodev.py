#!/usr/bin/env python3
"""
acc_vs_test_sz_nodev.py - Create line graph comparing validation data percentage vs accuracy for both datasets

Iterates over folders in the specified base directory (or default /home/bustaman/rbyz/Results/keep/acc_vs_test_size_nodev)
with format {dataset}_<percentage>%vd, extracts data from R_final_data.log files,
and creates a line graph showing mean accuracy for both MNIST and CIFAR datasets.

Usage:
    python acc_vs_test_sz_nodev.py
    python acc_vs_test_sz_nodev.py --base-dir /path/to/custom/directory
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


def collect_data(dataset: str, base_dir: str = "/home/bustaman/rbyz/Results/keep/acc_vs_test_size_nodev") -> List[Tuple[float, List[float]]]:
    """Collect accuracy data from all folders for the specified dataset"""
    base_path = os.path.join(base_dir, dataset)
    
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist")
        return []
    
    data_points = []
    
    # Iterate through all subdirectories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
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
        
        print(f"Processing {dataset} {folder_name} (percentage: {percentage}%)")
        
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
    
    return data_points


def create_line_graph(mnist_data: List[Tuple[float, List[float]]], 
                     cifar_data: List[Tuple[float, List[float]]]) -> None:
    """Create line graph comparing both datasets"""
    plt.figure(figsize=(10, 6))
    
    # Process MNIST data
    if mnist_data:
        mnist_percentages = []
        mnist_means = []
        
        for percentage, accuracies in mnist_data:
            mnist_percentages.append(percentage)
            mnist_means.append(np.mean(accuracies))
            print(f"MNIST {percentage}%: mean={np.mean(accuracies):.2f}, std={np.std(accuracies):.2f}")
        
        # Plot MNIST line
        plt.plot(mnist_percentages, mnist_means, 
                marker='o', 
                linewidth=2, 
                markersize=6, 
                label='MNIST', 
                color='blue')
    
    # Process CIFAR data
    if cifar_data:
        cifar_percentages = []
        cifar_means = []
        
        for percentage, accuracies in cifar_data:
            cifar_percentages.append(percentage)
            cifar_means.append(np.mean(accuracies))
            print(f"CIFAR {percentage}%: mean={np.mean(accuracies):.2f}, std={np.std(accuracies):.2f}")
        
        # Plot CIFAR line
        plt.plot(cifar_percentages, cifar_means, 
                marker='s', 
                linewidth=2, 
                markersize=6, 
                label='CIFAR', 
                color='red')
    
    # Customize the plot
    plt.xlabel('Max Validation Data per client %', fontsize=12)
    plt.ylabel('Model Accuracy', fontsize=12)
    plt.title('RByz Accuracy vs Validation Data Percentage (No Dev)', fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Set reasonable axis limits
    all_percentages = []
    all_means = []
    
    if mnist_data:
        all_percentages.extend([p for p, _ in mnist_data])
        all_means.extend([np.mean(acc) for _, acc in mnist_data])
    
    if cifar_data:
        all_percentages.extend([p for p, _ in cifar_data])
        all_means.extend([np.mean(acc) for _, acc in cifar_data])
    
    if all_percentages and all_means:
        plt.xlim(min(all_percentages) - 1, max(all_percentages) + 1)
        plt.ylim(min(all_means) - 2, max(all_means) + 2)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, "acc_vs_test_sz_nodev_line.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create line graph comparing validation data percentage vs accuracy for both datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python acc_vs_test_sz_nodev.py
    python acc_vs_test_sz_nodev.py --base-dir /path/to/custom/directory
    python acc_vs_test_sz_nodev.py -b /path/to/custom/directory
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/acc_vs_test_size_nodev",
                       help='Base directory containing dataset folders (default: /home/bustaman/rbyz/Results/keep/acc_vs_test_size_nodev)')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    print("Collecting data from MNIST and CIFAR log files...")
    
    # Collect data for both datasets
    mnist_data = collect_data('mnist', args.base_dir)
    cifar_data = collect_data('cifar', args.base_dir)
    
    if not mnist_data and not cifar_data:
        print("No data found for either dataset. Exiting.")
        sys.exit(1)
    
    print(f"\nFound data:")
    if mnist_data:
        print(f"  MNIST: {len(mnist_data)} configurations")
        for percentage, accuracies in mnist_data:
            print(f"    {percentage}%: {len(accuracies)} runs")
    else:
        print("  MNIST: No data found")
    
    if cifar_data:
        print(f"  CIFAR: {len(cifar_data)} configurations")
        for percentage, accuracies in cifar_data:
            print(f"    {percentage}%: {len(accuracies)} runs")
    else:
        print("  CIFAR: No data found")
    
    print(f"\nCreating line graph...")
    create_line_graph(mnist_data, cifar_data)


if __name__ == "__main__":
    main()