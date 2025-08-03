#!/usr/bin/env python3
"""
attacks_acc.py - Create bar chart comparing number of byzantine clients vs accuracy for RByz and FLtrust algorithms

Iterates over directories in the specified base directory with format {Algorithm_type}_{dataset}_set_{setting}
that contain subdirectories for each run with different numbers of byzantine clients.
Creates bar charts showing mean accuracy with standard deviation error bars.
The 0 byzantine clients run appears as a red horizontal line (Model not poisoned).

USAGE PATTERNS:

1. Generate ALL plots from a parent directory containing multiple algorithm/dataset/setting combinations:
   python attacks_acc.py -b /path/to/parent/directory
   
   This will find all subdirectories matching {Algorithm_type}_{dataset}_set_{setting} pattern
   and generate a separate plot for each combination.

2. Generate a SINGLE plot for a specific algorithm/dataset/setting:
   python attacks_acc.py -b /path/to/{Algorithm_type}_{dataset}_set_{setting}
   
   Example: python attacks_acc.py -b /home/bustaman/rbyz/Results/keep/byz_attacks/cifar/R_cifar_set_3

3. Filter results when generating from parent directory:
   python attacks_acc.py -b /path/to/parent -d mnist -a R -s 1
   
   This generates plots only for combinations matching the filters.

EXAMPLES:
    # Generate all plots from default directory
    python attacks_acc.py
    
    # Generate all plots from custom parent directory
    python attacks_acc.py -b /home/bustaman/rbyz/Results/keep/byz_attacks
    
    # Generate single plot for specific setting
    python attacks_acc.py -b /home/bustaman/rbyz/Results/keep/byz_attacks/cifar/R_cifar_set_3
    
    # Generate only RByz plots for MNIST
    python attacks_acc.py -b /home/bustaman/rbyz/Results/keep/byz_attacks -d mnist -a R
    
    # Generate only plots for attack setting 1 (Random label flipping)
    python attacks_acc.py -b /home/bustaman/rbyz/Results/keep/byz_attacks -s 1

DIRECTORY STRUCTURE EXPECTED:
    Parent Directory/
    ├── R_mnist_set_0/
    │   ├── 0_byz/
    │   │   └── R_final_data.log
    │   ├── 2_byz/
    │   │   └── R_final_data.log
    │   └── 4_byz/
    │       └── R_final_data.log
    ├── R_mnist_set_1/
    │   ├── 0_byz/
    │   └── 2_byz/
    └── F_cifar_set_2/
        ├── 0_byz/
        └── 1_byz/
"""

import os
import re
import sys
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add the current directory to path so we can import logToDic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def extract_algorithm_dataset_setting(folder_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Extract algorithm type, dataset, and setting from folder name like 'R_mnist_set_1' or 'F_cifar_set_0'"""
    pattern = r'([RF])_([a-z]+)_set_(\d+)'
    match = re.search(pattern, folder_name)
    if match:
        algorithm = match.group(1)
        dataset = match.group(2)
        setting = int(match.group(3))
        return algorithm, dataset, setting
    return None, None, None


def extract_byzantine_count(folder_name: str) -> Optional[int]:
    """Extract byzantine count from folder name like '2_byz' or '10_byz'"""
    pattern = r'(\d+)_byz'
    match = re.search(pattern, folder_name)
    if match:
        return int(match.group(1))
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


def get_attack_name(setting: int) -> str:
    """Convert setting number to attack name"""
    attack_names = {
        0: "No attack",
        1: "Random label flipping",
        2: "Random image corruption",
        3: "Targeted Label flipping (3)",
        4: "Targeted Label flipping (4)",
        5: "Targeted Label flipping (5)",
        6: "Targeted Label flipping (6)"
    }
    return attack_names.get(setting, f"Unknown attack ({setting})")


def get_algorithm_name(algorithm: str) -> str:
    """Convert algorithm code to full name"""
    algorithm_names = {
        'R': 'RByz',
        'F': 'FLtrust'
    }
    return algorithm_names.get(algorithm, algorithm)


def collect_data(base_dir: str, target_dataset: Optional[str] = None, 
                target_algorithm: Optional[str] = None, target_setting: Optional[int] = None) -> Dict:
    """Collect accuracy data from all matching folders"""
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return {}
    
    # Structure: {algorithm: {dataset: {setting: {byz_count: [accuracies]}}}}
    data = {}
    
    # Check if base_dir is directly a {Algorithm_type}_{dataset}_set_{setting} directory
    base_dir_name = os.path.basename(base_dir)
    algorithm, dataset, setting = extract_algorithm_dataset_setting(base_dir_name)
    
    if algorithm and dataset and setting is not None:
        # Direct path to a specific setting directory
        print(f"Processing {base_dir_name} (Algorithm: {get_algorithm_name(algorithm)}, Dataset: {dataset.upper()}, Setting: {setting})")
        
        # Apply filters if specified
        if target_dataset and dataset != target_dataset:
            print(f"Dataset filter {target_dataset} doesn't match {dataset}")
            return {}
        if target_algorithm and algorithm != target_algorithm:
            print(f"Algorithm filter {target_algorithm} doesn't match {algorithm}")
            return {}
        if target_setting is not None and setting != target_setting:
            print(f"Setting filter {target_setting} doesn't match {setting}")
            return {}
        
        # Initialize nested dictionaries
        data[algorithm] = {}
        data[algorithm][dataset] = {}
        data[algorithm][dataset][setting] = {}
        
        # Look for byzantine client subdirectories in the base directory
        for byz_folder in os.listdir(base_dir):
            byz_folder_path = os.path.join(base_dir, byz_folder)
            
            if not os.path.isdir(byz_folder_path):
                continue
            
            # Extract byzantine count
            byz_count = extract_byzantine_count(byz_folder)
            if byz_count is None:
                continue
            
            # Look for the log file
            log_file = os.path.join(byz_folder_path, f"{algorithm}_final_data.log")
            if not os.path.exists(log_file):
                print(f"Warning: Log file not found at {log_file}")
                continue
            
            print(f"  Processing {byz_count} byzantine clients")
            
            # Extract data using logToDic.py
            log_data = get_log_data_via_script(log_file)
            
            if not log_data or 'Accuracy' not in log_data:
                print(f"Warning: No accuracy data found in {log_file}")
                continue
            
            accuracies = log_data['Accuracy']
            print(f"    Found {len(accuracies)} accuracy values")
            
            data[algorithm][dataset][setting][byz_count] = accuracies
    
    else:
        # Original behavior: iterate through subdirectories
        # First check if we have dataset subdirectories (mnist, cifar)
        dataset_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item in ['mnist', 'cifar']:
                dataset_dirs.append(item_path)
        
        # If we found dataset directories, search within them
        if dataset_dirs:
            search_dirs = dataset_dirs
        else:
            # No dataset subdirectories, search directly in base_dir
            search_dirs = [base_dir]
        
        for search_dir in search_dirs:
            for folder_name in os.listdir(search_dir):
                folder_path = os.path.join(search_dir, folder_name)
                
                # Skip if not a directory
                if not os.path.isdir(folder_path):
                    continue
                
                # Extract algorithm, dataset, and setting
                algorithm, dataset, setting = extract_algorithm_dataset_setting(folder_name)
                if not all([algorithm, dataset, setting is not None]):
                    continue
                
                # Apply filters if specified
                if target_dataset and dataset != target_dataset:
                    continue
                if target_algorithm and algorithm != target_algorithm:
                    continue
                if target_setting is not None and setting != target_setting:
                    continue
                
                print(f"Processing {folder_name} (Algorithm: {get_algorithm_name(algorithm)}, Dataset: {dataset.upper()}, Setting: {setting})")
                
                # Initialize nested dictionaries
                if algorithm not in data:
                    data[algorithm] = {}
                if dataset not in data[algorithm]:
                    data[algorithm][dataset] = {}
                if setting not in data[algorithm][dataset]:
                    data[algorithm][dataset][setting] = {}
                
                # Look for byzantine client subdirectories
                for byz_folder in os.listdir(folder_path):
                    byz_folder_path = os.path.join(folder_path, byz_folder)
                    
                    if not os.path.isdir(byz_folder_path):
                        continue
                    
                    # Extract byzantine count
                    byz_count = extract_byzantine_count(byz_folder)
                    if byz_count is None:
                        continue
                    
                    # Look for the log file
                    log_file = os.path.join(byz_folder_path, f"{algorithm}_final_data.log")
                    if not os.path.exists(log_file):
                        print(f"Warning: Log file not found at {log_file}")
                        continue
                    
                    print(f"  Processing {byz_count} byzantine clients")
                    
                    # Extract data using logToDic.py
                    log_data = get_log_data_via_script(log_file)
                    
                    if not log_data or 'Accuracy' not in log_data:
                        print(f"Warning: No accuracy data found in {log_file}")
                        continue
                    
                    accuracies = log_data['Accuracy']
                    print(f"    Found {len(accuracies)} accuracy values")
                    
                    data[algorithm][dataset][setting][byz_count] = accuracies
    
    return data


def create_chart(data: Dict, algorithm: str, dataset: str, setting: int) -> None:
    """Create bar chart for a specific algorithm, dataset, and setting"""
    if (algorithm not in data or dataset not in data[algorithm] or 
        setting not in data[algorithm][dataset]):
        print(f"No data found for {get_algorithm_name(algorithm)} {dataset.upper()} setting {setting}")
        return
    
    setting_data = data[algorithm][dataset][setting]
    
    if not setting_data:
        print("No data points to plot")
        return
    
    # Separate 0 byzantine clients for horizontal line
    baseline_accuracy = None
    if 0 in setting_data:
        baseline_accuracies = setting_data[0]
        baseline_accuracy = np.mean(baseline_accuracies)
        print(f"Baseline (0 byzantine): {baseline_accuracy:.2f}")
    
    # Prepare data for bar chart (exclude 0 byzantine)
    byz_counts = []
    means = []
    stds = []
    
    for byz_count in sorted(setting_data.keys()):
        if byz_count == 0:  # Skip 0 byzantine for bars
            continue
            
        accuracies = setting_data[byz_count]
        byz_counts.append(byz_count)
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
        
        print(f"{byz_count} byzantine: mean={np.mean(accuracies):.2f}, std={np.std(accuracies):.2f}")
    
    if not byz_counts:
        print("No byzantine client data to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    
    # Create bar chart
    bars = plt.bar(byz_counts, means, 
                   color='black', 
                   alpha=0.8,
                   width=max(byz_counts) * 0.08,  # Adjust width based on range
                   edgecolor='black',
                   linewidth=1)
    
    # Add error bars
    plt.errorbar(byz_counts, means, yerr=stds, 
                fmt='none', 
                color='green', 
                capsize=5, 
                capthick=2,
                linewidth=2)
    
    # Add horizontal line for baseline (0 byzantine)
    if baseline_accuracy is not None:
        plt.axhline(y=baseline_accuracy, 
                   color='red', 
                   linestyle='--', 
                   linewidth=2,
                   alpha=0.8,
                   label=f'$M_{{NP}}$ (No attack)')
    
    # Customize the plot
    plt.xlabel('Byzantine clients', fontsize=18)
    plt.ylabel('Model Accuracy', fontsize=18)
    
    attack_name = get_attack_name(setting)
    algorithm_name = get_algorithm_name(algorithm)
    plt.title(f'{algorithm_name} - {attack_name} - {dataset.upper()}', fontsize=18)
    
    # Set y-axis limits for better visualization
    all_values = means + ([baseline_accuracy] if baseline_accuracy else [])
    min_acc = min(all_values) - max(stds + [1]) - 1
    max_acc = max(all_values) + max(stds + [1]) + 1
    plt.ylim(min_acc, max_acc)
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    plt.xticks(byz_counts)
    
    # Add legend if baseline exists
    if baseline_accuracy is not None:
        plt.legend(fontsize=14)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    output_path = os.path.join(output_dir, f"attacks_acc_{algorithm}_{dataset}_set_{setting}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create bar chart comparing number of byzantine clients vs accuracy for different algorithms and attack settings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python attacks_acc.py
    python attacks_acc.py -b /path/to/custom/directory
    python attacks_acc.py -d mnist -a R -s 1
    python attacks_acc.py -d cifar -a F -s 2
    
Attack Settings:
    0: No attack
    1: Random label flipping
    2: Random image corruption
    3, 4, 5, 6: Targeted Label flipping (setting)
    
Algorithm Types:
    R: RByz
    F: FLtrust
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/byz_attacks",
                       help='Base directory containing algorithm folders (default: /home/bustaman/rbyz/Results/keep/byz_attacks)')
    
    parser.add_argument('-d', '--dataset', 
                       choices=['mnist', 'cifar'],
                       help='Filter by dataset (mnist or cifar)')
    
    parser.add_argument('-a', '--algorithm', 
                       choices=['R', 'F'],
                       help='Filter by algorithm (R for RByz, F for FLtrust)')
    
    parser.add_argument('-s', '--setting', 
                       type=int,
                       choices=[0, 1, 2, 3, 4, 5, 6],
                       help='Filter by attack setting (0-6)')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    if args.dataset:
        print(f"Filtering by dataset: {args.dataset.upper()}")
    if args.algorithm:
        print(f"Filtering by algorithm: {get_algorithm_name(args.algorithm)}")
    if args.setting is not None:
        print(f"Filtering by setting: {args.setting} ({get_attack_name(args.setting)})")
    
    print(f"Collecting data from log files...")
    
    # Collect data
    data = collect_data(args.base_dir, args.dataset, args.algorithm, args.setting)
    
    if not data:
        print("No data found. Exiting.")
        sys.exit(1)
    
    # Generate charts for all found combinations
    chart_count = 0
    for algorithm in data:
        for dataset in data[algorithm]:
            for setting in data[algorithm][dataset]:
                print(f"\nCreating chart for {get_algorithm_name(algorithm)} {dataset.upper()} setting {setting}...")
                create_chart(data, algorithm, dataset, setting)
                chart_count += 1
    
    print(f"\nGenerated {chart_count} charts")


if __name__ == "__main__":
    main()
