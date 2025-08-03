#!/usr/bin/env python3
"""
attacks_line_graph.py - Create line graph comparing FLtrust and RByz algorithms showing percentage of byzantine clients vs error rate

Creates a line graph with percentage of byzantine clients on the X-axis and error rate (1-accuracy) on the Y-axis.
Compares both FLtrust and RByz algorithms on the same plot for easy comparison.

The script automatically extracts client information from the trust scores log files:
- Algorithm name (line 1)
- Number of rounds (line 2) 
- Total number of clients (line 3)
- Number of byzantine clients (line 4)

Byzantine percentage is calculated as: (byzantine_clients / total_clients) * 100
Error rate is calculated as: 1 - accuracy

USAGE PATTERNS:

1. Generate line graph for specific dataset and setting:
   python attacks_line_graph.py -b /path/to/dataset/directory -s 6
   
   Example: python attacks_line_graph.py -b /home/bustaman/rbyz/Results/keep/byz_attacks_large/mnist -s 6

2. Use default directory structure:
   python attacks_line_graph.py -s 1
   
   This will search in the default directory for the specified setting.

EXAMPLES:
    # Generate line graph for MNIST setting 6 (Targeted Label flipping (6))
    python attacks_line_graph.py -b /home/bustaman/rbyz/Results/keep/byz_attacks_large/mnist -s 6
    
    # Generate line graph for CIFAR setting 1 (Random label flipping)
    python attacks_line_graph.py -b /home/bustaman/rbyz/Results/keep/byz_attacks_large/cifar -s 1
    
    # Use default directory with setting 2
    python attacks_line_graph.py -s 2

DIRECTORY STRUCTURE EXPECTED:
    Base Directory/
    ├── R_{dataset}_set_{setting}/
    │   ├── 12_byz/
    │   │   ├── R_trust_scores.log
    │   │   └── R_final_data.log
    │   ├── 14_byz/
    │   │   ├── R_trust_scores.log
    │   │   └── R_final_data.log
    │   └── ...
    └── F_{dataset}_set_{setting}/
        ├── 12_byz/
        │   ├── F_trust_scores.log
        │   └── F_final_data.log
        └── ...
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


def parse_trust_scores_header(trust_scores_file: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """Parse the header of trust scores log to extract algorithm, rounds, total clients, and byzantine clients"""
    try:
        with open(trust_scores_file, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 4:
            print(f"Warning: Trust scores file {trust_scores_file} has less than 4 lines")
            return None, None, None, None
            
        algorithm = lines[0].strip()
        rounds = int(lines[1].strip())
        total_clients = int(lines[2].strip())
        byzantine_clients = int(lines[3].strip())
        
        return algorithm, rounds, total_clients, byzantine_clients
        
    except Exception as e:
        print(f"Error parsing trust scores file {trust_scores_file}: {e}")
        return None, None, None, None


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


def collect_data(base_dir: str, setting: int) -> Dict:
    """Collect accuracy data and byzantine percentages from algorithm folders"""
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return {}
    
    # Structure: {algorithm: {byz_percentage: [error_rates]}}
    data = {}
    
    # Look for algorithm folders matching the pattern
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Extract algorithm, dataset, and setting
        algorithm, dataset, folder_setting = extract_algorithm_dataset_setting(folder_name)
        if not all([algorithm, dataset, folder_setting is not None]):
            continue
        
        # Skip if setting doesn't match
        if folder_setting != setting:
            continue
        
        print(f"Processing {folder_name} (Algorithm: {get_algorithm_name(algorithm)}, Dataset: {dataset.upper()}, Setting: {setting})")
        
        # Initialize algorithm data
        if algorithm not in data:
            data[algorithm] = {}
        
        # Look for byzantine client subdirectories
        for byz_folder in os.listdir(folder_path):
            byz_folder_path = os.path.join(folder_path, byz_folder)
            
            if not os.path.isdir(byz_folder_path):
                continue
            
            # Extract byzantine count
            byz_count = extract_byzantine_count(byz_folder)
            if byz_count is None:
                continue
            
            # Look for trust scores log to get client information
            trust_scores_file = os.path.join(byz_folder_path, f"{algorithm}_trust_scores.log")
            if not os.path.exists(trust_scores_file):
                print(f"Warning: Trust scores file not found at {trust_scores_file}")
                continue
            
            # Parse trust scores header
            parsed_algorithm, rounds, total_clients, byzantine_clients = parse_trust_scores_header(trust_scores_file)
            
            if not all([parsed_algorithm, rounds, total_clients, byzantine_clients is not None]):
                print(f"Warning: Could not parse trust scores header from {trust_scores_file}")
                continue
            
            # Verify byzantine count matches
            if byzantine_clients != byz_count:
                print(f"Warning: Byzantine count mismatch. Folder: {byz_count}, File: {byzantine_clients}")
            
            # Calculate byzantine percentage
            byz_percentage = (byzantine_clients / total_clients) * 100
            
            # Look for the accuracy log file
            log_file = os.path.join(byz_folder_path, f"{algorithm}_final_data.log")
            if not os.path.exists(log_file):
                print(f"Warning: Log file not found at {log_file}")
                continue
            
            print(f"  Processing {byzantine_clients}/{total_clients} clients ({byz_percentage:.1f}% byzantine)")
            
            # Extract data using logToDic.py
            log_data = get_log_data_via_script(log_file)
            
            if not log_data or 'Accuracy' not in log_data:
                print(f"Warning: No accuracy data found in {log_file}")
                continue
            
            accuracies = log_data['Accuracy']
            # Convert percentage to decimal and then to error rates
            error_rates = [1.0 - (acc / 100.0) for acc in accuracies]  # Convert to error rates
            mean_error_rate = np.mean(error_rates)
            print(f"    Found {len(accuracies)} accuracy values, mean error rate: {mean_error_rate:.3f}")
            
            if byz_percentage not in data[algorithm]:
                data[algorithm][byz_percentage] = []
            
            # Store the mean error rate for this run, not individual error rates
            data[algorithm][byz_percentage].append(mean_error_rate)
    
    return data


def create_line_graph(data: Dict, setting: int, dataset: str) -> None:
    """Create line graph comparing algorithms"""
    if not data:
        print("No data points to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    colors = {'R': 'blue', 'F': 'red'}
    markers = {'R': 'o', 'F': 's'}
    algorithm_names = {'R': 'RByz', 'F': 'FLtrust'}
    
    for algorithm in sorted(data.keys()):
        if not data[algorithm]:
            continue
            
        # Prepare data points
        percentages = []
        mean_errors = []
        std_errors = []
        
        for byz_percentage in sorted(data[algorithm].keys()):
            error_rates = data[algorithm][byz_percentage]
            percentages.append(byz_percentage)
            mean_errors.append(np.mean(error_rates))
            std_errors.append(np.std(error_rates))
            
            print(f"{algorithm_names[algorithm]} {byz_percentage:.1f}%: mean error={np.mean(error_rates):.3f}, std={np.std(error_rates):.3f}")
        
        if not percentages:
            continue
        
        # Plot line with error bars
        plt.errorbar(percentages, mean_errors, yerr=std_errors,
                    color=colors.get(algorithm, 'black'),
                    marker=markers.get(algorithm, 'o'),
                    linewidth=2,
                    markersize=8,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    capsize=5,
                    capthick=2,
                    label=algorithm_names.get(algorithm, algorithm))
    
    # Customize the plot
    plt.xlabel('Byzantine clients (%)', fontsize=16)
    plt.ylabel('Error rate', fontsize=16)
    
    attack_name = get_attack_name(setting)
    plt.title(f'{attack_name} - {dataset.upper()}', fontsize=18)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add legend
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    output_path = os.path.join(output_dir, f"attacks_line_graph_{dataset}_set_{setting}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create line graph comparing FLtrust and RByz algorithms showing percentage of byzantine clients vs error rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python attacks_line_graph.py -b /home/bustaman/rbyz/Results/keep/byz_attacks_large/mnist -s 6
    python attacks_line_graph.py -b /home/bustaman/rbyz/Results/keep/byz_attacks_large/cifar -s 1
    python attacks_line_graph.py -s 2
    
Attack Settings:
    0: No attack
    1: Random label flipping
    2: Random image corruption
    3, 4, 5, 6: Targeted Label flipping (setting)
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/byz_attacks_large",
                       help='Base directory containing dataset folder (default: /home/bustaman/rbyz/Results/keep/byz_attacks_large)')
    
    parser.add_argument('-s', '--setting', 
                       type=int,
                       required=True,
                       choices=[0, 1, 2, 3, 4, 5, 6],
                       help='Attack setting (0-6)')
    
    args = parser.parse_args()
    
    # Try to extract dataset from base directory path
    dataset = None
    if 'mnist' in args.base_dir.lower():
        dataset = 'mnist'
    elif 'cifar' in args.base_dir.lower():
        dataset = 'cifar'
    else:
        # Try to find dataset subdirectory
        if os.path.exists(args.base_dir):
            for item in os.listdir(args.base_dir):
                if item in ['mnist', 'cifar']:
                    dataset = item
                    args.base_dir = os.path.join(args.base_dir, item)
                    break
    
    if not dataset:
        dataset = "unknown"
    
    print(f"Using base directory: {args.base_dir}")
    print(f"Attack setting: {args.setting} ({get_attack_name(args.setting)})")
    print(f"Dataset: {dataset.upper()}")
    
    print(f"Collecting data from log files...")
    
    # Collect data
    data = collect_data(args.base_dir, args.setting)
    
    if not data:
        print("No data found. Exiting.")
        sys.exit(1)
    
    # Create line graph
    print(f"\nCreating line graph for setting {args.setting}...")
    create_line_graph(data, args.setting, dataset)
    
    print("Line graph generation complete!")


if __name__ == "__main__":
    main()
