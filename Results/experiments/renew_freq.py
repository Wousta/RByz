#!/usr/bin/env python3
"""
renew_freq.py - Create bar chart or line chart comparing test renewal frequency vs accuracy

Iterates over directories in the specified base directory (or default /home/bustaman/rbyz/Results/keep/renew_freq)
with format {dataset}_<frequency>, extracts data from R_acc.log files,
and creates a bar chart (default) or line chart showing mean accuracy with standard deviation error bars.
The dataset (mnist or cifar) is automatically detected from the directory structure.

Usage:
    python renew_freq.py
    python renew_freq.py -b /path/to/custom/directory
    python renew_freq.py -l  # Line chart
    python renew_freq.py --line  # Line chart
    python renew_freq.py -t  # Dual bar chart with trust scores
    python renew_freq.py --trust  # Dual bar chart with trust scores
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


def extract_frequency_from_folder(folder_name: str, dataset: str) -> int:
    """Extract frequency from folder name like 'mnist_5' or 'cifar_10'"""
    pattern = rf'{dataset}_(\d+)$'
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


def parse_trust_scores_log(log_file_path: str) -> List[float]:
    """
    Parse R_trust_scores.log file and return list of averaged trust scores for each execution
    
    Args:
        log_file_path: Path to R_trust_scores.log file
        
    Returns:
        List of average trust scores (one per execution)
    """
    executions_trust_scores = []
    current_execution_scores = []
    
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for end of execution marker
            if line == '$ END OF EXECUTION $':
                if current_execution_scores:
                    # Calculate average trust score for this execution
                    avg_trust = np.mean(current_execution_scores)
                    executions_trust_scores.append(avg_trust)
                    current_execution_scores = []
                i += 1
                continue
            
            # Skip algorithm name, rounds, clients, byzantine clients lines
            if line in ['RByz'] or line.isdigit():
                i += 1
                continue
            
            # Check for round delimiter
            if line == '- Round end -':
                # Read trust scores for this round
                i += 1
                while i < len(lines):
                    score_line = lines[i].strip()
                    if score_line == '- Round end -' or score_line == '$ END OF EXECUTION $':
                        break
                    try:
                        trust_score = float(score_line)
                        current_execution_scores.append(trust_score)
                    except ValueError:
                        pass
                    i += 1
                continue
            
            i += 1
        
        # Handle last execution if it doesn't end with marker
        if current_execution_scores:
            avg_trust = np.mean(current_execution_scores)
            executions_trust_scores.append(avg_trust)
            
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found")
        return []
    except Exception as e:
        print(f"Error reading trust scores file: {e}")
        return []
    
    return executions_trust_scores


def parse_acc_log_final_accuracies(log_file_path: str) -> List[float]:
    """
    Parse R_acc.log file and return list of final accuracies for each execution
    
    Args:
        log_file_path: Path to R_acc.log file
        
    Returns:
        List of final accuracy values (one per execution)
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
    
    # Extract final accuracy from each execution (last accuracy value)
    final_accuracies = []
    for execution in executions:
        if execution:
            # Get the last accuracy value from this execution
            final_accuracy = execution[-1][1]  # (round, accuracy) -> accuracy
            final_accuracies.append(final_accuracy)
    
    return final_accuracies
    """
    Parse R_acc.log file and return list of final accuracies for each execution
    
    Args:
        log_file_path: Path to R_acc.log file
        
    Returns:
        List of final accuracy values (one per execution)
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
    
    # Extract final accuracy from each execution (last accuracy value)
    final_accuracies = []
    for execution in executions:
        if execution:
            # Get the last accuracy value from this execution
            final_accuracy = execution[-1][1]  # (round, accuracy) -> accuracy
            final_accuracies.append(final_accuracy)
    
    return final_accuracies


def collect_data(base_dir: str = "/home/bustaman/rbyz/Results/keep/renew_freq", use_trust: bool = False) -> Tuple[str, List[Tuple[int, List[float], List[float]]]]:
    """Collect accuracy data and optionally trust scores from all folders and determine dataset automatically"""
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
        
        # Extract frequency from folder name
        frequency = extract_frequency_from_folder(folder_name, dataset)
        if frequency is None:
            print(f"Warning: Could not extract frequency from folder '{folder_name}' for dataset '{dataset}'")
            continue
        
        # Look for R_acc.log
        log_file = os.path.join(folder_path, "R_acc.log")
        if not os.path.exists(log_file):
            print(f"Warning: R_acc.log not found in {folder_path}")
            continue
        
        print(f"Processing {folder_name} (frequency: {frequency})")
        
        # Extract final accuracies from R_acc.log
        final_accuracies = parse_acc_log_final_accuracies(log_file)
        
        if not final_accuracies:
            print(f"Warning: No final accuracy data found in {log_file}")
            continue
        
        print(f"  Found {len(final_accuracies)} final accuracy values")
        
        if use_trust:
            # Look for R_trust_scores.log
            trust_log_file = os.path.join(folder_path, "R_trust_scores.log")
            if not os.path.exists(trust_log_file):
                print(f"Warning: R_trust_scores.log not found in {folder_path}")
                continue
            
            # Extract trust scores
            trust_scores = parse_trust_scores_log(trust_log_file)
            
            if not trust_scores:
                print(f"Warning: No trust score data found in {trust_log_file}")
                continue
            
            print(f"  Found {len(trust_scores)} trust score values")
            data_points.append((frequency, final_accuracies, trust_scores))
        else:
            data_points.append((frequency, final_accuracies, []))
    
    # Sort by frequency
    data_points.sort(key=lambda x: x[0])
    
    return dataset, data_points


def create_chart(data_points: List[Tuple[int, List[float], List[float]]], dataset: str, use_line_chart: bool = False, use_trust: bool = False) -> None:
    """Create bar chart or line chart with error bars, optionally including trust scores"""
    if not data_points:
        print("No data points to plot")
        return
    
    # Extract frequencies and calculate statistics
    frequencies = []
    means = []
    stds = []
    trust_means = []
    trust_stds = []
    
    if use_trust:
        for frequency, accuracies, trust_scores in data_points:
            frequencies.append(frequency)
            means.append(np.mean(accuracies))
            stds.append(np.std(accuracies))
            trust_means.append(np.mean(trust_scores))
            trust_stds.append(np.std(trust_scores))
            
            print(f"Frequency {frequency}: accuracy mean={np.mean(accuracies):.2f}, std={np.std(accuracies):.2f}")
            print(f"                    trust mean={np.mean(trust_scores):.3f}, std={np.std(trust_scores):.3f}")
    else:
        for frequency, accuracies, _ in data_points:
            frequencies.append(frequency)
            means.append(np.mean(accuracies))
            stds.append(np.std(accuracies))
            
            print(f"Frequency {frequency}: mean={np.mean(accuracies):.2f}, std={np.std(accuracies):.2f}")
    
    # Create the plot
    if use_trust:
        # Create dual bar chart for accuracy and trust scores
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Set the width of bars and positions
        bar_width = 0.35
        x_pos = np.arange(len(frequencies))
        
        # Create bars for accuracy (left y-axis)
        bars1 = ax1.bar(x_pos - bar_width/2, means, bar_width,
                        color='blue', alpha=0.7, label='Accuracy',
                        edgecolor='black', linewidth=1)
        
        # Add error bars for accuracy
        ax1.errorbar(x_pos - bar_width/2, means, yerr=stds,
                    fmt='none', color='darkblue', capsize=5,
                    capthick=2, linewidth=2)
        
        # Create secondary y-axis for trust scores
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x_pos + bar_width/2, trust_means, bar_width,
                        color='red', alpha=0.7, label='Trust Score',
                        edgecolor='black', linewidth=1)
        
        # Add error bars for trust scores
        ax2.errorbar(x_pos + bar_width/2, trust_means, yerr=trust_stds,
                    fmt='none', color='darkred', capsize=5,
                    capthick=2, linewidth=2)
        
        # Customize the plot
        ax1.set_xlabel('Test Renewal Frequency', fontsize=18)
        ax1.set_ylabel('Model Accuracy', color='blue', fontsize=18)
        ax2.set_ylabel('Average Trust Score', color='red', fontsize=18)
        ax1.set_title(f'RByz Accuracy and Trust vs Test Renewal Frequency - {dataset.upper()}', fontsize=18)
        
        # Set x-axis ticks
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(frequencies)
        
        # Color the y-axis labels
        ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=15)
        
        chart_type = "Dual Bar"
        output_suffix = "dual_bars"
        
    else:
        plt.figure(figsize=(7, 6))
        
        if use_line_chart:
            # Create line chart
            plt.plot(frequencies, means, 
                    marker='o', 
                    linewidth=2, 
                    markersize=6, 
                    color='blue',
                    markerfacecolor='blue',
                    markeredgecolor='darkblue')
            
            # Add error bars
            plt.errorbar(frequencies, means, yerr=stds, 
                        fmt='none', 
                        color='green', 
                        capsize=5, 
                        capthick=2,
                        linewidth=2)
            
            chart_type = "Line"
            output_suffix = "line"
        else:
            # Create bar chart
            bars = plt.bar(frequencies, means, 
                           color='black', 
                           alpha=0.8,
                           width=0.8,  # Adjust width as needed
                           edgecolor='black',
                           linewidth=1)
            
            # Add error bars
            plt.errorbar(frequencies, means, yerr=stds, 
                        fmt='none', 
                        color='green', 
                        capsize=5, 
                        capthick=2,
                        linewidth=2)
            
            chart_type = "Bar"
            output_suffix = "bars"
        
        # Customize the plot
        plt.xlabel('Test Renewal Frequency', fontsize=18)
        plt.ylabel('Model Accuracy', fontsize=18)
        plt.title(f'RByz Accuracy vs Test Renewal Frequency - {dataset.upper()}', fontsize=18)

        # Set y-axis limits for better visualization
        min_acc = min(means) - max(stds) - 1
        max_acc = max(means) + max(stds) + 1
        plt.ylim(min_acc, max_acc)
        
        # Make tick labels bigger
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        plt.xticks(frequencies)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot as PDF
    output_path_pdf = os.path.join(output_dir, f"renew_freq_{output_suffix}_{dataset}.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF plot saved as {output_path_pdf}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create bar chart or line chart comparing test renewal frequency vs accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python renew_freq.py
    python renew_freq.py -b /path/to/custom/directory
    python renew_freq.py -l  # Line chart
    python renew_freq.py --line  # Line chart
    python renew_freq.py -t  # Dual bar chart with trust scores
    python renew_freq.py --trust  # Dual bar chart with trust scores
        """
    )
    
    parser.add_argument('-b', '--base-dir', 
                       default="/home/bustaman/rbyz/Results/keep/renew_freq",
                       help='Base directory containing dataset folders (default: /home/bustaman/rbyz/Results/keep/renew_freq)')
    
    parser.add_argument('-l', '--line', 
                       action='store_true',
                       help='Create line chart instead of bar chart')
    
    parser.add_argument('-t', '--trust', 
                       action='store_true',
                       help='Include trust scores in dual bar chart (forces bar chart mode)')
    
    args = parser.parse_args()
    
    print(f"Using base directory: {args.base_dir}")
    
    # Trust mode forces bar chart
    if args.trust:
        chart_type = "dual bar (with trust scores)"
        use_line_chart = False
    else:
        chart_type = "line" if args.line else "bar"
        use_line_chart = args.line
    
    print(f"Collecting data from R_acc.log files...")
    if args.trust:
        print("Also collecting trust score data from R_trust_scores.log files...")
    
    # Collect data and automatically determine dataset
    dataset, data_points = collect_data(args.base_dir, args.trust)
    
    if not dataset:
        print("Could not determine dataset. Exiting.")
        sys.exit(1)
    
    if not data_points:
        print("No data found. Exiting.")
        sys.exit(1)
    
    print(f"\nFound data for {len(data_points)} configurations:")
    if args.trust:
        for frequency, accuracies, trust_scores in data_points:
            print(f"  Frequency {frequency}: {len(accuracies)} accuracy runs, {len(trust_scores)} trust score runs")
    else:
        for frequency, accuracies, _ in data_points:
            print(f"  Frequency {frequency}: {len(accuracies)} runs")
    
    print(f"\nCreating {chart_type} chart for {dataset.upper()}...")
    create_chart(data_points, dataset, use_line_chart, args.trust)


if __name__ == "__main__":
    main()
