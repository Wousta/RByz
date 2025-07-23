#!/usr/bin/env python3
"""
ts_evo.py - Create line graph showing Trust Score evolution for honest vs byzantine clients

Reads a single .log file containing trust score information during multiple experiment executions
and generates a line graph showing the average trust scores over rounds.
- Blue line: Average trust score of honest clients
- Red line: Average trust score of byzantine clients

The script supports moving average smoothing to reduce spikiness in the graphs using the -w option.

Usage:
    python ts_evo.py /path/to/trust_scores.log
    python ts_evo.py /path/to/trust_scores.log -o custom_output.pdf
    python ts_evo.py /path/to/trust_scores.log -w 10  # Apply 10-round moving average
    python ts_evo.py /path/to/trust_scores.log -w 5 -o smooth_output.pdf
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

def extract_dataset_from_path(file_path: str) -> str:
    """Extract dataset name from the file path by looking for 'cifar' or 'mnist' directories"""
    path_parts = Path(file_path).parts
    for part in path_parts:
        if 'cifar' in part.lower():
            return 'CIFAR-10'
        elif 'mnist' in part.lower():
            return 'MNIST'
    return 'Unknown'


def parse_trust_scores_log(log_file_path: str) -> Tuple[str, List[List[Tuple[List[float], List[float]]]]]:
    """
    Parse trust scores log file and return algorithm name and execution data
    
    Args:
        log_file_path: Path to the trust scores log file
        
    Returns:
        Tuple of (algorithm_name, executions)
        where executions is a list of executions, each execution is a list of rounds,
        and each round is a tuple of (byzantine_scores, honest_scores)
    """
    executions = []
    current_execution = []
    algorithm_name = None
    
    try:
        with open(log_file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for end of execution marker
            if line == '$ END OF EXECUTION $':
                if current_execution:
                    executions.append(current_execution)
                    current_execution = []
                i += 1
                continue
            
            # Check if this is the start of a new execution (algorithm name)
            if line and not line.startswith('-') and not line.replace('.', '').replace('-', '').isdigit():
                # This should be the algorithm name
                if algorithm_name is None:
                    algorithm_name = line
                
                # Read execution metadata
                if i + 3 < len(lines):
                    try:
                        num_rounds = int(lines[i + 1])
                        total_clients = int(lines[i + 2])
                        num_byzantine = int(lines[i + 3])
                        i += 4  # Move past metadata
                        
                        print(f"Processing execution: {num_rounds} rounds, {total_clients} clients, {num_byzantine} byzantine")
                        
                        # Parse rounds for this execution
                        round_data = []
                        current_round_scores = []
                        
                        while i < len(lines) and lines[i] != '$ END OF EXECUTION $':
                            line = lines[i].strip()
                            
                            if line == '- Round end -':
                                if current_round_scores:
                                    # Split scores into byzantine and honest
                                    byzantine_scores = current_round_scores[:num_byzantine]
                                    honest_scores = current_round_scores[num_byzantine:]
                                    round_data.append((byzantine_scores, honest_scores))
                                    current_round_scores = []
                            elif line and not line.startswith('$'):
                                try:
                                    score = float(line)
                                    current_round_scores.append(score)
                                except ValueError:
                                    pass
                            
                            i += 1
                        
                        # Add the last round if it exists
                        if current_round_scores:
                            byzantine_scores = current_round_scores[:num_byzantine]
                            honest_scores = current_round_scores[num_byzantine:]
                            round_data.append((byzantine_scores, honest_scores))
                        
                        current_execution = round_data
                        print(f"  Found {len(round_data)} rounds of data")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing execution metadata: {e}")
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # Add the last execution if it doesn't end with the marker
        if current_execution:
            executions.append(current_execution)
            
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found")
        return None, []
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, []
    
    print(f"Parsed {len(executions)} executions for algorithm {algorithm_name}")
    return algorithm_name, executions


def calculate_average_trust_scores(executions: List[List[Tuple[List[float], List[float]]]]) -> Tuple[List[float], List[float]]:
    """
    Calculate average trust scores across all executions
    
    Args:
        executions: List of executions, each containing rounds with (byzantine_scores, honest_scores)
        
    Returns:
        Tuple of (avg_byzantine_scores_per_round, avg_honest_scores_per_round)
    """
    if not executions:
        return [], []
    
    # Find the maximum number of rounds across all executions
    max_rounds = max(len(execution) for execution in executions)
    
    avg_byzantine_scores = []
    avg_honest_scores = []
    
    for round_num in range(max_rounds):
        byzantine_scores_this_round = []
        honest_scores_this_round = []
        
        for execution in executions:
            if round_num < len(execution):
                byzantine_scores, honest_scores = execution[round_num]
                
                # Average across clients in this round for this execution
                if byzantine_scores:
                    byzantine_scores_this_round.append(np.mean(byzantine_scores))
                if honest_scores:
                    honest_scores_this_round.append(np.mean(honest_scores))
        
        # Average across executions for this round
        if byzantine_scores_this_round:
            avg_byzantine_scores.append(np.mean(byzantine_scores_this_round))
        else:
            avg_byzantine_scores.append(0.0)
            
        if honest_scores_this_round:
            avg_honest_scores.append(np.mean(honest_scores_this_round))
        else:
            avg_honest_scores.append(0.0)
    
    return avg_byzantine_scores, avg_honest_scores


def apply_moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Apply moving average smoothing to data
    
    Args:
        data: List of values to smooth
        window_size: Size of the moving average window
        
    Returns:
        Smoothed data using moving average
    """
    if window_size <= 1 or len(data) <= window_size:
        return data
    
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            # For the first few points, use expanding window
            window_start = 0
            window_end = i + 1
        else:
            # Use rolling window
            window_start = i - window_size + 1
            window_end = i + 1
        
        window_values = data[window_start:window_end]
        smoothed_data.append(np.mean(window_values))
    
    return smoothed_data


def create_trust_score_evolution_graph(algorithm: str, dataset: str, 
                                     avg_byzantine_scores: List[float], 
                                     avg_honest_scores: List[float],
                                     window_size: int = 1,
                                     output_path: Optional[str] = None) -> None:
    """Create line graph showing trust score evolution"""
    if not avg_byzantine_scores and not avg_honest_scores:
        print("No data to plot")
        return
    
    # Apply moving average if window size > 1
    if window_size > 1:
        print(f"Applying moving average with window size {window_size}")
        if avg_byzantine_scores:
            avg_byzantine_scores = apply_moving_average(avg_byzantine_scores, window_size)
        if avg_honest_scores:
            avg_honest_scores = apply_moving_average(avg_honest_scores, window_size)
    
    rounds = list(range(1, max(len(avg_byzantine_scores), len(avg_honest_scores)) + 1))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot byzantine clients (red line)
    if avg_byzantine_scores:
        plt.plot(rounds[:len(avg_byzantine_scores)], avg_byzantine_scores, 
                color='red', 
                linewidth=2, 
                label='Byzantine clients',
                marker='o',
                markersize=4)
    
    # Plot honest clients (blue line)
    if avg_honest_scores:
        plt.plot(rounds[:len(avg_honest_scores)], avg_honest_scores, 
                color='blue', 
                linewidth=2, 
                label='Honest clients',
                marker='s',
                markersize=4)
    
    # Customize the plot
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Average Trust Score', fontsize=14)
    plt.title(f'TS evolution of {algorithm} for {dataset}', fontsize=16)
    
    # Set reasonable axis limits
    plt.xlim(0.5, max(rounds) + 0.5)
    
    # Set Y-axis from 0 to 1 (trust scores are typically in this range)
    plt.ylim(0, 1)
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        # Create output directory if it doesn't exist
        output_dir = "/home/bustaman/rbyz/Results/experiments/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Include window size in filename if smoothing is applied
        suffix = f"_w{window_size}" if window_size > 1 else ""
        output_path = os.path.join(output_dir, f"TS_evo_{algorithm}_{dataset.replace('-', '')}{suffix}.pdf")
    
    # Save the plot as PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create line graph showing Trust Score evolution for honest vs byzantine clients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python TS_evo.py /path/to/trust_scores.log
    python TS_evo.py /path/to/trust_scores.log -o custom_output.pdf
    python TS_evo.py /path/to/trust_scores.log -w 10  # Moving average with window 10
    python TS_evo.py /home/bustaman/rbyz/Results/keep/trust_logs/mnist/R_trust.log -w 5
        """
    )
    
    parser.add_argument('log_file', 
                       help='Path to the trust scores log file')
    
    parser.add_argument('-o', '--output', 
                       help='Output PDF file path (optional)')
    
    parser.add_argument('-w', '--window', 
                       type=int,
                       default=1,
                       help='Moving average window size for smoothing (default: 1, no smoothing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' does not exist")
        sys.exit(1)
    
    print(f"Processing trust scores from: {args.log_file}")
    if args.window > 1:
        print(f"Moving average smoothing enabled with window size: {args.window}")
    
    # Extract dataset from path
    dataset = extract_dataset_from_path(args.log_file)
    print(f"Detected dataset: {dataset}")
    
    # Parse the log file
    algorithm, executions = parse_trust_scores_log(args.log_file)
    
    if not algorithm or not executions:
        print("No valid data found in log file. Exiting.")
        sys.exit(1)
    
    print(f"Algorithm: {algorithm}")
    print(f"Found {len(executions)} executions")
    
    # Calculate average trust scores
    avg_byzantine_scores, avg_honest_scores = calculate_average_trust_scores(executions)
    
    print(f"Calculated averages for {len(avg_byzantine_scores)} rounds (byzantine) and {len(avg_honest_scores)} rounds (honest)")
    
    # Create the graph
    create_trust_score_evolution_graph(algorithm, dataset, avg_byzantine_scores, avg_honest_scores, args.window, args.output)


if __name__ == "__main__":
    main()
