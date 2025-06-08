import matplotlib.pyplot as plt
import os
import glob
import re

def parse_log_file(filepath):
    """Parse a log file and return lists of rounds and accuracies."""
    rounds = []
    accuracies = []
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        round_num = int(parts[0])
                        accuracy = float(parts[1])
                        rounds.append(round_num)
                        accuracies.append(accuracy)
                    except ValueError:
                        continue
    
    return rounds, accuracies

def find_log_files(directory):
    """Find all F_acc_*.log and R_acc_*.log files in the directory."""
    f_files = glob.glob(os.path.join(directory, "F_acc_*.log"))
    r_files = glob.glob(os.path.join(directory, "R_acc_*.log"))
    return f_files, r_files

def extract_run_id(filepath):
    """Extract run ID from filename for labeling."""
    filename = os.path.basename(filepath)
    match = re.search(r'[FR]_acc_(\d+)\.log', filename)
    return match.group(1) if match else "unknown"

def plot_comparison():
    """Create comparison plot of FLTrust vs RByz accuracy."""
    # Define the directory containing the log files
    log_directory = "accLogs"
    
    if not os.path.exists(log_directory):
        print(f"Directory {log_directory} not found!")
        return
    
    f_files, r_files = find_log_files(log_directory)
    
    if not f_files and not r_files:
        print("No log files found!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot FLTrust files
    for f_file in f_files:
        rounds, accuracies = parse_log_file(f_file)
        if rounds and accuracies:
            run_id = extract_run_id(f_file)
            plt.plot(rounds, accuracies, 'b-', linewidth=2, alpha=0.7, 
                    label=f'FLTrust (Run {run_id})')
    
    # Plot RByz files
    for r_file in r_files:
        rounds, accuracies = parse_log_file(r_file)
        if rounds and accuracies:
            run_id = extract_run_id(r_file)
            plt.plot(rounds, accuracies, 'r-', linewidth=2, alpha=0.7, 
                    label=f'RByz (Run {run_id})')
    
    # Customize the plot
    plt.xlabel('Round Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('FLTrust vs RByz: Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to show percentages if values are between 0 and 1
    if plt.gca().get_ylim()[1] <= 1.0:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_comparison()