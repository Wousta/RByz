#!/usr/bin/env python3
"""
logToDic.py - Parse log files and convert to dictionary format

Takes a log file path as input and parses it into a dictionary where:
- Keys are the parameter names from the log
- Values are lists of values for each execution run

Usage:
    Basic usage: 
        python logToDic.py R_final_data.log
    
    Save to file:
        python logToDic.py R_final_data.log -o output.py
    
    JSON output:
        python logToDic.py R_final_data.log --json
    
    JSON to file:
        python logToDic.py R_final_data.log --json -o output.json

Example output:
    data = {
        "Accuracy": [90.120003, 91.239998, 91.839996],
        "Time": [473, 473, 474],
        "clnt_vd_proportion": [0.25, 0.25, 0.25],
        "src_targ_class": [[-1, -1], [-1, -1], [-1, -1]],
        "missclassed_samples": [0, 0, 0]
    }
"""

import sys
import argparse
from typing import Dict, List, Any, Union


def parse_value(value_str: str) -> Union[int, float, str]:
    """Parse a string value to appropriate type (int, float, or string)"""
    try:
        # Try integer first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        # If neither int nor float, return as string
        return value_str


def parse_log_file(file_path: str) -> Dict[str, List[Any]]:
    """
    Parse log file and return dictionary with parsed data
    
    Args:
        file_path: Path to the log file
        
    Returns:
        Dictionary with parameter names as keys and lists of values
    """
    data = {}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Skip empty lines and end markers
                if not line or line.startswith('//') or line == '$ END OF EXECUTION $':
                    continue
                
                # Split line into parts
                parts = line.split()
                if len(parts) < 2:
                    continue
                    
                key = parts[0]
                values = parts[1:]
                
                # Initialize key in dictionary if not exists
                if key not in data:
                    data[key] = []
                
                # Parse values based on how many there are
                if len(values) == 1:
                    # Single value
                    parsed_value = parse_value(values[0])
                    data[key].append(parsed_value)
                else:
                    # Multiple values - store as list
                    parsed_values = [parse_value(v) for v in values]
                    data[key].append(parsed_values)
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    return data


def print_dictionary(data: Dict[str, List[Any]]) -> None:
    """Print dictionary in a formatted way"""
    if not data:
        print("No data found or error parsing file")
        return
    
    print("data = {")
    for i, (key, values) in enumerate(data.items()):
        print(f'    "{key}": {values}', end='')
        if i < len(data) - 1:
            print(',')
        else:
            print()
    print("}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Parse log file to dictionary format')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('--output', '-o', help='Output file (optional, prints to stdout if not specified)')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Parse the log file
    data = parse_log_file(args.log_file)
    
    if not data:
        sys.exit(1)
    
    # Output results
    if args.json:
        import json
        json_output = json.dumps(data, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"JSON output written to {args.output}")
        else:
            print(json_output)
    else:
        if args.output:
            with open(args.output, 'w') as f:
                original_stdout = sys.stdout
                sys.stdout = f
                print_dictionary(data)
                sys.stdout = original_stdout
            print(f"Dictionary output written to {args.output}")
        else:
            print_dictionary(data)


if __name__ == "__main__":
    main()