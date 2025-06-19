#!/bin/bash

log_dir="./logs/stepTimes"

# Check if directory exists
if [ ! -d "$log_dir" ]; then
    echo "Error: Directory $log_dir does not exist"
    exit 1
fi

# Array to store averages indexed by client ID
declare -A client_averages
declare -a client_indices

# Process each log file
for log_file in "$log_dir"/*.log; do
    if [ -f "$log_file" ]; then
        # Read the file line by line
        line_count=0
        client_id=""
        sum=0
        count=0
        
        while IFS= read -r line; do
            if [ $line_count -eq 0 ]; then
                # First line is the client index
                client_id="$line"
                client_indices+=("$client_id")
            else
                # Subsequent lines are times
                sum=$((sum + line))
                count=$((count + 1))
            fi
            line_count=$((line_count + 1))
        done < "$log_file"
        
        # Calculate average rounded up (ceiling)
        if [ $count -gt 0 ]; then
            # Use bash arithmetic for ceiling: (sum + count - 1) / count
            avg=$(( (sum + count - 1) / count ))
            #avg=$(( (avg * 3) / 2 ))
            #avg=$(( (avg * 2) ))
            client_averages["$client_id"]="$avg"
        fi
    fi
done

# Sort client indices numerically
IFS=$'\n' sorted_indices=($(sort -n <<<"${client_indices[*]}"))

# Print averages in order of client index
output=""
for i in "${sorted_indices[@]}"; do
    if [ -n "$output" ]; then
        output="$output, {${client_averages[$i]}}"
    else
        output="{${client_averages[$i]}}"
    fi
done

echo "$output"