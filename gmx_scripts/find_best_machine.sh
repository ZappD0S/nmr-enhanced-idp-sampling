#!/bin/bash

set -eu

# Function to run the temperature averaging function on a remote host via SSH
get_avg_temp() {
    hostname=$1
    # Run the temperature averaging function on the remote host via SSH
    ssh -o "StrictHostKeyChecking no" "$hostname" 'nvidia-smi | grep -oP "\d+(?=C)" | awk "{sum+=\$1; count+=1} END {if(count > 0) print sum/count; else print \"N/A\"}"'
}

# Generate the list of hostnames from matlab1.nmrbox.org to matlab9.nmrbox.org
hostnames=(matlab{1..9}.nmrbox.org)

# Initialize variables for the smallest temperature and corresponding hostname
smallest_temp=99999
best_hostname=""

# Loop through each hostname and get the average temperature
for hostname in "${hostnames[@]}"; do
    avg_temp=$(get_avg_temp "$hostname")

    # Check if the result is valid and not "N/A"
    if [[ "$avg_temp" != "N/A" && $(echo "$avg_temp < $smallest_temp" | bc -l) -eq 1 ]]; then
        smallest_temp=$avg_temp
        best_hostname=$hostname
    fi
done

# Output the hostname with the smallest temperature
if [ -n "$best_hostname" ]; then
    echo "Hostname with the smallest average temperature: $best_hostname"
    echo "Average temperature: $smallest_temp"
else
    echo "No valid temperature readings found."
fi
