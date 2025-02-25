"""
Read in the baseline times from the file and categorize by time ranges
To get a sense of the kernel runtime distributions
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json


def analyze_timing_data(data, specified_level=None):
    """Analyze timing data and count operations in different time ranges"""
    # Define time ranges in seconds
    ranges = [
        (0, 0.00001, "< 10 μs (microsecond)"),
        (0.00001, 0.00002, "10-20 μs (microsecond)"),
        (0.00002, 0.00005, "20-50 μs (microsecond)"),
        (0.00005, 0.0001, "50-100 μs (microsecond)"),
        (0.0001, 0.001, "0.1-1 ms (millisecond)"),
        (0.001, 0.01, "1-10 ms (millisecond)"),
        (0.01, 0.1, "10-100 ms (millisecond)"),
        (0.1, 1.0, "0.1-1 s"),
        (1.0,  float('inf'), "> 1 s"),
    ]
    
    counts = {label: 0 for _, _, label in ranges}
    total_ops = 0
    
    # Process each level
    for level, operations in data.items():

        if specified_level:
            if level != specified_level:
                continue
        for op_name, metrics in operations.items():

            # Convert from milliseconds to seconds
            mean_time = metrics["mean"] / 1000.0

            # mean_time = metrics["mean"]
            total_ops += 1
            
            # Categorize the operation
            for lower, upper, label in ranges:
                if lower <= mean_time < upper:
                    counts[label] += 1
                    break
    
    return counts, total_ops


def main():
    
    path_to_baseline = "results/timing/L40S_matx3/baseline_time_torch.json"
    assert os.path.exists(path_to_baseline), f"Path to baseline does not exist: {path_to_baseline}"
    
    # Load the data directly with json module instead of pandas
    with open(path_to_baseline, 'r') as f:
        data = json.load(f)
    
    # Display basic information
    print(f"Dataset contains {len(data)} levels of operations")
    for level, operations in data.items():
        print(f"{level}: {len(operations)} operations")

        # Analyze for Particular Level
        print(f"\nTiming Distribution for {level}:")

        import pdb
        # Analyze timing data
        counts, total_ops = analyze_timing_data(data, specified_level=level)
        
        # Display results
        print(f"Total operations: {total_ops}")
        print("-" * 40)
        for label, count in counts.items():
            percentage = (count / total_ops) * 100
            print(f"{label}: {count} operations ({percentage:.2f}%)")
        
    #
    
    # # Create a bar chart
    # plt.figure(figsize=(12, 6))
    # labels = list(counts.keys())
    # values = list(counts.values())
    
    # plt.bar(labels, values)
    # plt.xlabel('Time Range')
    # plt.ylabel('Number of Operations')
    # plt.title('Distribution of Operation Execution Times')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    
    # # Save the plot
    # output_dir = os.path.dirname(path_to_baseline)
    # plt.savefig(f"{output_dir}/timing_distribution.png")
    # print(f"\nPlot saved to {output_dir}/timing_distribution.png")

if __name__ == "__main__":
    main()