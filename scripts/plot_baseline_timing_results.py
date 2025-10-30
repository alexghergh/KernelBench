import os
import json
import matplotlib.pyplot as plt


REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
TIMING_RESULTS_PATH = os.path.join(REPO_TOP_PATH, "results", "timing")

# directories containing the JSON files
directories = [
    d
    for d in os.listdir(TIMING_RESULTS_PATH)
    if os.path.isdir(os.path.join(TIMING_RESULTS_PATH, d))
]

# list of JSON files to process
json_files = [
    "baseline_time_torch.json",
    "baseline_time_torch_compile_inductor_default.json",
    "baseline_time_torch_compile_inductor_reduce-overhead.json",
    "baseline_time_torch_compile_inductor_max-autotune.json",
    "baseline_time_torch_compile_inductor_max-autotune-no-cudagraphs.json",
    "baseline_time_torch_compile_cudagraphs.json",
]

def extract_mean_times(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    # collect the mean times from all levels in the structure
    mean_times = {"level1": [], "level2": [], "level3": []}

    # loop through each level to extract mean times
    for level, level_data in data.items():
        for key, value in level_data.items():
            # check if value is a dictionary and contains the "mean" key
            if isinstance(value, dict) and "mean" in value:
                mean_times[level].append(value["mean"])
            else:
                print(f"Warning: Skipping {key} in {level} due to missing or malformed 'mean' key.")

    return mean_times

# create a plot for each JSON file
for json_file in json_files:
    level1_data = []
    level2_data = []
    level3_data = []
    directories_with_data = []

    # loop through each directory to extract the data
    for directory in directories:
        file_path = os.path.join(TIMING_RESULTS_PATH, directory, json_file)
        if os.path.exists(file_path):
            mean_times = extract_mean_times(file_path)
            level1_data.append(mean_times["level1"])
            level2_data.append(mean_times["level2"])
            level3_data.append(mean_times["level3"])
            directories_with_data.append(directory)

    # Plotting the results for each level
    plt.figure(figsize=(12, 6))

    # Level 1 plot
    plt.subplot(131)
    for i, data in enumerate(level1_data):
        plt.plot(range(1, len(data) + 1), data, label=f'{directories_with_data[i]}')
    plt.title(f'{json_file} - Level 1')
    plt.xlabel('Problem index')
    plt.ylabel('Mean Time (ms)')
    plt.legend()

    # Level 2 plot
    plt.subplot(132)
    for i, data in enumerate(level2_data):
        plt.plot(range(1, len(data) + 1), data, label=f'{directories_with_data[i]}')
    plt.title(f'{json_file} - Level 2')
    plt.xlabel('Problem index')
    plt.ylabel('Mean Time (ms)')
    plt.legend()

    # Level 3 plot
    plt.subplot(133)
    for i, data in enumerate(level3_data):
        plt.plot(range(1, len(data) + 1), data, label=f'{directories_with_data[i]}')
    plt.title(f'{json_file} - Level 3')
    plt.xlabel('Problem index')
    plt.ylabel('Mean Time (ms)')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
