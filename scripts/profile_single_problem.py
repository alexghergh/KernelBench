import subprocess

def run_ncu(ncu_command: str):
    print(f"Running ncu with command: {ncu_command}")
    output = subprocess.run(ncu_command, shell=True, capture_output=True, text=True)
    print(output.stdout)
    print(output.stderr)
    return output

def main():
    metrics = [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_active",
        # "dram__bytes_read.sum.peak_sustained_active",
        # "dram__bytes_read.sum.peak_sustained_elapsed",
        # "dram__bytes_write.sum.peak_sustained_active",
    ]
    metrics_str = ",".join(metrics)
    ncu_command = f"sudo -E env PATH=\"$PATH\" /usr/local/cuda-12.2/bin/ncu --target-processes all --launch-skip 5 --launch-count 1 --kernel-name \"matmul_kernel\" --metrics {metrics_str} python3 scripts/run_single_arch.py"
    run_ncu(ncu_command)

if __name__ == "__main__":
    main()
