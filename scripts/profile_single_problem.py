import subprocess

def run_ncu(ncu_command: str):
    print(f"Running ncu with command: {ncu_command}")
    output = subprocess.run(ncu_command, shell=True, capture_output=True, text=True)
    print(output.stdout)
    print(output.stderr)
    return output

def main():
    ncu_command = "sudo -E /usr/local/cuda-12.2/bin/ncu --target-processes all env PATH="$PATH" python3 scripts/run_single_arch.py"
    run_ncu(ncu_command)

if __name__ == "__main__":
    main()
