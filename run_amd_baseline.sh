python3 scripts/eval_from_generations.py run_name=baseline/claude-3.5-sonnet dataset_src=local level=1 num_gpu_devices=8 timeout=300
python3 scripts/eval_from_generations.py run_name=baseline/deepseek-r1 dataset_src=local level=1 num_gpu_devices=8 timeout=300
python3 scripts/eval_from_generations.py run_name=baseline/deepseek-v3 dataset_src=local level=1 num_gpu_devices=8 timeout=300
python3 scripts/eval_from_generations.py run_name=baseline/gpt-4o dataset_src=local level=1 num_gpu_devices=8 timeout=300
python3 scripts/eval_from_generations.py run_name=baseline/llama-3.1-405b dataset_src=local level=1 num_gpu_devices=8 timeout=300
python3 scripts/eval_from_generations.py run_name=baseline/llama-3.1-70b dataset_src=local level=1 num_gpu_devices=8 timeout=300
python3 scripts/eval_from_generations.py run_name=baseline/openai-o1 dataset_src=local level=1 num_gpu_devices=8 timeout=300