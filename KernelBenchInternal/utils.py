import re
import os
import multiprocessing
import time
import concurrent
from functools import cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()  # Load variables from .env early so query_server can see them

from openai.types.shared.reasoning_effort import ReasoningEffort
from tqdm import tqdm
from transformers import AutoTokenizer

# API clients
import anthropic
from openai import OpenAI
from google import genai
from together import Together


# Define API key access
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")  # for Local Deployment
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")


########################################################
# Inference Helpers
########################################################

@cache
def load_deepseek_tokenizer():
    # TODO: Should we update this for new deepseek? Same tokenizer?
    # return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Instruct-0724")
    return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)

# Buffer because deepseek totally blocks us if we send stuff that's too long :(
TOO_LONG_FOR_DEEPSEEK = 115_000


def is_safe_to_send_to_deepseek(prompt):
    tokenizer = load_deepseek_tokenizer()
    # print(f"Prompt: {len(prompt)}")
    # print(f"Prompt length: {len(tokenizer(prompt, verbose=False)['input_ids'])}")

    if type(prompt) == str:
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK


def set_gpu_arch(arch_list: list[str]):
    """
    Set env variable for torch cuda arch list to build kernels for specified
    architectures.
    """
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


def _query_openai(client: OpenAI,
                  model: str,
                  system_prompt: str,
                  prompt: str,
                  num_completions: int,
                  temperature: float,
                  max_tokens: int,
                  top_p: float,
                  top_k: int,
                  use_reasoning_model: bool,
                  reasoning_effort: str,
                  budget_tokens: int,
                  ):
    if use_reasoning_model:
        assert reasoning_effort in ['low', 'medium', 'high'], (
            "Reasoning effort can only be 'low', 'medium' or 'high'"
        )
        print(f"Using OpenAI reasoning model {model} with reasoning effort {reasoning_effort}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                { "role": "user", "content": prompt },
            ],
            reasoning_effort=reasoning_effort,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": prompt },
            ],
            stream=False,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    outputs = [choice.message.content for choice in response.choices]
    token_usage = response.usage

    return outputs, token_usage


def _query_anthropic(client: anthropic.Anthropic,
                     model: str,
                     system_prompt: str,
                     prompt: str,
                     num_completions: int,
                     temperature: float,
                     max_tokens: int,
                     top_p: float,
                     top_k: int,
                     use_reasoning_model: bool,
                     reasoning_effort: str,
                     budget_tokens: int,
                     ):
    if use_reasoning_model:
        response = client.with_options(timeout=10000000).messages.create(
            model=model,
            system=system_prompt,
            messages=[
                { "role": "user", "content": prompt },
            ],
            max_tokens=max_tokens,
            # Claude thinking requires budget_tokens for thinking (reasoning)
            thinking={
                "type": "enabled",
                "budget_tokens": (budget_tokens if budget_tokens != 0 else max_tokens // 2),
            },
        )
    else:
        # Use standard endpoint for normal models
        response = client.with_options(timeout=10000000).messages.create(
            model=model,
            system=system_prompt,
            messages=[
                { "role": "user", "content": prompt },
            ],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
    outputs = [choice.text for choice in response.content if choice.type == 'text']
    token_usage = response.usage

    return outputs, token_usage


def _query_google(client: genai.Client,
                  model: str,
                  system_prompt: str,
                  prompt: str,
                  num_completions: int,
                  temperature: float,
                  max_tokens: int,
                  top_p: float,
                  top_k: int,
                  use_reasoning_model: bool,
                  reasoning_effort: str,
                  budget_tokens: int,
                  ):
    if use_reasoning_model:
        assert 'gemini-2.5-' in model, "Only Gemini 2.5 models can use reasoning"

    # this can use either reasoning or no-reasoning normally, depending on the
    # model
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
    )
    outputs = response.text
    token_usage = response.usage_metadata

    return outputs, token_usage


def _query_deepseek(client: OpenAI,
                    model: str,
                    system_prompt: str,
                    prompt: str,
                    num_completions: int,
                    temperature: float,
                    max_tokens: int,
                    top_p: float,
                    top_k: int,
                    use_reasoning_model: bool,
                    reasoning_effort: str,
                    budget_tokens: int,
                    ):
    if model in ["deepseek-chat", "deepseek-coder"]:
        # regular deepseek model
        response = client.chat.completions.create(
                model=model,
                messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": prompt },
            ],
            stream=False,
            temperature=temperature,
            n=1,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    else:
        # deepseek reasoner
        assert use_reasoning_model and model == "deepseek-reasoner", (
            "Only deepseek-reasoner is supported with reasoning enabled"
        )
        response = client.chat.completions.create(
                model=model,
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": prompt },
            ],
            stream=False,
            n=num_completions, # beam search
            max_tokens=max_tokens,
        )
    outputs = [choice.message.content for choice in response.choices]
    token_usage = response.usage

    return outputs, token_usage


def _query_together(client: OpenAI,
                    model: str,
                    system_prompt: str,
                    prompt: str,
                    num_completions: int,
                    temperature: float,
                    max_tokens: int,
                    top_p: float,
                    top_k: int,
                    use_reasoning_model: bool,
                    reasoning_effort: str,
                    budget_tokens: int,
                    ):
    response = client.chat.completions.create(
        model=model,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": prompt },
        ],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=False,
    )
    outputs = [choice.message.content for choice in response.choices]
    token_usage = response.usage

    return outputs, token_usage


def _query_sambanova(client: OpenAI,
                    model: str,
                    system_prompt: str,
                    prompt: str,
                    num_completions: int,
                    temperature: float,
                    max_tokens: int,
                    top_p: float,
                    top_k: int,
                    use_reasoning_model: bool,
                     reasoning_effort: str,
                    budget_tokens: int,
                    ):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": prompt },
        ],
        temperature=temperature,
        top_p=top_p,
    )
    outputs = [choice.message.content for choice in response.choices]
    token_usage = response.usage

    return outputs, token_usage


def _query_fireworks(client: OpenAI,
                     model: str,
                     system_prompt: str,
                     prompt: str,
                     num_completions: int,
                     temperature: float,
                     max_tokens: int,
                     top_p: float,
                     top_k: float,
                     use_reasoning_model: bool,
                     reasoning_effort: str,
                     budget_tokens: int,
                     ):
    response = client.chat.completions.create(
        model=model,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": prompt },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=False,
    )
    outputs = [choice.message.content for choice in response.choices]
    token_usage = response.usage

    return outputs, token_usage


def _query_generic(client: OpenAI,
                   model: str,
                   system_prompt: str,
                   prompt: str,
                   num_completions: int,
                   temperature: float,
                   max_tokens: int,
                   top_p: float,
                   top_k: float,
                   use_reasoning_model: bool,
                   reasoning_effort: str,
                   budget_tokens: int,
                   ):
    if type(prompt) is str:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            n=num_completions,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        outputs = [choice.text for choice in response.choices]
    else:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            n=num_completions,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        outputs = [choice.message.content for choice in response.choices]
    token_usage = response.usage

    return outputs, token_usage


def query_server(
    prompt: str | list[dict], # string if normal prompt, list of dicts to assemble if chat prompt
    system_prompt: str = "You are a helpful assistant", # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50,
    num_completions: int = 1, # beam search
    max_tokens: int = 128, # max output tokens to generate
    server_type: str = "sglang",
    server_address: str = "localhost",
    server_port: int = 30000, # only for local server hosted on SGLang
    model_name: str = "default", # specify model type

    # reasoning models
    use_reasoning_model: bool = True, # whether to use reasoning version
    budget_tokens: int = 0, # for claude thinking
    reasoning_effort: str = 'high', # for gpt-5
):
    """
    Query various sort of LLM inference API providers
    Supports:
    - OpenAI
    - Anthropic
    - Gemini / Google AI Studio
    - Deepseek
    - Together
    - Sambanova
    - Fireworks (OpenAI compatible)
    - SGLang (Local Server)
    """
    # create server based on arguments
    match server_type:
        case "openai":
            client = OpenAI(api_key=OPENAI_KEY)

        case "anthropic":
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        case "google":
            client = genai.Client(api_key=GEMINI_KEY)

        case "deepseek":
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            assert model_name in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"], (
                "Only support deepseek-chat, deepseek-coder, deepseek-reasoner for now"
            )
            if not is_safe_to_send_to_deepseek(prompt):
                raise RuntimeError("Prompt is too long for DeepSeek")

        case "together":
            client = Together(api_key=TOGETHER_KEY)

        case "sambanova":
            client = OpenAI(
                api_key=SAMBANOVA_API_KEY,
                base_url="https://api.sambanova.ai/v1"
            )

        case "fireworks":
            client = OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                timeout=10000000,
                max_retries=3,
            )

        case "sglang":
            url = f"http://{server_address}:{server_port}"
            client = OpenAI(
                api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
            )

        case _:
            raise NotImplementedError

    assert client is not None, "Client is not set, cannot proceed to generations"
    print(f"Querying {server_type}, model {model_name} with temp {temperature}, and max tokens {max_tokens}")

    # query the LLM server
    if server_type == "openai":
        query_server_func = _query_openai
    elif server_type == "anthropic":
        query_server_func = _query_anthropic
    elif server_type == "google":
        query_server_func = _query_google
    elif server_type == "deepseek":
        query_server_func = _query_deepseek
    elif server_type == "together":
        query_server_func = _query_together
    elif server_type == "sambanova":
        query_server_func = _query_sambanova
    elif server_type == "fireworks":
        query_server_func = _query_fireworks
    else:
        # for all other kinds of servers, use standard API
        query_server_func = _query_generic

    outputs, token_usage = query_server_func(client,
                                             model_name,
                                             system_prompt,
                                             prompt,
                                             num_completions,
                                             temperature,
                                             max_tokens,
                                             top_p,
                                             top_k,
                                             use_reasoning_model,
                                             reasoning_effort,
                                             budget_tokens)
    # note, token_usage structure depends on the individual inference provider
    return outputs, token_usage


# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6,
        "model_name": "deepseek",
        "max_tokens": 4096
    },
    "google": {
        "model_name": "gemini-1.5-flash-002",
        "temperature": 0.7, # need to experiment with temperature
        "max_tokens": 8192,
    },
    "together": { # mostly for Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "sglang": {  # this is for running locally, mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": 34561,
        "server_address": "localhost",
        # "server_port": 10210,
        # "server_address": "matx2.stanford.edu",
        "max_tokens": 16384,
    },
    "anthropic": {  # for Claude 3.5 Sonnet
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.8,
        "max_tokens": 4096,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",
        # "model_name": "o1-preview-2024-09-12", # be careful with this one
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "sambanova": {
        "model_name": "Meta-Llama-3.1-405B-Instruct",
        "temperature": 0.1,
        "max_tokens": 8192,
    },
}


def create_inference_server_from_presets(server_type: str = None,
                                         greedy_sample: bool = False,
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Querying server {server_type} with args: {server_args}")

        if time_generation:
            start_time = time.time()
            response = query_server(
                prompt, server_type=server_type, **server_args
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_server(
                prompt, server_type=server_type, **server_args
            )

    return _query_llm


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def print_messages(messages):
    for message in messages:
        print(message["role"])
        print(message["content"])
        print("-" * 50)
        print("\n\n")


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```\w+\n(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```\w+\n(.*?)```", trimmed, re.DOTALL)

    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code

    return None


def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    '''
    pattern = r'```\w+\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)

    return " \n ".join(combined_code) if combined_code else ""


################################################################################
# Scale up experiments in parallel
################################################################################

def maybe_multithread(func, instances, num_workers, time_interval=0.0, *shared_args, **shared_kwargs):
    """
    Multithreaded execution of func, with optional time interval between queries
    Ideal for querying LLM APIs, does not provide process isolation
    """
    output_data = []
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

                # Submit tasks one at a time with delay between them
                futures = []
                for instance in instances:
                    futures.append(
                        executor.submit(
                            func,
                            instance,
                            *shared_args,
                            **shared_kwargs
                        )
                    )
                    time.sleep(time_interval)  # sleep between submitting each task



                # Wait for each future to complete
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            output_data.append(result)
                    except Exception as e:
                        print("Got an error!", e)
                        continue
    else:
        for instance in tqdm(instances):
            output = func(instance, *shared_args, **shared_kwargs)
            if output is not None: output_data.append(output)

    return output_data


def maybe_multiprocess_cuda(
    func, instances, num_workers, *shared_args, **shared_kwargs
):
    """
    From monkeys, but modified to work with CUDA
    """
    output_data = []
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # this is necessary for CUDA to work

    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(func, instance, *shared_args, **shared_kwargs): None
                for instance in instances
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    if result is not None:
                        output_data.append(result)
                except Exception as e:
                    print("Got an error!", e)
                    continue
    return output_data


# src/random_inputs.py
import os, torch, itertools
from torch.distributions import Normal, Uniform, Laplace, Exponential, LogNormal

# Pick which distributions are allowed in “random” mode.
_DEFAULT_RANDOM_POOL = (
    ("normal",      lambda shape: Normal(0, 1).sample(shape)),
    ("uniform",     lambda shape: Uniform(-1, 1).sample(shape)),
    ("laplace",     lambda shape: Laplace(0, 1).sample(shape)),
    ("exponential", lambda shape: Exponential(1).sample(shape)),   # strictly >0
    ("lognormal",   lambda shape: LogNormal(0, 1).sample(shape)),  # strictly >0
)


def sample(shape, mode="random"):
    """
    shape : torch.Size or tuple
    mode  : "random"  – draw from a rotating pool of distributions
            "target"  – return a tensor from a randomly chosen edge-case pattern
            <dist>    – force a single distribution name, e.g. "laplace"
    """
    if mode == "random":
        # Round-robin through default pool
        idx = int(torch.empty((), dtype=torch.int64).random_()) % len(_DEFAULT_RANDOM_POOL)
        _, fn = _DEFAULT_RANDOM_POOL[idx]
        return fn(shape)

    # Explicit distribution name
    pool = dict(_DEFAULT_RANDOM_POOL)
    if mode not in pool:
        raise ValueError(f"Unknown distribution {mode}")
    return pool[mode](shape)


# ------------------------------------------------------------------
# Public helper: rand_mix / rand_mix_like
# ------------------------------------------------------------------

def rand_mix(*size, dist: str = "random", device=None, dtype=None, requires_grad: bool = False):
    """Return a tensor drawn from a chosen distribution (or randomly chosen).

    Parameters
    ----------
    *size : int or tuple
        Dimensions of the output tensor (same semantics as ``torch.randn``).
    dist : str, optional
        • "random"   – randomly cycle through the default pool defined above.
        • "target"   – pick from the specialised _TARGETED_CASES pool.
        • any key in the default pool ("normal", "uniform", "laplace", ...).
    device, dtype, requires_grad : any
        Forwarded to ``Tensor.to`` / ``Tensor.requires_grad_`` for convenience.
    """
    # normalise *size → shape tuple
    shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, torch.Size)) else size

    t = sample(shape, mode=dist)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def rand_mix_like(tensor: torch.Tensor, dist: str = "random", **kwargs):
    """rand_mix variant that infers shape from *tensor*."""
    return rand_mix(*tensor.shape, dist=dist, **kwargs)


# Register convenience aliases under torch namespace (does not shadow existing fns)
setattr(torch, "rand_mix", rand_mix)
setattr(torch, "rand_mix_like", rand_mix_like)
