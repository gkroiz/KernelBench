import torch
import numpy as np
from src.eval import (
    load_original_model_and_inputs,
    fetch_ref_arch_from_problem_id,
)

from src.tpu.eval import (
    time_execution,
    get_timing_stats,
    set_seed,
)

from src.dataset import construct_problem_dataset_from_problem_dir
from src.utils import read_file
import os
import json
from tqdm import tqdm
import jax
import torchax
from torchax.interop import jax_jit
import functools

"""
Generate baseline time for KernelBench
This profiles the wall clock time for each KernelBench reference problem

You can find a list of pre-generated baseline time in /results/timing/
But we recommend you run this script to generate the baseline time for your own hardware configurations

Using various configurations
- torch (torchax)

Torch Compile with various modes
https://pytorch.org/docs/main/generated/torch.compile.html
- torch.compile: backend="inductor", mode="default" (this is usually what happens when you do torch.compile(model))
- torch.compile: backend="inductor", mode="reduce-overhead" 
- torch.compile: backend="inductor", mode="max-autotune"
- torch.compile: backend="inductor", mode="max-autotune-no-cudagraphs"

In addition to default Torch Compile backend, you can always use other or your custom backends
https://pytorch.org/docs/stable/torch.compiler.html
- torch.compile: backend="cudagraphs" (CUDA graphs with AOT Autograd)
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")

def fetch_ref_arch_from_dataset(dataset: list[str], 
                                problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None
    
    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    
    ref_arch_src = read_file(ref_arch_path)

    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str, 
        num_trials: int = 100,
        use_jax_jit: bool = False,
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> dict:
    """
    Measure the time of a KernelBench reference architecture
    """
    torchax.enable_globally()
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.to(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.to(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]
            
            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)
            
            model = model.to(device=device)
                        
            if use_jax_jit:
                print(f"Using JAX JIT to compile model {ref_arch_name}")
                def model_func(state, *inputs):
                    return torch.func.functional_call(model, state, inputs)
                
                jitted = jax_jit(model_func)
                partial_jitted = functools.partial(jitted, model.state_dict())
            else:
                print(f"Using Torchax Eager Execution on {ref_arch_name}")
            
            
            elapsed_times = time_execution(
                partial_jitted, *inputs, num_trials=num_trials, verbose=verbose
            )
            runtime_stats = get_timing_stats(elapsed_times)

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")
            
            return runtime_stats
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")



def record_baseline_times(use_jax_jit: bool = False, 
                          file_name: str="baseline_time.json"):
    """
    Generate baseline time for KernelBench, 
    configure profiler options for PyTorch
    save to specified file
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif jax.default_backend() == "tpu":
        device = "jax"
    else:
        device = "cpu"

    # Check if file_name exists and load existing results
    save_path = os.path.join(TIMING_DIR, file_name)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            json_results = json.load(f)
        print(f"Loaded existing results from {save_path}")
    else:
        json_results = {}

    # TODO: Add level 3
    for level in [1, 2]:
        PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
        dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
        
        if f"level{level}" not in json_results:
            json_results[f"level{level}"] = {}

        num_problems = len(dataset)
        for problem_id in tqdm(range(1, num_problems + 1)):
            ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)
            if ref_arch_name in json_results[f"level{level}"] and json_results[f"level{level}"][ref_arch_name] is not None:
                print(f"Skipping problem {problem_id} in level {level}, already measured")
                continue

            runtime_stats = measure_program_time(
                ref_arch_name=ref_arch_name,
                ref_arch_src=ref_arch_src,
                use_jax_jit=use_jax_jit,
                device=device,
                verbose=False # do not print 
            )
            json_results[f"level{level}"][ref_arch_name] = runtime_stats

    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(json_results, f)
    return json_results

def test_measure_particular_program(level_num: int, problem_id: int):
    """
    Test measure_program_time on a particular program
    """
    device = torch.device("cuda:0")

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)

    exec_stats = measure_program_time(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        use_jax_jit=True,
        device=device,
        verbose=False
    )

    print(f"Execution time for {ref_arch_name}: {exec_stats}")


if __name__ == "__main__":
    # DEBUG and simple testing
    # test_measure_particular_program(2, 28)
    
    # Replace this with whatever hardware you are running on 
    hardware_name = "tpu-v4"

    input(f"You are about to start recording baseline time for {hardware_name}, press Enter to continue...")
    # Systematic recording of baseline time

    if os.path.exists(os.path.join(TIMING_DIR, hardware_name)):
        input(f"Directory {hardware_name} already exists, Are you sure you want to overwrite? Enter to continue...")

    # 1. Record Torchax Eager
    record_baseline_times(use_jax_jit=False, 
                          file_name=f"{hardware_name}/baseline_time_torchax.json")
    
    # 2. Record Torchax Jitted
    record_baseline_times(use_jax_jit=True, 
                          file_name=f"{hardware_name}/baseline_time_torchax_jitted.json")
    

    # Random debuging
    # get_torch_compile_triton(2, 12)
    # record_baseline_times()

    # run_profile(2, 43)
    # get_time(2, 43, torch_compile=False)
    # get_time(2, 43, torch_compile=True)
