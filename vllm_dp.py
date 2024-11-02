from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
import os
import pandas as pd
import csv
import datasets
import gc
import torch
from datasets import load_dataset
import json
import re
from argparse import Namespace
import os
from typing import Dict, List
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import socket
import math

pattern = re.compile(r'(Human|Assistant): (.*?)(?=\n(?:Human|Assistant):|$)', re.DOTALL)

def fix_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@ray.remote
class LLMRayActor:

    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            import vllm_wrap
            vllm.worker.worker.Worker = vllm_wrap.WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.4.1":
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = "wrap"
                    kwargs["worker_class_name"] = "WorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):

        splited_data = kwargs.pop('splited_data')
        outputs =  self.llm.generate(
            prompt_token_ids=[s[1] for s in splited_data], 
            **kwargs
        )
        gen = [ output.outputs[0].text for output in outputs]
        return [(s[0], g) for s, g in zip(splited_data , gen)]


    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    # def update_weight(self, name, dtype, shape, empty_cache=False):
    #     self.stop_remote_worker_execution_loop()

    #     if self.use_gpu_executor:
    #         return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
    #     else:
    #         return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r', encoding='utf8')]

def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode, encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def preprocess(dataset, tokenizer, apply_chat_template=False):

    ROLE = {
        'human': 'user',
        'assistant': 'assistant',
    }
    data = from_jsonl(dataset)
    data = [d['prompt'] for d in data]
    if not apply_chat_template:
        return tokenizer(data)['input_ids']

    # convert  \n\nHuman \n\nAssistant to sharegpt format
    if args.debug:
        matches = pattern.findall(data[3])
        parsed_data = [{"role": ROLE[role.lower()], "content": content.strip()} for role, content in matches]
        message = tokenizer.apply_chat_template([parsed_data], add_generation_prompt=True,tokenize=False)
        import pdb;pdb.set_trace()
    else:
        def format(text):
            matches = pattern.findall(text)
            parsed_data = [{"role": ROLE[role.lower()], "content": content.strip()} for role, content in matches]
            return parsed_data
        
        import multiprocess
        with multiprocess.Pool(10) as pool:
            message = list(pool.imap(format, data, chunksize=1))
        message = tokenizer.apply_chat_template(message, add_generation_prompt=True,tokenize=True)

    return data, message

def postprocess(out_path, prompt, gen):
    results = [{"prompt": p, "response": g} for p, g in zip(prompt, gen)]
    to_jsonl(results, out_path)


def run(args):

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except:
        if 'gemma' in args.model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        else:
            import pdb;pdb.set_trace()

    prompt, input_ids = preprocess(args.dataset, tokenizer, args.apply_chat_template)

    # load model
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        stop=["<|endoftext|>",'</s>', '<|eot_id|>'],
        max_tokens=2048,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        seed=args.seed,
        max_model_len=4096,
        enable_chunked_prefill=True,
        # max_num_seqs=4096,
        # max_num_batched_tokens=4096,
        # pipeline_parallel_size=8,
    )

    # generate
    outputs = llm.generate(
        prompt_token_ids = input_ids,
        sampling_params = sampling_params,
        use_tqdm=True
    )
    gen = [ output.outputs[0].text for output in outputs]
    postprocess(args.outpath, prompt, gen)
    # calc metrics

def create_vllm_engines(
    args
):
    vllm_engines = []
    for i in range(args.data_parallel_size):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(args.tensor_parallel_size == 1)
        scheduling_strategy = None

        if args.tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * args.tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=args.model,
                trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype="bfloat16",
                seed=args.seed + i,
                enable_chunked_prefill=True,
                max_model_len=4096,
                enable_prefix_caching=False,
            )
        )

    return vllm_engines

def split_dataset(dataset, rank, world_size):
    total_size = len(dataset)
    per_process_size = math.ceil(total_size / world_size)
    start_index = rank * per_process_size
    end_index = min(start_index + per_process_size, total_size)
    if isinstance(dataset, torch.utils.data.Dataset):
        subset = torch.utils.data.Subset(dataset, list(range(start_index, end_index)))
    elif isinstance(dataset, list):
        subset = dataset[start_index: end_index]
    else:
        raise Exception('Not Implemented')
    return subset

def gather_from_all_processes(data):
    """Gather data from all processes and concatenate."""
    gathered_data = [None] * torch.distributions.get_world_size()
    torch.distributions.all_gather_object(gathered_data, data)
    return [item for sublist in gathered_data for item in sublist]


def run_dp(args):

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except:
        if 'gemma' in args.model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        else:
            import pdb;pdb.set_trace()

    prompt, input_ids = preprocess(args.dataset, tokenizer, args.apply_chat_template)
    world_size = args.data_parallel_size * args.tensor_parallel_size
    vllm_engines = create_vllm_engines(args)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        stop=["<|endoftext|>",'</s>', '<|eot_id|>'],
        max_tokens=2048,
    )

    # to ensure the dp output order, we concat prompt and input_ids as tuple 
    splited_outs = []
    for rank in range(args.data_parallel_size):
        splited_data = split_dataset( [(p,i) for p,i in zip(prompt, input_ids)] , rank, world_size)
        splited_outs.append(vllm_engines[rank].generate.remote(
            splited_data=splited_data,
            sampling_params=sampling_params,
        ))

    all_data = ray.get(splited_outs)
    all_data = [a for aa in all_data for a in aa]
    postprocess(args.outpath, [a[0] for a in all_data], [a[1] for a in all_data])


## starter
def main(
    *,
    # for debug
    model: str ,
    dataset: str = 'test.jsonl', 
    outpath: str,
    debug: bool = False,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    apply_chat_template: bool = False,
    seed: int = 42, 
):

    import inspect
    frame = inspect.currentframe()
    keys, _, _, values = inspect.getargvalues(frame)
    values = {k: values[k] for k in keys}
    # utils.args = utils.SimpleNamespace(**values)

    fix_seed(seed)
    global args
    args = Namespace(**values)

    if data_parallel_size > 1:
        ray.init()
        run_dp(args)
    else:
        run(args)


if __name__ == '__main__':
    import defopt
    try:
        defopt.run(main)
    except:
        import sys, pdb, bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit or type == SystemExit:
            exit()
        print(type, value)
        pdb.post_mortem(tb)
