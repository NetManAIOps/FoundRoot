# Copyright 2026 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoTokenizer
import multiprocessing
from tqdm import tqdm
import json
import os
from loguru import logger
import re
import time
import traceback
from typing import *


# Config
MODEL_PATH = "[PATH_TO_LLM]"
CTX_LENGTH = 30000
NUM_GPUS = 8
GPUS_PER_MODEL = 1
BATCH_SIZE = 16
ENGINE = 'vllm'


def worker_vllm(input_queue, output_queue, gpu_id, batch_size, sample_n, finished_flag, ready_cnt, model_path=MODEL_PATH):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from vllm import LLM, SamplingParams
        default_sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=CTX_LENGTH, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'], n=sample_n)
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=CTX_LENGTH, tensor_parallel_size=len(gpu_id.split(',')), gpu_memory_utilization=0.95, dtype='half')
        print(f"[worker {gpu_id}] Initialization finished.")
        ready_cnt.value = ready_cnt.value + 1
        
        while not finished_flag.get():
            if input_queue.empty():
                time.sleep(1)
                continue

            batch_prompts = []
            batch_args = []

            sampling_params = default_sampling_params

            for _ in range(batch_size):
                if not input_queue.empty():
                    try:
                        cur_items = input_queue.get_nowait()
                    except Exception as err:
                        break
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break
            
            if batch_prompts:
                if type(batch_args[0][-1]) == SamplingParams:
                    sampling_params = batch_args[0][-1]
                    print(f"[worker {gpu_id}] Using custom sampling params: {sampling_params}")
                else:
                    sampling_params = default_sampling_params
                sample_n = sampling_params.n
                answers = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                if sample_n > 1:
                    answers = [i.outputs for i in answers]
                    answers = [[j.text for j in i] for i in answers]
                else:
                    answers = [i.outputs[0].text for i in answers]

                print('==================================')
                print(batch_prompts[0])
                print('----------------------------------')
                print(answers[0])
                for answer, args in zip(answers, batch_args):
                    output_queue.put((answer, *args))
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        traceback.print_exc()
        time.sleep(5)


class LLMClient:
    def __init__(self, model_path=MODEL_PATH, engine=ENGINE, num_gpus=NUM_GPUS, gpu_range: Optional[List[int]]=None, gpus_per_model=GPUS_PER_MODEL, batch_size=BATCH_SIZE, sample_n: int=1, chat_template: Optional[str]=None, system_prompt: str="You are a helpful assistant."):
        # Create clients
        manager = multiprocessing.Manager()
        self.input_queue = manager.Queue()
        self.output_queue = manager.Queue()
        self.finished_flag = manager.Value('b', False)
        self.ready_cnt = manager.Value('i', 0)
        self.engine = engine
        self.sample_n = sample_n

        # Apply chat template
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.system_prompt = system_prompt

        if chat_template:
            self.tokenizer.chat_template = chat_template

        if gpu_range is None:
            gpu_range = list(range(num_gpus))
        else:
            print(f"[LLMClient] Using GPU range: {gpu_range}")

        self.processes = []
        for idx in range(0, len(gpu_range), gpus_per_model):
            gpu_id_str = ",".join(map(str, gpu_range[idx:idx+gpus_per_model]))
            print(f"[LLMClient] Starting worker {idx} on GPU {gpu_id_str}")
            if engine == 'vllm':
                p = multiprocessing.Process(target=worker_vllm, args=(self.input_queue, self.output_queue, gpu_id_str, batch_size, sample_n, self.finished_flag, self.ready_cnt, model_path))
            else:
                raise NotImplementedError(f"Unrecognized inference engine: {engine}")
            self.processes.append(p)
            p.start()
        
        print(f"[LLMClient] {len(self.processes)} workers started.")

    def wait_for_ready(self):
        while self.ready_cnt.value < len(self.processes):
            time.sleep(1)
        print(f"[LLMClient] All workers are ready!")

    def _apply_chat_template(self, prompt: str) -> str:
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.decode(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True))
        
    def llm_batch_generate(self, batch_prompts: List[str], dryrun_outputs: Optional[Union[List[str], List[List[str]]]] = None, use_chat_template=True, sampling_params=None):
        while not self.output_queue.empty():
            self.output_queue.get()
        self.finished_flag.set(False)

        total_cnt = 0

        if dryrun_outputs is not None:
            logger.warning(f"[llm_batch_generate] Dryrun mode. {len(batch_prompts)=}, {len(dryrun_outputs)=}")

        for i, item in enumerate(batch_prompts):
            if use_chat_template:
                inputs = self._apply_chat_template(item)
            else:
                inputs = item
            if dryrun_outputs is not None:
                self.input_queue.put((inputs, i, item, dryrun_outputs[i]))
            elif sampling_params is not None:
                self.input_queue.put((inputs, i, item, sampling_params))
            else:
                self.input_queue.put((inputs, i, item))
            total_cnt += 1

        answer_dict = {}

        with tqdm(total=total_cnt) as pbar:
            while len(answer_dict) < total_cnt:
                line = self.output_queue.get()
                pbar.update()

                # Append to answer
                answer_dict[line[1]] = line[0]
        
        answer_list = []
        for i in range(len(batch_prompts)):
            if i not in answer_dict:
                answer_list.append(None)
            else:
                answer_list.append(answer_dict[i])

        return answer_list

    def kill(self):
        self.finished_flag.set(True)
        print(f"[LLMClient] Killing workers...")
        time.sleep(5.0)
        for p in self.processes:
            p.join()
        print(f"[LLMClient] All workers have been killed!")


def try_fix_json(json_string: str, special_words=['question', 'answer', 'success', 'reference', ',', ':', '\n', '}', '{']):
    # Fix unmatched quots
    quotes_indices = [m.start() for m in re.finditer(r'"', json_string)]
    fixed_json = list(json_string)
    for i in quotes_indices:
        for special in special_words:
            if json_string[i + 1:].startswith(special) or json_string[:i].endswith(': '):
                break
        else:
            fixed_json[i] = r'\"'

    # Fix special words
    result = ''.join(fixed_json)
    result = result.replace('True', 'true').replace('False', 'false')

    # Fix delimeters
    result = re.sub(r'"\s*\n\s*"', '",\n"', result)

    return result

def escape_newlines_in_quotes(json_string):
    matches = list(re.finditer(r'(?<!\\)"([^"\\]*(?:\\.[^"\\]*)*)"', json_string, re.DOTALL))
    fixed_json = []
    last_end = 0
    
    for match in matches:
        start, end = match.span()
        text_between_quotes = json_string[start:end]
        escaped_text = text_between_quotes.replace('\n', '\\n')
        fixed_json.append(json_string[last_end:start])
        fixed_json.append(escaped_text)
        last_end = end
    
    fixed_json.append(json_string[last_end:])
    
    return ''.join(fixed_json)

def parse_llm_json(json_string, special_words=['question', 'answer', 'success', 'reference', ',', ':', '\n', '}', '{']):
    json_string = json_string.replace('```json', '').replace('```', '')
    try:
        json.loads(json_string)
    except Exception as err:
        json_string = try_fix_json(json_string, special_words)
        json_string = escape_newlines_in_quotes(json_string)
    
    return json.loads(json_string)

def match_metric_name(metric: str, sentence: str) -> bool:
    pattern = r'[^\u4e00-\u9fa5a-zA-Z]'
    sentence = re.sub(pattern, '', sentence).lower()
    metric = re.sub(pattern, '', metric).lower()

    return metric in sentence
