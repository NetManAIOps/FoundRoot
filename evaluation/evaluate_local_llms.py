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

from evaluation.utils import evaluate_cases
from utils.llm_utils import LLMClient
from vllm import SamplingParams
from datasets import load_dataset

import json
import os
import numpy as np
from typing import List


# Config
EXP = "causal-foundroot"
MODEL_PATH = "[LLM_MODEL_PATH]"

DATASET_SPLIT = "test"
NUM_GPUS = 4
NUM_GPUS_PER_PROCESS = 1
SAMPLE_N = 1


EVALUATION_DATASET_DICT = {
    "a": "../datasets/a",
    "b": "../datasets/b",
    "c": "../datasets/c",
    "d": "../datasets/d"
}

if "sft_checkpoints" in MODEL_PATH:
    if "sft" in os.path.basename(MODEL_PATH) or "sft" in EXP:
        SYSTEM_PROMPT = "You are a helpful assistant"
    else:
        SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format, with the answer included between the <answer> and </answer> tags: <think>\n...\n</think>\n<answer>\n{json content}\n</answer>"

    # Set CHAT_TEMPLATE for deepseek-r1 model
    if "foundroot" in MODEL_PATH.lower() or "foundroot" in EXP.lower() or "r1" in EXP.lower():
        CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"
    else:
        CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
else:
    CHAT_TEMPLATE = None

    if "r1" in MODEL_PATH.lower():
        SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format, with the answer included between the <answer> and </answer> tags: <think>\n...\n</think>\n<answer>\n{json content}\n</answer>"
    else:
        SYSTEM_PROMPT = "You are a helpful assistant."

# Load dataset and evaluate
def evaluate():
    # Load dataset
    dataset_names = []
    dataset_items = []

    for name, path in EVALUATION_DATASET_DICT.items():
        print(f"[evaluate] Loading dataset {name}")
        dataset = load_dataset(path, split=DATASET_SPLIT)
        for item in dataset:
            dataset_names.append(name)
            dataset_items.append(item)
    print(f"[evaluate] Loaded {len(dataset_items)} cases")

    # Evaluate
    print(f"[evaluate] Evaluating...")
    question_list = []
    label_list = []
    for case in dataset_items:
        question_list.append(case["problem"])
        label_list.append(case["solution"])
    
    # Inference
    if os.path.exists(f"exp/{EXP}/generated_answer.json"):
        print(f"[evaluate] Loading generated_answer...")
        answer_list = json.load(open(f"exp/{EXP}/generated_answer.json", 'r'))
    else:
        print(f"[evaluate] Inference...")
        # Initialize model
        print(f"[evaluate] Initializing model {MODEL_PATH}")
        llm_client = LLMClient(model_path=MODEL_PATH, engine='vllm', num_gpus=NUM_GPUS, gpus_per_model=NUM_GPUS_PER_PROCESS, batch_size=8, system_prompt=SYSTEM_PROMPT, chat_template=CHAT_TEMPLATE, sample_n=SAMPLE_N)
        print(f"[evaluate] Initialized model {MODEL_PATH}")
        sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=10000, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'], n=SAMPLE_N)
        answer_list = llm_client.llm_batch_generate(question_list, use_chat_template=True, sampling_params=sampling_params)
        llm_client.kill()

        # Save generated_answer
        print(f"[evaluate] Saving generated_answer...")
        os.makedirs(f"exp/{EXP}", exist_ok=True)
        json.dump(answer_list, open(f"exp/{EXP}/generated_answer.json", 'w'), indent=4, ensure_ascii=False)

    # Evaluate
    print(f"[evaluate] Evaluating...")

    # Check if sample_n > 1
    if type(answer_list[0]) == list:
        # process dataset_names, answer_list, and dataset_items
        print(f"[evaluate] Sample_n > 1, processing dataset_names, answer_list, and dataset_items...")
        new_dataset_names = []
        new_answer_list = []
        new_dataset_items = []
        for i, name in enumerate(dataset_names):
            for j in range(SAMPLE_N):
                new_dataset_names.append(name)
                new_answer_list.append(answer_list[i][j])
                new_dataset_items.append(dataset_items[i])
        dataset_names = new_dataset_names
        answer_list = new_answer_list
        dataset_items = new_dataset_items

    dataset_result_list = evaluate_cases(dataset_names, answer_list, dataset_items)

    # Print and save result
    print(f"[evaluate] Result:")
    for result in dataset_result_list:
        print(f"[evaluate] Dataset: {result['dataset']}, Top-1: {result['top1']}, Top-3: {result['top3']}, MRR: {result['mrr']}, Success Parsed: {result['success']}, Total: {result['total']}")

    # Save result
    print(f"[evaluate] Saving result...")
    os.makedirs(f"exp/{EXP}", exist_ok=True)
    for result in dataset_result_list:
        json.dump(result, open(f"exp/{EXP}/{result['dataset']}_result.json", 'w'), indent=4, ensure_ascii=False)

    print(f"[evaluate] Done!")


if __name__ == "__main__":
    evaluate()
