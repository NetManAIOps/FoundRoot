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

"""
    Source code for generating the warmup SFT dataset using a step-by-step approach with an LLM.
"""

import os
import json
import re
from loguru import logger
from utils.llm_utils import LLMClient
from vllm import SamplingParams
from data.reward_completion import RewardCompletion
from transformers import AutoTokenizer
from loguru import logger
from typing import *
import click


# CONFIG
OUTPUT_PATH = 'result/warmup_dataset'
LLM_MODEL_PATH = "[LLM_MODEL_PATH]"
LLM_GPU_RANGE = list(range(0, 8))

# Hyperparameters for sampling
MAX_CNT = 8000
STEP1_SAMPLE_N = 24      # samples for Step 1
STEP_N_SAMPLE = 2        # samples for Steps 2-4
SAMPLING_PARAMS_1 = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048,
    stop=['## Step 2:'],
    n=1
)
# base params for later steps; override stop and n per step
BASE_SAMPLING_PARAMS = {
    'temperature': 0.6,
    'top_p': 0.95,
    'max_tokens': 4096
}

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)

class StepByStepGenerator:
    def __init__(self, data: dict):
        self.data = data
        self.num_components = len(set(data["groups"]))
        self.components = list(set(data["groups"]))
        self.num_metrics = len(data["metrics"])
        self.problem = data["problem"]
        self.feedback = RewardCompletion(self.data)

    def _apply_chat_template(self, completion: str = ""):
        result = "<｜begin▁of▁sentence｜>You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format, with the answer included between the <answer> and </answer> tags: <think>\n...\n</think>\n<answer>\n{json content}\n</answer><｜User｜>" + self.problem + "\n" + "<｜Assistant｜><think>\n" + completion.strip().lstrip("<think>").lstrip()
        return result

    def generate_one(self, llm_client: LLMClient) -> List[Dict[str, Any]]:
        # 1. build base context prompt
        prefix = (
            f"Alright, we have {self.num_components} components and {self.num_metrics} metrics in total. "
            "I need to perform the following 4 steps to solve this problem: Step 1, Step 2, Step 3, and Step 4. "
            "I will format my thinking steps in the form of '## Step 1', '## Step 2', '## Step 3' and '## Step 4'.\n\n"
        )
        base_context = prefix

        # 2. Step 1: batch generate
        step1_body = (
            base_context +
            f"## Step 1: Analyze all the components\n"
            f"""I will first simply analyze all the {self.num_components} components and their metrics in order (""" + "; ".join(f"{i + 1}. {comp}" for (i, comp) in enumerate(self.components)) + """). After this step, I will start to perform "## Step 2". I will keep this step simple. In the form like: "1. Service A. ...\n2. ...": \n 1."""
        )
        prompts1 = [self._apply_chat_template(step1_body) for _ in range(STEP1_SAMPLE_N)]
        raw1 = llm_client.llm_batch_generate(
            prompts1,
            sampling_params=SAMPLING_PARAMS_1,
            use_chat_template=False
        )  # list of STEP1_SAMPLE_N outputs
        # filter valid by bullet count
        contexts: List[str] = []
        all_cnt, success_cnt = 0, 0
        for out in raw1:
            all_cnt += 1
            # Check token count
            token_count = len(tokenizer.encode(out))
            assert SAMPLING_PARAMS_1.max_tokens is not None
            if token_count > 0.98 * SAMPLING_PARAMS_1.max_tokens:
                logger.warning(f"Token count exceeded: {token_count} > 2048")
                continue

            lines = [ln for ln in f"1. {out}".split("\n") if re.match(r"^\s*\d+\.\s*", ln)]
            print(f"{len(lines)=}")
            if len(lines) == self.num_components:
                contexts.append(step1_body + out + "\n\n")
                success_cnt += 1
        logger.warning(f"Step 1: {success_cnt}/{all_cnt} valid outputs")

        # 3. Steps 2-4: batch per step
        for step in (2, 3, 4):
            # prepare batch prompts
            batch_bodies = []
            raw_bodies = []
            stop_seq = []
            cur_max_tokens = 2048
            for ctx in contexts:
                if step == 2:
                    body = ctx + "## Step 2: Figure out the failure propagation\n\nOK, now I will analyze the failure propagation of the components according to the metrics and their relationship. After this step, I will start to perform ## Step 3.  I will make a detailed analysis about this. Now let's start:\n\n"
                    stop_seq = ['## Step 3:', '</think>', '<answer>', "{\n", "```json"]
                elif step == 3:
                    body = ctx + "## Step 3: Review the process and check the possible errors before\nAlright, let me check if there are any possible errors in the process before. After this step, I will start to perform ## Step 4. I will make a detailed analysis about this. Now let's start:\n"
                    stop_seq = ['## Step 4:', '</think>', '<answer>', "{\n", "```json"]
                else:
                    body = ctx + (
                        f"## Step 4: The rank results in order (from rank 1 to rank {self.num_components}), "
                        "where n is the number of components. After this step, I will output the final answer in the json format between the answer tags. Let's get started:\n 1."
                    )
                    stop_seq = []
                    cur_max_tokens = 4096
                batch_bodies.append(self._apply_chat_template(body))
                raw_bodies.append(body)

            # override sampling params per step
            params = SamplingParams(
                temperature=BASE_SAMPLING_PARAMS['temperature'],
                top_p=BASE_SAMPLING_PARAMS['top_p'],
                max_tokens=cur_max_tokens,
                stop=stop_seq,
                n=STEP_N_SAMPLE
            )
            # batch inference
            raw_batches = llm_client.llm_batch_generate(
                batch_bodies,
                sampling_params=params,
                use_chat_template=False
            )  # list of lists, len(raw_batches)==len(batch_bodies)

            new_contexts = []
            all_cnt, success_cnt = 0, 0
            # unwrap outputs
            for ctx, outs in zip(raw_bodies, raw_batches):
                for out in outs:
                    all_cnt += 1

                    # Check token count
                    token_count = len(tokenizer.encode(out))
                    if token_count > 0.98 * cur_max_tokens:
                        logger.warning(f"Token count exceeded: {token_count} > {cur_max_tokens}")
                        continue

                    # validate bullets for Step 4
                    if step == 4:
                        lines = [ln for ln in f"1. {out}".split("\n") if re.match(r"^\s*\d+\.\s*", ln)]
                        if len(lines) != self.num_components:
                            continue
                        success_cnt += 1
                    else:
                        success_cnt += 1
                    new_contexts.append(ctx + out + "\n\n")
            contexts = new_contexts
        
            logger.warning(f"Step {step}: {success_cnt}/{all_cnt} valid outputs")

        # 4. finalize records
        records: List[Dict[str, Any]] = []
        for full_output in contexts:
            score = self.feedback._reward_completions(full_output)
            records.append({
                "input": self.problem,
                "output": full_output.strip(),
                "reward": score
            })
        logger.success(f"Generated {len(records)} records")
        return records


@click.command()
@click.option('--dataset', '-d', default='b', help='Path to the input JSONL file')
def generate_dataset(dataset: str):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    llm_client = LLMClient(
        LLM_MODEL_PATH,
        engine="vllm",
        gpu_range=LLM_GPU_RANGE,
        batch_size=8,
        gpus_per_model=2,
    )
    llm_client.wait_for_ready()

    cur_cnt = 0

    content = [json.loads(line) for line in open(dataset)]
    for data in content:
        case_idx = data["case_idx"]
        out_file = os.path.join(OUTPUT_PATH, f"{case_idx}.jsonl")
        if os.path.exists(out_file):
            logger.warning(f"Skip existing {out_file}")
            cur_cnt += len(open(out_file).readlines())
            continue

        logger.info(f"Generating case {case_idx}...")
        gen = StepByStepGenerator(data)
        recs = gen.generate_one(llm_client)

        with open(out_file, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        cur_cnt += len(recs)
        if cur_cnt >= MAX_CNT:
            logger.warning(f"Reached max count {MAX_CNT}, stopping...")
            break

    llm_client.kill()

if __name__ == "__main__":
    generate_dataset()
