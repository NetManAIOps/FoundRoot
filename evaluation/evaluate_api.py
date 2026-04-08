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
from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from openai import OpenAI
import os
import json
from tqdm import tqdm
import random
import time


# Config
MODEL_NAME = "deepseek-r1"
EXP = f"api-{MODEL_NAME}-new"
EVALUATION_DATASET_DICT = {
    "a": "../datasets/a",
    "b": "../datasets/b",
    "c": "../datasets/c",
    "d": "../datasets/d"
}
DATASET_SPLIT = "test"
SYSTEM_PROMPT = "You are a helpful assistant."
BASE_URL = "[LLM_BASE_URL (e.g., https://api.openai.com/v1)]"
API_KEY = "[API_KEY]"
MODEL_ID = MODEL_NAME

NUM_WORKERS = 32

print(f"[evaluate] Using {MODEL_NAME} with {NUM_WORKERS} workers")

# Initialize OpenAI Client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def _infer_single(args):
    idx, question = args
    max_retries = 10

    for attempt in range(1, max_retries + 1):
        try:
            messages = []
            if "r1" not in EXP.lower():
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": question})

            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=8000
            )
            answer = completion.choices[0].message.content
            return idx, answer
        except Exception as e:
            print(f"[inference] Error at {idx} on attempt {attempt}: {e}")
            if attempt < max_retries:
                wait_seconds = random.randint(30, 60)
                print(f"[inference] Waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)
            else:
                print(f"[inference] Failed after {max_retries} attempts.")
                return idx, ""

def generate_answers_openai(question_list, num_workers=NUM_WORKERS):
    answers = [None] * len(question_list)
    args_list = list(enumerate(question_list))

    print(f"{num_workers=}")
    with Pool(processes=num_workers) as pool:
        for idx, answer in tqdm(pool.imap_unordered(_infer_single, args_list), total=len(args_list), desc="Inferencing..."):
            answers[idx] = answer

    return answers

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

    question_list = [case["problem"] for case in dataset_items]
    label_list = [case["solution"] for case in dataset_items]

    # Inference
    output_path = f"exp/{EXP}/generated_answer.json"
    answer_list = ["" for _ in range(len(question_list))]
    if os.path.exists(output_path):
        print(f"[evaluate] Loading generated answers from {output_path}...")
        answer_list = json.load(open(output_path, 'r'))

    # Check if any remaining idx to inference
    idx_to_infer = list(range(len(question_list)))
    for idx, answer in enumerate(answer_list):
        if len(answer):
            idx_to_infer.remove(idx)
    question_to_infer = [question_list[i] for i in idx_to_infer]

    if len(question_to_infer) > 0:
        print(f"[evaluate] Inference {len(question_to_infer)} questions with OpenAI API...")
        answer_to_infer = generate_answers_openai(question_to_infer)

        for idx, answer in zip(idx_to_infer, answer_to_infer):
            answer_list[idx] = answer

        # Save answers
        os.makedirs(f"exp/{EXP}", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(answer_list, f, indent=4, ensure_ascii=False)

    # Evaluate
    print(f"[evaluate] Evaluating...")
    dataset_result_list = evaluate_cases(dataset_names, answer_list, dataset_items)

    # Print and save result
    print(f"[evaluate] Result:")
    for result in dataset_result_list:
        print(f"[evaluate] Dataset: {result['dataset']}, Top-1: {result['top1']}, Top-3: {result['top3']}, MRR: {result['mrr']}, Success Parsed: {result['success']}, Total: {result['total']}")

    # Save result
    for result in dataset_result_list:
        result_path = f"exp/{EXP}/{result['dataset']}_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[evaluate] Done!")

if __name__ == "__main__":
    evaluate()
