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

import json
import os
from tqdm import tqdm
import random
import re
import copy
from loguru import logger
import numpy as np
import json
from transformers import AutoTokenizer


# Config
INPUT_DIR = "[SOURCE_DATASET_DIR]"
OUTPUT_DIR = "[OUTPUT_DATASET_DIR]"
MODEL_PATH = "[PATH_TO_LLM]"
SAMPLE_CNT = 10
MAX_TOKENS = 11000


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def sample_single_item(item: dict) -> dict:
    item = copy.deepcopy(item)
    problem = item["problem"].split('### Output format')[0].strip()
    
    # Randomly choose a window
    timeseries, timestamps = item["timeseries"], item["timestamps"]
    window_size = len(timeseries[0]) // 2
    window_start = random.randint(5, window_size - 5)
    window_end = window_start + window_size
    new_timeseries = [np.array(ts[window_start:window_end]).tolist() for ts in timeseries]
    new_timestamps = [np.array(ts[window_start:window_end]).tolist() for ts in timestamps]

    # Parse solution and convert to component level
    cur_solution = json.loads(item["solution"])
    metrics = item["metrics"]
    components = item["groups"]
    metric_to_component = dict(zip(metrics, components))
    root_cause_metrics = cur_solution["root_cause"]

    # Sample metrics
    target_cnt = random.randint(int(len(metrics) * 0.3), int(len(metrics) * 0.6))
    if len(root_cause_metrics) > target_cnt:
        root_cause_metrics = random.sample(root_cause_metrics, target_cnt)
        sampled_metrics = root_cause_metrics
    else:
        sampled_metrics = random.sample(list(set(metrics) - set(root_cause_metrics)), target_cnt - len(root_cause_metrics)) + root_cause_metrics

    sampled_metrics = list(set(sampled_metrics))
    random.shuffle(sampled_metrics)
    sampled_components = [metric_to_component[m] for m in sampled_metrics]
    sampled_timeseries = [new_timeseries[metrics.index(m)] for m in sampled_metrics]
    sampled_timestamps = [new_timestamps[metrics.index(m)] for m in sampled_metrics]

    # Update problem
    problem = item["problem"].replace(f"There are {len(metrics)} monitoring metrics with detected anomalies", f"There are {len(sampled_metrics)} monitoring metrics with detected anomalies")
    problem_start = problem.split('is: "MetricA" in "ComponentX"): ')[0] + 'is: "MetricA" in "ComponentX"): '
    problem_end = '\n\n## Component Graph\n' + problem.split('\n\n## Component Graph\n')[-1]
    
    problem = problem_start
    for idx, metric in enumerate(sampled_metrics):
        cur_value = sampled_timeseries[idx]
        cur_value_str = ','.join([str(round(x, 2)) for x in cur_value])
        metric_name = metric.split('in ')[0].strip()
        problem += f'\n- "{metric_name}" in "{sampled_components[idx]}" is a metric with length of {len(cur_value)}: [{cur_value_str}]'
    problem += problem_end

    # Replace the anomalies point
    problem = problem.split("has encountered some anomalies near point ")[0] + "has encountered some anomalies near point " + f"{window_size // 2}" + ", and some metrics have" + problem.split(", and some metrics have")[-1]

    # Update result
    item["timeseries"] = sampled_timeseries
    item["timestamps"] = sampled_timestamps
    item["metrics"] = sampled_metrics
    item["groups"] = sampled_components
    new_rank_list = [{"metric": metric} for metric in sampled_metrics]

    result_solution = {
        "root_cause": root_cause_metrics,
        "rank_list": new_rank_list,
        "level": "metric",
        "conclusion": f"The groundtruth root cause is in the following metrics(s): " + ",".join(root_cause_metrics)
    }

    item["solution"] = json.dumps(result_solution, ensure_ascii=False)
    item["problem"] = problem

    return item

# Walk in INPUT_DIR to find json, and create the same json file in the target DIR with the same subdir levels
def sample_train_datasets(input_dir, output_dir):
    # Walk through the input directory
    for root, dirs, files in tqdm(os.walk(input_dir)):
        for file in files:
            if file.endswith('.jsonl'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))
                logger.success(f"Converting {input_file_path}...")
                
                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                # Convert the prompt
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                new_data = []
                for line in tqdm(lines):
                    item = json.loads(line)

                    # Sample a single item
                    for _ in range(SAMPLE_CNT):
                        new_item = sample_single_item(item)
                        # Check if prompt is too long
                        input_ids = tokenizer(new_item["problem"])['input_ids']
                        # logger.info(f"Prompt length: {len(input_ids)}")
                        if len(input_ids) > MAX_TOKENS:
                            logger.error(f"Prompt ({len(input_ids)}) is too long, skip this case...")
                            continue
                        new_data.append(new_item)

                # Write the new data to the output file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for item in new_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                logger.success(f"Sampled {input_file_path} to {output_file_path}. Original Samples: {len(lines)}, New samples: {len(new_data)}")


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert the files
    sample_train_datasets(INPUT_DIR, OUTPUT_DIR)
    logger.success(f"Converted files from {INPUT_DIR} to {OUTPUT_DIR}")
