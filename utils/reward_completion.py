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

import numpy as np
import random
from tqdm import tqdm
from utils.llm_utils import parse_llm_json, LLMClient, match_metric_name
import networkx as nx
from typing import Any, Dict, Tuple, Set, List, Optional, Type, Union
from multiprocessing import Pool, cpu_count
from loguru import logger
import json
import os


class RewardCompletion:
    def __init__(self, data):
        self.data = data

        # Feedback Tree
        self.remaining_completions = [""]
        self.finished_completions = []

        # Prompt
        self.input_prompt = data["problem"]
        self.initial_prompt = "You are an expert assistant specializing in reviewing Root Cause Analysis (RCA) reasoning chains based on system metrics. Your goal is to carefully identify major flaws of the student's reasoning chain in the reasoning and provide first-person reflective corrections."
        self.problem_solution_prompt = self._extract_problem_solution()
        self.task_prompt = """You are given a reasoning chain for a Root Cause Analysis (RCA) task in a microservice system.  
Each step of the reasoning is numbered as "[line i] ...".  

Your task is:

1. Carefully read the entire reasoning chain.
2. Identify the **single most critical error** based on the following error types:
    - `missing_metric`: Missing an important metric that should have been analyzed
    - `wrong_causality`: Incorrect causal reasoning between metrics
    - `wrong_summary`: Incorrectly summarizing or omitting key analysis points
    - `premature_conclusion`: Drawing a conclusion too early without sufficient evidence
    - `conflicting_analysis`: Contradictory statements within the reasoning chain

3. For the most critical error:
    - Specify the **line_number** (integer), which indicates where the error initially occurs.  
      (This is where the first wrong statement or flawed reasoning appears.)
    - Specify the **insert_line_number** (integer), which indicates where the first-person reflection should be inserted **after**.  
      (This should be as late as possible — after the error has significantly affected the reasoning, but before the final conclusion is made.)
      Clearly distinguish:
      - `line_number` = where the mistake **first appears**.
      - `insert_line_number` = where the mistake's **impact becomes serious enough to need correction**, but still **before conclusions are finalized**. The relection will be inserted after the insert_line_number (so this line will still be kept).
    - Write a **first-person natural reflection** that acknowledges and corrects the mistake.
    - Assign the correct **error_type** (choose from the list above).

4. Strict rules for writing the reflection:
    - Focus only on recognizing and correcting the specific mistake.
    - **Do not introduce or reveal the final root cause.**
    - **Do not add any new conclusions or hints about the final answer.**
    - Make the reflection sound like a natural self-check, only addressing the flaw. (starts with words like: Wait / I think I should / But / ... )

4. Output your result strictly in the following JSON format (without list brackets `[]`):

{
  "line_number": <integer>,
  "insert_line_number": <integer>,
  "reflection": "<first-person reflection>",
  "error_type": "<one of the error types>"
}

Important notes:
- Focus only on the most significant error.
- Do not fabricate new information.
- Ensure the reflection is natural and consistent with the existing reasoning tone.
- Do not introduce or reveal the final root cause."""

    def _extract_problem_solution(self):
        result = self.data["problem"].split("### Output format")[0].rstrip().replace("## Basic Information", "## RCA Problem")
        result += "\n\n## RCA Label\nThe **groundtruth** root cause metrics of this case is: " + json.loads(self.data["solution"])["conclusion"]
        return result

    def _feedback_prompt(self, completion: str):
        # Split completion by line
        think_steps = completion.split("<think>")[-1].split("</think>")[0]
        answer = completion.split("</think>")[-1].replace('<answer>', '').replace('</answer>', '')
        lines = think_steps.split('\n')

        # Extract the reasoning chain
        reasoning_chain = []
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                reasoning_chain.append(f"[line {len(reasoning_chain)}] " + line)

        reasoning_prompt = "\n".join(reasoning_chain)

        # Construct the full prompt
        prompt = f"""{self.initial_prompt}

{self.problem_solution_prompt}

## Student's Reasoning Chain
{reasoning_prompt}

## Student's Answer (rank_list from the root cause to the most unlikely one)
{answer}

## Your Task
{self.task_prompt}

## Your Answer\n"""

        return prompt

    def _apply_chat_template(self, completion: str = ""):
        result = "<｜begin▁of▁sentence｜>You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format, with the answer included between the <answer> and </answer> tags: <think>\n...\n</think>\n<answer>\n{json content}\n</answer><｜User｜>" + self.input_prompt + "\n" + "<｜Assistant｜><think>\n" + completion.strip().lstrip("<think>").lstrip()
        return result
    
    def _reward_completions(self, completion: str) -> float:
        """RCA Reward function that uses MRR to evaluate the rank of the ground truth root cause."""
        # Try parse content
        content = completion
        sol = self.data['solution']
        metric = self.data['metrics']
        group = self.data['groups']
        try:
            # Extract items between <answer> and </answer>
            if '<answer>' in content and '</answer>' in content:
                content = content.split('<answer>')[-1].split('</answer>')[0]
            content = content.strip().replace('```json', '').replace('```', '')
            if '{' in content and '}' in content:
                content = content[content.find('{'):content.rfind('}') + 1]
            content = parse_llm_json(content, special_words=['metric', 'component', 'conclusion', ',', ':', '\n', '}', '{', 'upstream', 'type', 'description'])
            sol = json.loads(sol)
        except Exception as err:
            logger.error(f"[rca_accuracy_reward] Failed to parse content: {content} or sol: {sol} ({err})")
            mrr_reward = 0.0
            return mrr_reward
        
        # Calculate MRR reward
        try:
            if sol.get("level", "metric") == "metric":
                metric_to_group = dict((m, g) for m, g in zip(metric, group))
                gt_root_cause_group = metric_to_group[sol["rank_list"][0]["metric"]]

                # Get rank of group
                group_rank = []
                for item in content["rank_list"]:
                    # Check if the metric in root_cause list
                    flag = False
                    if "root_cause" in sol:
                        for root_cause_metric in sol["root_cause"]:
                            if match_metric_name(item["metric"], root_cause_metric):
                                cur_group = gt_root_cause_group
                                # print(f"[DEBUG RCA_ACC] {idx=} {item['metric']} -> {cur_group} (root cause group)")
                                if cur_group not in group_rank:
                                    group_rank.append(cur_group)
                                flag = True
                                break
                    if flag:
                        continue
                    for m in metric:
                        if match_metric_name(item["metric"], m):
                            cur_group = metric_to_group[m]
                            # print(f"[DEBUG RCA_ACC] {idx=} {item['metric']} -> {cur_group}")
                            if cur_group not in group_rank:
                                group_rank.append(cur_group)
                            break
                    else:
                        # If no match found, add a placeholder
                        group_rank.append("Unknown")
                        # print(f"[DEBUG RCA_ACC] {idx=} {item['metric']} -> Unknown")

                # Find position of ground truth in predicted rank_list
                mrr_reward = 0.0
                for i, item in enumerate(group_rank):
                    if item == gt_root_cause_group:
                        # Calculate MRR: 1/position (1-indexed)
                        mrr_reward = 1.0 / (i + 1)
                        break
            else:
                gt_root_cause_group = sol["root_cause"]
                group_rank = []
                for item in content["rank_list"]:
                    ans_group = None
                    for comp in group:
                        if match_metric_name(item["component"], comp):
                            ans_group = comp
                            break
                    if ans_group is None:
                        ans_group = "Unknown"
                    if ans_group not in group_rank:
                        group_rank.append(ans_group)
                # Find position of ground truth in predicted rank_list
                mrr_reward = 0.0
                for i, item in enumerate(group_rank):
                    if item in sol["root_cause"]:
                        # Calculate MRR: 1/position (1-indexed)
                        mrr_reward = 1.0 / (i + 1)
                        break
        except Exception as err:
            logger.error(f"[rca_accuracy_reward] Failed to calculate MRR: {content} ({err})")
            mrr_reward = 0.0

        return mrr_reward
