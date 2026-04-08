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

from causal.metric_utils import match_metric_name
from utils.llm_utils import parse_llm_json
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
import json


def calculate_top1(content: dict, sol: dict, metric: List[str], group: List[str]) -> float:
    """RCA Reward function that uses Top-1 accuracy to evaluate the top predicted root cause."""
    try:
        # Determine evaluation level
        if sol.get("level", "metric") == "metric":
            # Map each metric to its group
            metric_to_group = {m: g for m, g in zip(metric, group)}
            # Ground truth group for the first ground-truth metric
            gt_root_cause_group = metric_to_group.get(sol["rank_list"][0]["metric"])

            # Build predicted group ranking
            group_rank = []
            for item in content["rank_list"]:
                placed = False
                # Prioritize actual root-cause metrics
                if "root_cause" in sol:
                    for rc in sol["root_cause"]:
                        if match_metric_name(item["metric"], rc):
                            group_rank.append(gt_root_cause_group)
                            placed = True
                            break
                    if placed:
                        continue
                # Otherwise, assign based on metric-to-group mapping
                for m in metric:
                    if match_metric_name(item["metric"], m):
                        group_rank.append(metric_to_group[m])
                        placed = True
                        break
                # If no match, mark as unknown
                if not placed:
                    group_rank.append("Unknown")

            # Top-1: reward = 1 if first predicted group matches ground truth, else 0
            reward = 1.0 if group_rank and group_rank[0] == gt_root_cause_group else 0.0

        else:
            # When evaluating at the component/group level
            gt_root_cause = set(sol["root_cause"])
            group_rank = []
            all_groups = list(set([item['component'] for item in sol['rank_list']]))
            for item in content["rank_list"]:
                # Find the matching component group
                comp_match = next((comp for comp in all_groups if match_metric_name(item.get("component", ""), comp)), None)
                if comp_match and comp_match not in group_rank:
                    group_rank.append(comp_match)
                elif not comp_match:
                    group_rank.append("Unknown")

            # Top-1: reward = 1 if first predicted group is one of the ground truth groups
            reward = 1.0 if group_rank and group_rank[0] in gt_root_cause else 0.0

    except Exception as err:
        logger.warning(f"[rca_accuracy_reward] Failed to calculate Top-1 reward: {content} ({err})")
        reward = 0.0

    return reward

def rca_mrr_reward(content: dict, sol: dict, metric: List[str], group: List[str]):
    """RCA Reward function that uses MRR to evaluate the rank of the ground truth root cause."""    
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
                            if cur_group not in group_rank:
                                group_rank.append(cur_group)
                            flag = True
                            break
                if flag:
                    continue
                for m in metric:
                    if match_metric_name(item["metric"], m):
                        cur_group = metric_to_group[m]
                        if cur_group not in group_rank:
                            group_rank.append(cur_group)
                        break
                else:
                    # If no match found, add a placeholder
                    group_rank.append("Unknown")

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
            all_groups = list(set([item['component'] for item in sol['rank_list']]))
            for item in content["rank_list"]:
                ans_group = None
                for comp in all_groups:
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
        logger.warning(f"[rca_accuracy_reward] Failed to calculate MRR: {content} ({err})")
        mrr_reward = 0.0

    return mrr_reward

def rca_propagation_reward(answer: dict, sol: dict, **kwargs):
    """
    RCA Reward function that checks the accuracy of upstream and type fields for each metric.
    """
    # Calculate propagation reward
    try:            
        # Track accuracy for each metric
        total_metrics = len(sol["rank_list"])
        total_accuracy = 0.0
        
        for sol_metric in sol["rank_list"]:
            metric_name = sol_metric["metric"]
            
            # Check if metric exists in content
            if any(match_metric_name(metric_name, content_metric["metric"]) for content_metric in answer["rank_list"]):
                # Find the matching content metric
                content_metric = next(item for item in answer["rank_list"] 
                                        if match_metric_name(metric_name, item["metric"]))
                
                # Check type accuracy
                if "type" not in sol_metric:
                    type_accuracy = 1.0
                else:
                    type_accuracy = 1.0 if sol_metric["type"] == content_metric["type"] else 0.0
                
                # Check upstream accuracy
                if "upstream" not in sol_metric:
                    upstream_accuracy = 1.0
                else:
                    upstream_accuracy = 1.0 if match_metric_name(sol_metric["upstream"], content_metric["upstream"]) else 0.0
                
                # Average accuracy for this metric
                metric_accuracy = (type_accuracy + upstream_accuracy) / 2.0
                total_accuracy += metric_accuracy
            else:
                # Metric not found in content, accuracy is 0
                total_accuracy += 0.0
        
        # Calculate average accuracy across all metrics
        propagation_reward = total_accuracy / total_metrics if total_metrics > 0 else 0.0
            
    except Exception as err:
        print(f"[rca_propagation_reward] Failed to calculate propagation reward: {err}")
        propagation_reward = 0.0

    return propagation_reward

def parse_answer(answer: str) -> Optional[dict]:
    if '<answer>' in answer and '</answer>' in answer:
        answer = answer.split('<answer>')[1].split('</answer>')[0]
    answer = answer.strip().replace('```json', '').replace('```', '')
    if '{' in answer and '}' in answer:
        answer = answer[answer.find('{'):answer.rfind('}')+1]

    try:
        parsed_result = parse_llm_json(answer, special_words=['metric', 'component', 'conclusion', ',', ':', '\n', '}', '{'])
    except:
        return None
    
    return parsed_result

def evaluate_rca_case(answer: str, case: dict):
    answer_dict = parse_answer(answer)
    label_dict = json.loads(case["solution"])
    metrics = case["metrics"]
    groups = case["groups"]

    if answer_dict is None:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "mrr": 0.0,
            # "propagation": 0.0,
            "success": False
        }
    
    # Calculate top1
    top1 = calculate_top1(answer_dict, label_dict, metrics, groups)
    # Calculate mrr
    mrr = rca_mrr_reward(answer_dict, label_dict, metrics, groups)
    top3 = float(mrr > 0.32)
    # Calculate propagation
    # propagation = rca_propagation_reward(answer_dict, label_dict)

    return {
        "top1": top1,
        "top3": top3,
        "mrr": mrr,
        # "propagation": propagation,
        "success": True
    }

def evaluate_cases(dataset_name_list: List[str], answer_list: List[str], case_list: List[dict]):
    top1_list = []
    top3_list = []
    mrr_list = []
    # propagation_list = []
    success_list = []

    detail_list = []

    for idx in range(len(answer_list)):
        answer = answer_list[idx]
        case = case_list[idx]
        result = evaluate_rca_case(answer, case)
        top1_list.append(result['top1'])
        top3_list.append(result['top3'])
        mrr_list.append(result['mrr'])
        # propagation_list.append(result['propagation'])
        success_list.append(result['success'])
        
        # Add details
        result['answer'] = answer
        result['label'] = case['solution']
        result['idx'] = idx
        detail_list.append(result)

    # Avg by dataset_name
    top1_list = np.array(top1_list)
    top3_list = np.array(top3_list)
    mrr_list = np.array(mrr_list)
    # propagation_list = np.array(propagation_list)
    success_list = np.array(success_list)
    # Save details
    dataset_result_list = []
    # Save details
    for dataset_name in set(dataset_name_list):
        idx_list = [idx for idx, name in enumerate(dataset_name_list) if name == dataset_name]
        dataset_result_list.append({
            "dataset": dataset_name,
            "top1": np.mean(top1_list[idx_list]),
            "top3": np.mean(top3_list[idx_list]),
            "mrr": np.mean(mrr_list[idx_list]),
            # "propagation": np.mean(propagation_list[idx_list]),
            "total": len(idx_list),
            "success": len(success_list[idx_list]),
            "detail": [detail_list[i] for i in idx_list]
        })

    # Calculate overall
    dataset_result_list.append({
        "dataset": "overall",
        "top1": np.mean(top1_list),
        "top3": np.mean(top3_list),
        "mrr": np.mean(mrr_list),
        # "propagation": np.mean(propagation_list),
        "total": len(answer_list),
        "success": len(success_list),
        "detail": detail_list
    })

    return dataset_result_list
