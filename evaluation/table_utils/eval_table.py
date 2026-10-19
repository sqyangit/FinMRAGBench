
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
import sys
import json
import re
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from table_metric_util import evaluate_prediction, extract_markdown_from_response


def safe_generate(llm, query, image_list, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query, image=image_list, max_new_tokens=8192)
            # assert 'markdown' in response
            assert 'errorCode' not in response
            if not response:
                raise AssertionError
            return response  # 成功直接返回
        except Exception as e:
            print(f"[Retry {attempt}/{max_retries}] Generation failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                print("Maximum retry limit reached. Skipping this sample.")
                # print("response:", response)
                return "ERROR: generation failed"

                
# 离线评估求平均
def evaluate_all_with_average(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["result"]

    all_metrics = []

    for idx, item in enumerate(data):
        answer = item.get("answer", "")
        response = item.get("model_response", "")

        prediction = extract_markdown_from_response(response)
        result = evaluate_prediction(prediction, answer)
        all_metrics.append(result)

        print(f"\n[Sample {idx + 1}] Evaluation:")
        print(result)

    average_result = {}
    if all_metrics:
        keys = all_metrics[0].keys()
        for key in keys:
            average_result[key] = round(np.mean([sample[key] for sample in all_metrics]), 4)

    print("\n[Average Evaluation Result]")
    print(average_result)
    return all_metrics, average_result

def evaluate_all_with_average_2(output_jsonl_path):

    all_metrics = []
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            answer = item.get("answer", "")
            if "|" in answer:
                answer = answer[answer.index("|"):]
            response = item.get("response", "")
            try:
                prediction = extract_markdown_from_response(response)
            except:
                prediction = response
            result = evaluate_prediction(prediction, answer)
            all_metrics.append(result)
            
            print(f"\n[Sample {idx + 1}] Evaluation:")
            print(result)
            # if(idx == 33): break

        average_result = {}
        if all_metrics:
            keys = all_metrics[0].keys()
            for key in keys:
                average_result[key] = round(np.mean([sample[key] for sample in all_metrics]), 4)

        print("\n[Average Evaluation Result]")
        print(average_result)
        return all_metrics, average_result

def main():

    from inference.llm_local import VLM
    from inference.inference_prompt import table_generation_infer_prompt
    import os

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "gpt-4o"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    llm = VLM(model_path)
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')

    input_json_path = os.path.join(PROJECT_ROOT, 'data_bench/table_generation.json')
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["result"]

    output_json_path = "table_generation_rk_7b.json"
    updated_results = []

    for idx, item in enumerate(data):
        print(f"\n================= Example {idx + 1} =================")

        question = item["question"]
        ground_truth = item["answer"].strip()
        image_files = item["evidence_pages"]
        imagelist = [os.path.join(base_path, img) for img in image_files]
        query_vis = table_generation_infer_prompt + question
        # query_vis = question
        if "Qwen" in model_path:
            response = llm.generate(query=query_vis, image=imagelist)
        else:
            # 为远程模型调用设置重试次数
            response = safe_generate(llm, query=query_vis, image_list=imagelist)
            
        print("\n*************** Raw Response ***************")
        print(response)
        item["model_response"] = response
        updated_results.append(item)
        
        # 在线评估
        prediction = extract_markdown_from_response(response)
        print("\n*************** Prediction ***************")
        print(prediction)

        print("\n*************** Ground Truth ***************")
        print(ground_truth)

        result = evaluate_prediction(prediction, ground_truth)
        print("\n*************** Evaluation Result ***************")
        print(result)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"result": updated_results}, f, indent=4, ensure_ascii=False)

    # 离线评估
    evaluate_all_with_average(output_json_path)

if __name__ == '__main__':
    # main()
    # output_json_path = "table_generation_32b.json"
    # evaluate_all_with_average(output_json_path)
    
    exp_type = "vidorag2" # vanilla_rag, tooluse, gold_rag, tooluse_origin, hieret_rag, hrps_rag
    model_type = "gpt-5.1" # Qwen3-VL-8B-SFT-complex, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, gpt-5.1, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, InternVL3_5-8B, InternVL3_5-38B, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, MiniCPM-V-4_5, Llama-3.2-11B-Vision-Instruct


    output_json_path = f"{PROJECT_ROOT}/results/{exp_type}/{model_type}/{exp_type}_{model_type}_table_generation.jsonl"
    # output_json_path = "<PROJECT_PATH>"
    evaluate_all_with_average_2(output_json_path)