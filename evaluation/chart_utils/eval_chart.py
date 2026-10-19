
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
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from chart_process import *
from chart_metric_util import *
from matplotlib import pyplot as plt

def extract_finalanswer_from_response(response):
    if "[Final Answer]:" in response:
        final_answer = response.split("[Final Answer]:")[-1].strip()
        return final_answer
    elif "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[-1].strip()
        return final_answer
    else:
        return response

@timeout(15)
def execute(c):
    # exec(c)
    exec(c, globals())

@timeout(20)
def exec_and_get_y_reference(answer_code, chart_type):
    ecr_1 = False
    python_code, eval_code = build_eval_code(answer_code, chart_type)
    print("Code:", python_code)
    if python_code == "":
        return "", False
    try:
        python_code = surround_pycode_with_main(python_code)
        execute(python_code)
        ecr_1 = True
        plt.close('all')
    except Exception as e:
        print("Python Error: ",e)
        ecr_1 = False
        return "", False
    if ecr_1:
        pass
    try: 
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            chart_eval_code = surround_pycode_with_main(eval_code)
            execute(chart_eval_code)
        except Exception as e:
            print("Eval Error: ",e)
            return "", True
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
        print("OUTPUT VALUE: ",output_value)
    except Exception as e:
        print("Eval Error: ",e)
        output_value = ''

    if output_value != '':
        parsed_prediction = output_value.strip()
    else:
        parsed_prediction = ''
    plt.close('all')
    return parsed_prediction, ecr_1

def safe_generate(llm, query, image_list, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query, image=image_list, max_new_tokens=8192)
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
        metric_scores = {}
        answer = item.get("answer", "")
        response = item.get("model_response", "")
        chart_type = item.get("subtype", "").split()[0]

        final_answer = extract_finalanswer_from_response(response)
        python_code = final_answer
        prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
        metric_scores['ECR'] = ecr_1

        if prediction != '':
            try:
                prediction = ast.literal_eval(prediction)
                reference = ast.literal_eval(answer)
                if chart_type == 'PieChart': metric_scores['Pass'] = compute_pie_chart_metric_loose(reference, prediction)
                else: metric_scores['Pass'] = compute_general_chart_metric_loose(reference, prediction)
            except Exception as e:
                metric_scores['Pass'] = False
        else:
            metric_scores['Pass'] = False
            # metric_scores['Pass'] = 'None'

        result = {
            'ECR': metric_scores.get('ECR', ''),
            'Pass': metric_scores.get('Pass', '')
        }
        all_metrics.append(result)

        print(f"\n[Sample {idx + 1}] Evaluation:")
        print(result)
        print("-----------------------------------------")
        # break

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
            metric_scores = {}
            answer = item.get("answer", "")
            response = item.get("response", "") or ""
            chart_type = item.get("subtype", "").split()[0]

            final_answer = extract_finalanswer_from_response(response)
            python_code = final_answer
            prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
            metric_scores['ECR'] = ecr_1

            if prediction != '':
                try:
                    prediction = ast.literal_eval(prediction)
                    reference = ast.literal_eval(answer)
                    if chart_type == 'PieChart': metric_scores['Pass'] = compute_pie_chart_metric_loose(reference, prediction)
                    else: metric_scores['Pass'] = compute_general_chart_metric_loose(reference, prediction)
                except Exception as e:
                    metric_scores['Pass'] = False
            else:
                metric_scores['Pass'] = False
                # metric_scores['Pass'] = 'None'

            result = {
                'ECR': metric_scores.get('ECR', ''),
                'Pass': metric_scores.get('Pass', '')
            }
            all_metrics.append(result)

            print(f"\n[Sample {idx + 1}] Evaluation:")
            print(result)
            print("-----------------------------------------")
            # break

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
    from inference.inference_prompt import chart_generation_infer_prompt2
    import os

    # model_path = "<PROJECT_PATH>"
    model_path = "gpt-4o"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    llm = VLM(model_path)
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')

    input_json_path = os.path.join(PROJECT_ROOT, 'data_bench/chart_generation.json')
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["result"]

    output_json_path = "chart_generation_4o_gold2.json"
    updated_results = []

    for idx, item in enumerate(data):
        print(f"\n================= Example {idx + 1} =================")
        metric_scores = {}
        question = item["question"]
        ground_truth = item["answer"].strip()
        image_files = item["evidence_pages"]
        chart_type = item["subtype"].split()[0]
        imagelist = [os.path.join(base_path, img) for img in image_files]
        query_vis = chart_generation_infer_prompt2 + f"\nThe titles of the provided images are: [ {', '.join(item['evidence_pages'])} ]\nQuestion: " + question

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
        final_answer = extract_finalanswer_from_response(response)
        python_code = final_answer
        prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
        metric_scores['ECR'] = ecr_1
        print("\n*************** Prediction ***************")
        print(ecr_1, prediction)

        print("\n*************** Ground Truth ***************")
        print(ground_truth)


        if prediction != '':
            try:
                prediction = ast.literal_eval(prediction)
                reference = ast.literal_eval(ground_truth)
                if chart_type == 'PieChart': metric_scores['Pass'] = compute_pie_chart_metric_loose(reference, prediction)
                else: metric_scores['Pass'] = compute_general_chart_metric_loose(reference, prediction)
            except Exception as e:
                metric_scores['Pass'] = 'False'
        else:
            metric_scores['Pass'] = 'None'

        print("\n*************** Evaluation Result ***************")
        eval_result = {
            'ECR': metric_scores.get('ECR', ''),
            'Pass': metric_scores.get('Pass', '')
        }
        print(eval_result)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"result": updated_results}, f, indent=4, ensure_ascii=False)

    # 离线评估
    evaluate_all_with_average(output_json_path)

if __name__ == '__main__':
    # main()
    # output_json_path = "chart_generation_4o_gold.json"
    # evaluate_all_with_average(output_json_path)

    exp_type = "vidorag2" # vanilla_rag, tooluse, gold_rag, tooluse_origin, hieret_rag, hrps_rag
    model_type = "gpt-5.1" # Qwen3-VL-8B-SFT-complex, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, gpt-5.1, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, InternVL3_5-8B, InternVL3_5-38B, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, MiniCPM-V-4_5, Llama-3.2-11B-Vision-Instruct

    output_json_path = f"{PROJECT_ROOT}/results/{exp_type}/{model_type}/{exp_type}_{model_type}_chart_generation.jsonl"
    # output_json_path = "<PROJECT_PATH>"
    evaluate_all_with_average_2(output_json_path)