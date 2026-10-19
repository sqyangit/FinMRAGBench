
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_numerical.py
import json
import re
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from metrics.qa_metrics import QAMetric


def extract_boxed_value(text: str) -> str:
    """
    提取 \boxed{...} 里的内容
    """
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return text.strip()

def get_final_answer(response):
    final_answer = None
    if "[Final Answer]:" in response:
        final_answer = response.split("[Final Answer]:")[-1].strip()
    elif "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[-1].strip()
    else :
        final_answer = "No final answer found"
    return final_answer


def safe_generate(llm, query, image_list, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query, image=image_list)
            return response  # 成功直接返回
        except Exception as e:
            print(f"[Retry {attempt}/{max_retries}] Generation failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                print("Maximum retry limit reached. Skipping this sample.")
                return "ERROR: generation failed"

# 离线评估求平均
def evaluate_all_with_average(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["result"]

    all_metrics = []

    for idx, item in enumerate(data):
        answer = item.get("answer", "")
        response = item.get("model_response", "")

        # 标准化
        answer = answer.replace('$', '').replace('millions', '').replace('million', '').strip()
        answer = answer.replace('&amp;', '&')
        prediction = get_final_answer(response).replace('$', '').replace('millions', '').replace('million', '').strip()
        prediction = prediction.replace('&amp;', '&')
        prediction = extract_boxed_value(prediction)

        metric_scores = qa_metric.compute([answer], [prediction])
        result = {
            'F1': metric_scores.get('F1', ''),
            'EM': metric_scores.get('EM', ''),
        }

        all_metrics.append(result)

        print(f"\n[Sample {idx + 1}] Evaluation:")
        print(result)
        print("-----------------------------------------")

    average_result = {}
    if all_metrics:
        keys = all_metrics[0].keys()
        for key in keys:
            average_result[key] = round(np.mean([sample[key] for sample in all_metrics]), 4)

    print("\n[Average Evaluation Result]")
    print(average_result)
    
    return all_metrics, average_result


def evaluate_all_with_average2(output_jsonl_path):
    all_metrics = []

    # 逐行读取 jsonl，每行是一个 JSON 对象
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)

            answer = item.get("answer", "")
            response = item.get("response", "")

            # import pdb;pdb.set_trace()
            if isinstance(response, str):
                s = response.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            response = ", ".join(map(str, parsed))
                        else:
                            response = s
                    except Exception:
                        # 解析失败就保持原样
                        response = s
            # import pdb;pdb.set_trace()

            # 标准化
            answer = answer.replace('$', '').replace('millions', '').replace('million', '').strip()
            answer = answer.replace('&amp;', '&')

            # 按实验输出格式选择直接答案或提取 Final Answer 字段
            try :
                if "tooluse_Qwen3-VL-8B-SFT" in output_jsonl_path or "tooluse_origin" in output_jsonl_path or "react" in output_jsonl_path or "tooluse_gpt" in output_jsonl_path or "vidorag" in output_jsonl_path:
                    prediction = response.replace('&amp;', '&')
                else:
                    prediction = get_final_answer(response).replace('$', '').replace('millions', '').replace('million', '').strip()
                    prediction = prediction.replace('&amp;', '&')
            except Exception as e:
                print(f"Error: {e}")
                prediction = "None"
                # import pdb;pdb.set_trace()

            prediction = extract_boxed_value(prediction)

            metric_scores = qa_metric.compute([answer], [prediction])
            result = {
                'F1': metric_scores.get('F1', ''),
                'EM': metric_scores.get('EM', ''),
            }

            all_metrics.append(result)

            print(f"\n[Sample {idx + 1}] Evaluation:")
            print(result)
            print("-----------------------------------------")

    # 计算平均值
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
    from inference.inference_prompt import numerical_reasoning_infer_prompt
    import os

    # model_path = "<PROJECT_PATH>"
    model_path = "gpt-4o"

    
    llm = VLM(model_path)
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')

    input_json_path = os.path.join(PROJECT_ROOT, 'data_bench/numerical_reasoning.json')
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["result"]

    output_json_path = "numerical_4o_1.json"
    updated_results = []

    for idx, item in enumerate(data):
        print(f"\n================= Example {idx + 1} =================")

        question = item["question"]
        ground_truth = item["answer"].strip()
        image_files = item["evidence_pages"]
        imagelist = [os.path.join(base_path, img) for img in image_files]
        query_vis = numerical_reasoning_infer_prompt + question

        if "Qwen" in model_path:
            response = llm.generate(query=query_vis, image=imagelist)
        else:
            # 为远程模型调用设置重试次数
            response = safe_generate(llm, query=query_vis, image_list=imagelist)
            
        print("\n*************** Raw Response ***************")
        print(response)

        final_answer = None
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else :
            final_answer = "No final answer found"

        print("\n*************** Processed Response ***************")
        print(final_answer)
        item["model_response"] = response
        updated_results.append(item)
        # 在线评估
        prediction = final_answer
        print("\n*************** Prediction ***************")
        print(prediction)

        print("\n*************** Ground Truth ***************")
        print(ground_truth)

        metric_scores = qa_metric.compute([ground_truth], [prediction])
        print("\n*************** Evaluation Result ***************")
        eval_result = {
            'F1': metric_scores.get('F1', ''),
            'EM': metric_scores.get('EM', ''),
        }
        print(eval_result)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"result": updated_results}, f, indent=4, ensure_ascii=False)

    # 离线评估
    evaluate_all_with_average(output_json_path)
        
if __name__ == '__main__':
    
    qa_metric = QAMetric()
    # main()

    # output_json_path = "numerical_32b.json"
    # evaluate_all_with_average(output_json_path)

    # nohup python -u eval_numerical.py > log_vanilla.txt
    # exp_type = "vanilla_rag"
    # model_type = "Qwen3-VL-8B-Instruct"

    # exp_type = "tooluse_origin"
    # model_type = "Qwen3-VL-8B-Instruct"
# tooluse_Qwen3-VL-8B-SFT-complex_only3_numerical_reasoning
    exp_type = "tooluse" # vanilla_rag, tooluse, gold_rag, tooluse_origin, hieret_rag, hrps_rag
    model_type = "Qwen3-VL-8B-SFT-complex_only3" # Qwen3-VL-8B-SFT-complex, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, gpt-5.1, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, InternVL3_5-8B, InternVL3_5-38B, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, MiniCPM-V-4_5, Llama-3.2-11B-Vision-Instruct


    output_jsonl_path = f"{PROJECT_ROOT}/results/{exp_type}/{model_type}/{exp_type}_{model_type}_numerical_reasoning.jsonl"

    # output_jsonl_path = "<PROJECT_PATH>"
    evaluate_all_with_average2(output_jsonl_path)

    # reference = "37%".replace('&amp;', '&')
    # prediction = "0.37".replace('&amp;', '&')
    # metric_scores = qa_metric.compute([reference], [prediction])
    # eval_result = {
    #     'F1': metric_scores.get('F1', ''),
    #     'EM': metric_scores.get('EM', ''),
    #     # 'ROUGE-L': metric_scores.get('ROUGE-L', ''),
    #     # 'SacreBLEU': metric_scores.get('SacreBLEU', ''),
    #     # 'GPT_EVAL': metric_scores.get('GPT_EVAL', ''),
    #     # 'ECR': metric_scores.get('ECR', ''),
    #     # 'Pass': metric_scores.get('Pass', '')
    # }
    # print(eval_result)
