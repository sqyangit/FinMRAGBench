
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
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

from gpt_eval import llm_as_judge_fact_verify

def extract_json_from_response(response):
    """Extract json data from API response"""

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response, re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        return content
    
    return response.strip()

def safe_generate(llm, query, image_list, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query, image=image_list)
            assert 'answer' in response and 'explanation' in response
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

    metrics = []   # 存分数
    metrics2 = []  # 存是否 >=7 的布尔值
    updated_results = []  # 存更新后的结果
    for idx, item in enumerate(data):
        answer = item.get("answer", "")
        explanation = item.get("explanation", "")
        model_answer = item.get("model_answer", "")
        model_explanation = item.get("model_explanation", "")

        # if answer.strip().lower() != model_answer.strip().lower():
        #     score = 0
        #     passing = 0
        #     metrics.append(score)
        #     metrics2.append(passing)
        #     item["rationale"] = "Answer mismatch"
        #     item["score"] = score
        #     item["passing"] = passing
        #     updated_results.append(item)
        #     print(f"\n[Sample {idx + 1}] Evaluation: Skipped (answer mismatch)")
        #     continue
        
        result = llm_as_judge_fact_verify(item["question"], explanation, model_explanation)

        try:
            score = int(float(result.get("score", 0)))  
        except (ValueError, TypeError):
            score = 0

        passing = 1 if score >= 7 else 0
        rationale = result.get("rationale") or ""

        item["score"] = score
        item["passing"] = passing
        item["rationale"] = rationale
        metrics.append(score)
        metrics2.append(passing)
        updated_results.append(item)

        print(f"\n[Sample {idx + 1}] Evaluation:")
        print(result)
        print("-----------------------------------------")

    # --------- 计算平均分和通过率 ---------
    if metrics:
        avg_score = sum(metrics) / len(metrics)
        pass_rate = sum(metrics2) / len(metrics2)
    else:
        avg_score, pass_rate = 0, 0

    # --------- 生成新文件名 ---------
    base, ext = os.path.splitext(output_json_path)
    new_path = f"{base}_lj{ext}"
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump({"result": updated_results}, f, indent=4, ensure_ascii=False)

    print("\n*************** Summary ***************")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Passing Rate (>=7): {pass_rate:.2%}")

    return {"average_score": avg_score, "passing_rate": pass_rate}


def evaluate_all_with_average_2(output_jsonl_path):
    # ===== 1. 构造 lj 目录 + 新结果文件路径 =====
    base_dir = os.path.dirname(output_jsonl_path)
    filename = os.path.basename(output_jsonl_path)
    base, ext = os.path.splitext(filename)

    lj_dir = os.path.join(base_dir, "lj")
    os.makedirs(lj_dir, exist_ok=True)

    new_path = os.path.join(lj_dir, f"{base}_lj{ext}")

    # ===== 2. 先读已有 lj 结果：收集已评 id + 累积历史 metrics =====
    uid_already = set()
    metrics = []      # 所有样本（历史 + 新跑）的分数
    metrics2 = []     # 所有样本（历史 + 新跑）是否 >=7

    if os.path.exists(new_path):
        with open(new_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                _id = obj.get("id")
                if _id is not None:
                    uid_already.add(_id)

                # 把历史分数也统计进去
                if "score" in obj:
                    try:
                        s = int(float(obj["score"]))
                    except (ValueError, TypeError):
                        s = 0
                    p = 1 if s >= 7 else 0
                    metrics.append(s)
                    metrics2.append(p)

    # ===== 3. 从原始 JSONL 读数据，并按 id 过滤掉已经评过的 =====
    data = []
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("id") in uid_already:
                continue
            data.append(item)

    print(f"Need to evaluate {len(data)} new samples...")

    # ===== 4. 对“新数据”做评估（边跑边写 + tqdm） =====
    for item in tqdm(data, desc="Evaluating", ncols=100, dynamic_ncols=True):
        tqdm.write(f"Processing id={item.get('id')}")

        answer = item.get("answer", "")
        explanation = item.get("explanation", "")

        # 预处理：从 response 中抽出 model 的 answer / explanation
        response = item.get("response", "") or ""
        # import pdb;pdb.set_trace()
        # prediction_str = extract_json_from_response(response)
        if isinstance(response, (dict, list)): # 将 ViDoRAG 返回的结构化答案序列化为 JSON 字符串
            prediction_str = json.dumps(response, ensure_ascii=False)
        else:
            prediction_str = extract_json_from_response(response)

        try:
            prediction = json.loads(prediction_str)
            item["model_answer"] = prediction.get("answer", "")
            item["model_explanation"] = prediction.get("explanation", "")
        except json.JSONDecodeError as e:
            tqdm.write(f"JSON 解析失败: {e}")
            item["model_answer"] = ""
            # 兼容未按 JSON 格式输出答案的模型响应
            if "MiniCPM" in output_jsonl_path or "origin" in output_jsonl_path or "Thinking" in output_jsonl_path or "vidorag2" in output_jsonl_path:
                item["model_explanation"] = prediction_str
            else :
                item["model_explanation"] =""

        model_answer = item.get("model_answer", "")
        model_explanation = item.get("model_explanation", "")

        # 使用参考解释和模型解释进行事实验证评分
        result = llm_as_judge_fact_verify(item["question"], explanation, model_explanation)

        try:
            score = int(float(result.get("score", 0)))
        except (ValueError, TypeError):
            score = 0

        passing = 1 if score >= 7 else 0
        rationale = result.get("rationale") or ""

        item["score"] = score
        item["passing"] = passing
        item["rationale"] = rationale

        # 新样本分数加入“全局”统计（历史 + 新跑）
        metrics.append(score)
        metrics2.append(passing)

        # 边跑边写：每个 item 立刻 append 到 lj 结果文件
        with open(new_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ===== 5. 汇总历史记录和本次新增样本的评分 =====
    if metrics:
        avg_score = sum(metrics) / len(metrics)
        pass_rate = sum(metrics2) / len(metrics2)
    else:
        avg_score, pass_rate = 0, 0

    print("\n*************** Summary ***************")
    print(f"Average Score (ALL runs): {avg_score:.3f}")
    print(f"Passing Rate (>=7, ALL runs): {pass_rate:.2%}")
    print(f"Results written to: {new_path}")

    return {"average_score": avg_score, "passing_rate": pass_rate}

def main():

    from inference.llm_local import VLM
    from inference.inference_prompt import fact_verify_infer_prompt
    import os

    # model_path = "<PROJECT_PATH>"
    model_path = "o3-2025-04-16" # gpt-5.1, gpt-4.1-2025-04-14, o3-2025-04-16
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    llm = VLM(model_path)
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')

    input_json_path = os.path.join(PROJECT_ROOT, 'data_bench/explainable_fact_verification.json')
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["result"]

    output_json_path = os.path.join(PROJECT_ROOT, 'results/fact_verify.json')
    updated_results = []

    for idx, item in enumerate(data):
        print(f"\n================= Example {idx + 1} =================")

        question = item["question"]
        ground_truth = item["explanation"].strip()
        image_files = item["evidence_pages"]
        imagelist = [os.path.join(base_path, img) for img in image_files]

        query_vis = fact_verify_infer_prompt + f"\n# Retrieved Documents\nEach retrieved image file is named with the company name, year, and page number, as follows: [{', '.join(item['evidence_pages'])}], corresponding one-to-one with the provided images.\nQuestion: " + question
        # query_vis = fact_verify_infer_prompt+question

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
        prediction_str = extract_json_from_response(response)
        try:
            prediction = json.loads(prediction_str)
            item["model_answer"] = prediction["answer"]
            item["model_explanation"] = prediction["explanation"]
        except json.JSONDecodeError as e:
            print("JSON 解析失败:", e)
            prediction = prediction_str
            item["model_answer"] = ""
            item["model_explanation"] = ""


        print("\n*************** Prediction ***************")
        print(prediction)

        print("\n*************** Ground Truth ***************")
        print(ground_truth)

        # result = llm_as_judge_fact_verify(question, ground_truth, prediction)
        # print("\n*************** Evaluation Result ***************")
        # print(result)

    # 确保文件所在目录存在
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"result": updated_results}, f, indent=4, ensure_ascii=False)

    # 离线评估
    evaluate_all_with_average(output_json_path)

if __name__ == '__main__':
    # main()
    # output_json_path = "fact_verify_4o_yuwei4.json"
    # evaluate_all_with_average(output_json_path)

    # output_jsonl_path = "<PROJECT_PATH>"

    exp_type = "vidorag2" # vanilla_rag, tooluse, gold_rag, tooluse_origin, hieret_rag, hrps_rag
    model_type = "gpt-5.1" # Qwen3-VL-8B-SFT-complex, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, gpt-5.1, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, InternVL3_5-8B, InternVL3_5-38B, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, MiniCPM-V-4_5, Llama-3.2-11B-Vision-Instruct

    output_jsonl_path = f"{PROJECT_ROOT}/results/{exp_type}/{model_type}/{exp_type}_{model_type}_fact_verify.jsonl"

    evaluate_all_with_average_2(output_jsonl_path)
    
