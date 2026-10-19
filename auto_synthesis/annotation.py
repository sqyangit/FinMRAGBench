
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import sys
import json
from typing import List, Any
import time
from tqdm import tqdm

sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))
sys.path.append(os.path.join(PROJECT_ROOT, 'auto_synthesis/llm_utils'))
# from llm_local import VLM
from llm_annotation import VLM
from llm_vote import llm_majority_vote
from inference_prompt import numerical_reasoning_infer_prompt, rationale_prompt

def safe_generate(llm, query, image_list, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query, image=image_list, max_new_tokens=8192)
            # response = [item["message"]["content"] for item in response]# 旧接口
            response = response 

            # import pdb;pdb.set_trace()
            # 任务输出格式检查
            # assert 'rewritten_flag' in response and 'new_question' in response
            # API 错误响应检查
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

def get_final_answer(response):
    final_answer = None
    if "[Final Answer]:" in response:
        final_answer = response.split("[Final Answer]:")[-1].strip()
    elif "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[-1].strip()
    else :
        final_answer = "No final answer found"
    return final_answer

def normalize_flag(v: Any) -> str:
    if isinstance(v, (int, float, bool)):
        return str(v).lower()
    return str(v).strip().lower()

def choose_question(item: dict) -> str:
    flag = normalize_flag(item.get("rewritten_flag", ""))
    if flag in ("1", "true", "yes", "y"):
        return item.get("new_question") or item.get("question", "")
    return item.get("question", "")

def build_prompt(question: str) -> str:

    SYSTEM_INSTRUCTION = numerical_reasoning_infer_prompt
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
    )



def build_prompt2(question: str, image_list) -> str:
    # 使用 rationale_prompt 生成带来源信息的标注提示词
    SYSTEM_INSTRUCTION = rationale_prompt

    # 从 image_list 提取 source_name，如 "MCD_10K_56.jpg" -> "MCD_10K"
    sources_text = ""
    for idx, img in enumerate(image_list, start=1):
        base = img.split(".")[0]           # 去掉 .jpg / .png
        source_name = "_".join(base.split("_")[:-1])  # 去掉最后的页码部分
        sources_text += f"Image [{idx}]: {source_name}, "
    # 去掉最后一个多余的逗号+空格
    sources_text = sources_text.rstrip(", ")

    # 注入 prompt 占位符
    filled_prompt = SYSTEM_INSTRUCTION.format(
        question=question,
        source_name=sources_text    # 填入 source_name 占位符
    )
    # print(f"[Prompt] {filled_prompt}")
    # import pdb;pdb.set_trace()

    return filled_prompt


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
        


def annotate_answers(input_path, output_path):
    data = load_json(input_path)
    data = data.get("result", [])
    if not isinstance(data, list):
        raise ValueError("`result` 字段应为列表")

    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
    model_name = "gpt-5.1"
    llm = VLM(model_name)
    labeled = []

    # 读取已保存样本 ID，用于断点续跑
    if os.path.exists(output_path):
        results = []
        with open(output_path, "r") as f:
            for line in f:
                results.append(json.loads(line.strip()))
        uid_already = [item["id"] for item in results]
        data = [item for item in data if item["id"] not in uid_already]

    for i, item in enumerate(tqdm(data, desc="Annotating", unit="sample"), 1):
        # 根据 rewritten_flag 选择原始问题或改写问题
        used_question = choose_question(item).strip()
        # 证据图像：优先 evidence_pages；若无则回退 select_pages
        meta = item.get("synthesis_metadata") or {}
        evidence_pages = item.get("evidence_pages") or item.get("select_pages") or meta.get("select_pages")

        if not evidence_pages:
            print(f"[{i}] 警告：未找到 evidence_pages")
            continue
            evidence_pages = item.get("select_pages", [])
            print(f"[{i}] 警告：未找到 evidence_pages，使用 select_pages")

        # 直接用黄金文档
        images = [base_path + filename for filename in evidence_pages]
        prompt = build_prompt(used_question)

        # 可选：使用 select_pages 构造带推理说明的标注提示词
        # evidence_pages = meta.get("select_pages", [])
        # images = [base_path + filename for filename in evidence_pages]
        # prompt = build_prompt2(used_question, evidence_pages)

        try:
            image_arg = images if images else ''
            answer_text = safe_generate(llm, query=prompt, image_list=image_arg)
        except Exception as e:
            answer_text = f"[GenerationError] {e}"

        # safe_generate 返回多个候选答案，失败时返回错误字符串
        if isinstance(answer_text, list):
            samples_raw = answer_text
        else:
            # 失败或单样本情况，统一成列表
            samples_raw = [answer_text]

        samples_labeled = [get_final_answer(resp if isinstance(resp, str) else str(resp)) for resp in samples_raw]
        answer_final, vote_number, vote_rationale = llm_majority_vote(used_question, samples_labeled)
        # answer_labeled = get_final_answer(answer_text)
        # 记录标注结果（不破坏原字段）
        out_item = dict(item)  # 浅拷贝，保留原字段
        out_item["question"] = used_question # 保存当前选用的原始问题或改写问题
        # out_item["answer_raw"] = samples_raw
        # out_item["answer_labeled"] = samples_labeled
        # out_item["vote_number"] = vote_number
        # out_item["vote_rationale"] = vote_rationale
        # out_item["answer_final"] = answer_final
        # out_item["model"] = model_name
        out_item["annotation_metadata"] = {
            "answer_raw": samples_raw,
            "answer_labeled": samples_labeled,
            "vote_number": vote_number,
            "vote_rationale": vote_rationale,
            "answer_final": answer_final,
            "model": model_name,
        }


        # 逐条追加写入
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

        labeled.append(out_item)# 保留内存副本；结果已逐条写入文件

        print(f"[{i}] 完成：{used_question[:80]}{'...' if len(used_question)>80 else ''}")

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump({"result": labeled}, f, ensure_ascii=False, indent=4)

    print(f"\n标注完成，写入：{output_path}")

def finalize_output(path_in, path_out=None):
    """将逐行 JSON 文件封装为标准 JSON 格式 { "result": [...] }"""
    if path_out is None:
        path_out = path_in.replace(".json", "_final.json")

    with open(path_in, "r", encoding="utf-8") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump({"result": items}, f, ensure_ascii=False, indent=4)

    print(f"封装完成：{path_out} （共 {len(items)} 条）")


def finalize_output2(path_in, path_out=None):
    """将逐行 JSON 封装为标准 JSON 格式，并只保留指定字段"""
    if path_out is None:
        path_out = path_in.replace(".json", "_final.json")

    result_items = []

    with open(path_in, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # 取 annotation_metadata.answer_final，没有则给 None
            answer_final = None
            if isinstance(item.get("annotation_metadata"), dict):
                answer_final = item["annotation_metadata"].get("answer_final")

            processed = {
                "id": item.get("id"),
                "question": item.get("question"),
                "evidence_pages": item.get("evidence_pages", []),
                "answer": answer_final
            }

            result_items.append(processed)

    # 写回
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump({"result": result_items}, f, ensure_ascii=False, indent=4)

    print(f"封装完成：{path_out} （共 {len(result_items)} 条）")

if __name__ == "__main__":
    # questions_rewritten_path = "questions_rewritten_final.json"
    questions_rewritten_path = "questions_rewritten_part1.json"
    questions_labeled_path = "questions_rationale_labeled_merged.json"
    # annotate_answers(questions_rewritten_path, questions_labeled_path)
    finalize_output(questions_labeled_path)

    # questions_rewritten_path = "<PROJECT_PATH>"
    # questions_labeled_path = "<PROJECT_PATH>"
    # # annotate_answers(questions_rewritten_path, questions_labeled_path)
    # finalize_output2(questions_labeled_path)