
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

import json
import sys
import time
import re

sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))
from llm_local import VLM

def safe_generate(llm, query, image_list, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query, image=image_list, max_new_tokens=8192)
            # 任务输出格式检查
            assert 'number' in response and 'rationale' in response
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

def extract_json_from_response(response):
    """Extract json data from API response"""

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response, re.IGNORECASE)

    if match:
        content = match.group(1).strip()
        return content

    return response.strip()


def llm_majority_vote(question: str, candidates: list[str], max_retries=3):
    """
    使用大模型进行“多数投票”，选择最优答案。
    输入：
        - llm: VLM 实例（如 VLM("gpt-5.1")）
        - question: 当前问题文本
        - candidates: 候选答案列表（字符串列表）
    返回：
        - chosen_answer: 由大模型选择出的最佳答案文本
    """
    if not candidates:
        return "No final answer found", 0, "No candidate answers"
    llm = VLM("gpt-5.1")
    # --- 构造 prompt---
    numbered = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(candidates))
    # import pdb;pdb.set_trace()
    vote_prompt = """
You are a careful and impartial judge.
Given a question and several candidate answers,
choose the single answer that is most correct, well-reasoned, and factually accurate.

Question:
{question}

Candidate Answers:
{numbered}

Respond with STRICT JSON only, with two fields:
```json 
{{
  "number": <INTEGER index of the best answer starting from 1>,
  "rationale": "<brief explanation of why this answer is best>"
}}
```
Do not include any extra text outside the JSON.
""".format(
    question=question,
    numbered=numbered
).strip()

    # --- 调用模型投票 ---
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = safe_generate(llm, query=vote_prompt, image_list='')
            # safe_generate 返回 list[str]
            if isinstance(response, list) and response:
                text = response[0]
            else:
                text = str(response)

            json_str = extract_json_from_response(text)
            data = json.loads(json_str)

            raw_num = data.get("number", None)
            rationale = data.get("rationale", "")

            if isinstance(raw_num, str):
                # 只取第一个数字
                m = re.search(r"\b([1-9]\d*)\b", raw_num)
                number = int(m.group(1)) if m else None
            elif isinstance(raw_num, (int, float)):
                number = int(raw_num)
            else:
                number = None

            if number is not None and 1 <= number <= len(candidates):
                answer_final = candidates[number - 1]
                return answer_final, number, rationale

            print(f"[Vote {attempt}] 非法编号: {raw_num}（候选数={len(candidates)}）")
        except Exception as e:
            print(f"[Vote Retry {attempt}/{max_retries}] 投票失败：{e}")

    # --- 兜底策略：若模型输出无效，取众数或第一个 ---
    from collections import Counter
    counts = Counter(candidates)
    max_count = max(counts.values())
    ties = [c for c, n in counts.items() if n == max_count]
    answer = ties[0]
    return answer, candidates.index(answer) + 1, "Fallback: most frequent candidate"