from gpt_eval_prompt import llm_judge_prompt_fact_verify, llm_judge_prompt_general
from inference.llm_local import VLM
import time
import json
import re

def extract_json_from_response(response):
    """Extract json data from API response"""

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response, re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        return content
    
    return response.strip()

def safe_generate(llm, query, max_retries=5, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(query=query)
            assert 'score' in response and 'rationale' in response
            return response  # 成功直接返回
        except Exception as e:
            print(f"[Retry {attempt}/{max_retries}] Generation failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                print("Maximum retry limit reached. Skipping this sample.")
                return "ERROR: generation failed"

def llm_as_judge_fact_verify(claim, gold_explanation, model_explanation, model="gpt-5.1"):
    llm = VLM(model)
    prompt = llm_judge_prompt_fact_verify.format(
        claim=claim,
        gold_explanation=gold_explanation,
        model_explanation=model_explanation
    )
    response = safe_generate(llm, prompt)
    response = extract_json_from_response(response)
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # JSON 解析失败时返回零分占位结果
        result = {"rationale": "", "score": 0}

    return result


def llm_as_judge_general(question, gold_answer, model_answer, model="gpt-5.1"):
    llm = VLM(model)
    prompt = llm_judge_prompt_general.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer
    )
    response = safe_generate(llm, prompt)
    response = extract_json_from_response(response)
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # JSON 解析失败时返回零分占位结果
        result = {"rationale": "", "score": 0}

    return result