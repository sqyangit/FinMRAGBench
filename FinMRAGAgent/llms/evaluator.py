
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
from collections import Counter
import re
import json
import sys 
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm import LLM, LLM2
# from llm import LLM, LLM2

DEFAULT_SYSTEM_TEMPLATE = """System:
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query and reference answer
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.
- When comparing numerical values in the generated answer with the reference answer, allow for a reasonable margin of error (small deviations are acceptable and should not be penalized as mistakes).

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, but it is not as concise.

User:
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

class Evaluator:
    def __init__(self):
        self.llm = LLM("gpt-5.1")
        # self.llm = LLM("<PROJECT_PATH>")
        # self.llm = LLM2("<PROJECT_PATH>")
        self.system_template = DEFAULT_SYSTEM_TEMPLATE
        
    def llm_eval(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        try_times = 10
        while try_times > 0:
            try_times -= 1
            try:
                judge = self.llm.generate(query=system_prompt)
                # import pdb;pdb.set_trace()
                match = re.search(r'\d', judge)
                if match:
                    score =  int(match.group(0))
                    if score >= 4:
                        passing = 1
                    else:
                        passing = 0
                    break
            except Exception as e:
                print(e)
                continue
        else:
            raise RuntimeError("Numerical judge failed after 10 attempts")
        return score, passing, judge
    
    def evaluate(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        score, passing, judge = self.llm_eval(query, reference_answer, generated_answer)
        result = dict(
            score=score,
            passing=passing,
            judge=judge
        )
        return result


    
if __name__ == '__main__':
    evaluator = Evaluator()
    # query = "What is the capital of France?"
    # reference_answer = "The capital of France is Paris."
    # generated_answer = "Paris is the capital of France."
    # score, passing, judge = evaluator.llm_eval(query, reference_answer, generated_answer)
    # print(score, passing, judge)


    total_score = 0
    total_passing = 0
    count = 0

    exp_type = "tooluse" # vanilla_rag, tooluse, gold_rag, tooluse_origin, hieret_rag, hrps_rag
    model_type = "Qwen3-VL-8B-SFT-complex" # Qwen3-VL-8B-SFT-complex, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, gpt-5.1, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, InternVL3_5-8B, InternVL3_5-38B, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, MiniCPM-V-4_5, Llama-3.2-11B-Vision-Instruct, Qwen3-VL-32B-SFT-rebuttal

    file_path = f"{PROJECT_ROOT}/results/{exp_type}/{model_type}/{exp_type}_{model_type}_numerical_reasoning.jsonl"
    output_path = f"{PROJECT_ROOT}/results/{exp_type}/{model_type}/lj/{exp_type}_{model_type}_numerical_reasoning_lj.jsonl"
    # file_path = "<PROJECT_PATH>"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out_f = open(output_path, "w", encoding="utf-8")


    for line in tqdm(lines, desc="Evaluating"):
        data = json.loads(line)
        id = data.get("id")
        query = data.get("question")
        # query = data.get("query")
        reference_answer = data.get("answer")
        # reference_answer = data.get("reference_answer")
        
        generated_answer = data.get("response")
        evidence_pages = data.get("evidence_pages")    
        recall_results = data.get("recall_results")
        recall_pages = [item["node"]["metadata"]["file_name"] for item in recall_results["source_nodes"]]

        if query and reference_answer and generated_answer:
            score, passing, judge = evaluator.llm_eval(query, reference_answer, generated_answer)
            print(score, passing, judge)
            # 逐条写入 JSONL 评测记录
            record = {
                "id": id,
                "question": query,
                "answer": reference_answer,
                "response": generated_answer,
                "evidence_pages": evidence_pages,
                "recall_pages": recall_pages,
                "score": score,
                "passing": passing,
                "judge": judge
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()   # 及时保存记录，减少中断时的数据丢失
            # ==============================
            total_score += score
            total_passing += passing
            count += 1

    if count > 0:
        avg_score = total_score / count
        avg_passing = total_passing / count
        print(f"{exp_type}_{model_type}")
        print(f"平均 Score: {avg_score}")
        print(f"平均 Passing: {avg_passing}")
    else:
        print("没有有效数据")
