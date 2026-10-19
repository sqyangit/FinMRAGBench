
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import json
from PIL import Image
import math

def convert(prompt, file_name, output_name='cot_think_search_python_data', task_type='knowledge_reasoning'):
    """Convert trajectory histories into multimodal assistant/user messages."""
    with open(file_name, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    sft_results = []
    for item in data:
        query=item['question']
        # reference_answer=item['reference_answer']
        # score=item['eval_result']['score']
        # passing=item['eval_result']['passing']
        # reference_page=item['meta_info']['reference_page']
        
        # 可选的轨迹质量筛选条件
        # if score != 5:
        #     continue
        # if score < 4:
        #     continue
        overall_images = []
        conversations = [{
            "role": "user",
            "content": prompt.replace('{question}', query)
        }]
        good_data = True
        for message in item['history']:
            # import pdb;pdb.set_trace()
            if isinstance(message,dict) and 'think' in message and 'search' in message:
                # import pdb;pdb.set_trace()
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{message["think"]}</think>\n<search>{json.dumps(message["search"], ensure_ascii=False)}</search>'
                })
            elif isinstance(message,dict) and 'think' in message and 'answer' in message:
                if task_type == 'numerical_reasoning' or task_type == 'fact_verify':
                    conversations.append({
                        "role": "assistant",
                        "content": f'<think>{message["think"]}</think>\n<answer>{json.dumps(message["answer"], ensure_ascii=False)}</answer>'
                    })
                else:
                    conversations.append({
                        "role": "assistant",
                        "content": f'<think>{message["think"]}</think>\n<answer>{message["answer"]}</answer>'
                    })
            elif isinstance(message,dict) and 'think' in message and 'python' in message:
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{message["think"]}</think>\n<python>{message["python"]}</python>'
                })
            elif isinstance(message,list):
                base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
                # 过滤出所有含有 image 字段的项
                images = [base_path + item["image"] for item in message if "image" in item]
                # 生成对应数量的 <image> 标签，例如 <image><image><image>
                image_tags = "".join("<image>" for _ in images)

                conversations.append({
                    "role": "user",
                    "content": image_tags
                })
                
                overall_images.extend(images)
            elif isinstance(message,dict) and 'tool_response' in message:
                conversations.append({
                    "role": "user",
                    "content": message['tool_response']
                })
            else:
                continue

        if not good_data:
            continue

        sft_results.append(dict(
            messages=conversations,
            images=overall_images,
        ))
    print(len(sft_results))
    os.makedirs("data", exist_ok=True)
    with open(f"data/{output_name}.json", "w", encoding="utf-8") as f:
        json.dump(sft_results, f, indent=4, ensure_ascii=False)


prompt_tooluse = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. You can search as many times as your want. If you need to perform calculations, you can call the python interpreter by <python> code </python> and user will return the execution result. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. {specific_prompt} Question: {question}'''

specific_prompt_numerical = """For numerical reasoning tasks, your final answer inside <answer> MUST be a valid JSON array of strings. The number of items in the array MUST match the number of required outputs for the question. Each string MUST be the final value only (a number or entity name), as short as possible, without any explanation. Keep percentage signs. If a final value has decimals, retain two decimal places. You MUST NOT skip or omit search before answering.
"""

specific_prompt_table = """For table generation tasks, Your final answer inside <answer> MUST be formatted as a valid Markdown table, with no additional explanations, comments, or text outside the table. You MUST NOT skip or omit search before answering.
"""
specific_prompt_chart = """For chart generation tasks, The plotting code inside <answer> MUST follow the specified template: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show(). You MUST NOT skip or omit search before answering.
"""
specific_prompt_fact = """For fact verify with explanation tasks, your final answer inside <answer> MUST be a valid JSON object with the following structure:

{
    "answer": "Supported / Refuted / Insufficient",
    "explanation": "A concise explanation referencing the retrieved evidence and why it supports, refutes, or fails to resolve the claim."
}

The answer field MUST contain exactly one of: Supported, Refuted, or Insufficient. The explanation MUST be concise and based solely on retrieved evidence. You MUST NOT skip or omit search before answering.
"""

specific_prompt_knowledge = """For knowledge reasoning tasks, your final answer inside <answer> MUST be a concise but complete response in one or two short paragraphs, clearly combining reasoning and conclusion without unnecessary elaboration. You MUST NOT skip or omit search before answering.
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert filtered JSONL trajectories to SFT messages")
    parser.add_argument("--input", required=True, help="Filtered trajectory JSONL file")
    parser.add_argument("--output-name", default="cot_data", help="Writes data/<output-name>.json")
    parser.add_argument("--task", default="knowledge_reasoning", choices=["numerical_reasoning", "table_generation", "chart_generation", "fact_verify", "knowledge_reasoning"])
    args = parser.parse_args()
    specific_prompt = {"numerical_reasoning": specific_prompt_numerical,
                       "table_generation": specific_prompt_table,
                       "chart_generation": specific_prompt_chart,
                       "fact_verify": specific_prompt_fact,
                       "knowledge_reasoning": specific_prompt_knowledge}[args.task]
    prompt = prompt_tooluse.replace('{specific_prompt}', specific_prompt)
    convert(prompt, args.input, args.output_name, task_type=args.task)
