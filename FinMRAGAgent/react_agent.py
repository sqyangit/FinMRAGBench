
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import base64
import json
import re
import requests
import math
from io import BytesIO

from openai import OpenAI
from PIL import Image

from transformers import AutoModel

import sys
sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))
sys.path.append(os.path.join(PROJECT_ROOT, 'auto_synthesis'))
from llm_local import VLM
from stock_map import recheck_prompt




# prompt_tooluse = '''Answer the given question. You must conduct reasoning inside <thinking> and </thinking> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. {specific_prompt} Question: {question}'''


prompt_tooluse = '''Answer the given question. You must conduct reasoning inside <thinking> and </thinking> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. You can search as many times as your want. 

Workflow:
1) Always write <thinking>...</thinking>.
2) Then output either:
   - <search>{...json...}</search> (ONLY ONE per turn), OR
   - <answer>...</answer>.
3) Do NOT output both <search> and <answer>.

IMPORTANT SEARCH FORMAT:
When you use <search>, the content inside MUST be a single valid JSON object in the following schema:
{{
  "query": "A clear, atomic, and retrieval-ready search query describing exactly one company, one year, and one simple financial metric.",
  "industry": "The company's primary industry sector.",
  "company_name": "The full official name of the company.",
  "company_ticker": "The company's stock ticker symbol.",
  "year": "The specific fiscal year for the metric."
}}
Rules:
- Exactly ONE company, ONE fiscal year, and ONE simple metric per <search>.
- The JSON must be directly parseable (no markdown fences, no extra text).
- If multiple years/metrics are needed, issue multiple <search> calls.

If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. {specific_prompt} Question: {question}'''

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



def extract_json_from_response(response):
    """Extract json data from API response"""

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response, re.IGNORECASE)

    if match:
        content = match.group(1).strip()
        return content

    return response.strip()

class ReactAgent:
    def __init__(self, 
                base_url='http://127.0.0.1:8001/v1', 
                search_url='http://127.0.0.1:8111/search',
                check_url = 'http://127.0.0.1:8001/v1',
                generator=True,
                task_type = "table_generation",
                api_key=None,
                model_name=None,
                image_dir=None,
                check_model_name="Qwen/Qwen3-VL-8B-Instruct",
                max_steps=10):
        print("ReactAgent: "+base_url)
        print("search_url: "+search_url)
        print("check_url: "+check_url)
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        self.search_url = search_url

        self.model_name = model_name or 'Qwen/Qwen3-VL-8B-Instruct'
        self.task_type = "fact_verify" if task_type == "explainable_fact_verification" else task_type
        self.check_url = check_url
        self.check_model_name = check_model_name
        self.image_dir = os.fspath(image_dir) if image_dir is not None else os.path.join(
            PROJECT_ROOT, 'FinMRAGAgent', 'data', 'FinanceReport', 'img',
        )
        self.max_pixels = 512 * 28 * 28 * 2
        self.min_pixels = 256 * 28 * 28
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        self.max_steps = max_steps

        self.generator = generator  # Legacy option; run() returns a result directly.
        self.reranker = AutoModel.from_pretrained(
            os.environ.get('FINMRAG_RERANK_MODEL', 'jinaai/jina-reranker-m0'),
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to("cuda").eval()
        self.is_rerank = True

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image/jpeg;base64,{base64_string}"

        return image, base64_qwen

    def rerank(self, query, search_result, topk=10):
        image_list = []
        base_path = self.image_dir + os.sep
        nodes = search_result["source_nodes"]
        if not nodes:
            return []

        # 取前 300 个候选文档
        for node in nodes[:300]:
            filename = node["node"]["metadata"]["file_name"]
            image_list.append(base_path + filename)

        image_pairs = [[query, doc] for doc in image_list]
        scores = self.reranker.compute_score(image_pairs, max_length=2048, doc_type="image")

        if len(scores) != len(image_pairs):
            raise ValueError("Reranker returned a different number of scores than candidate pages")

        # 对节点按得分降序排序
        # enumerate(scores) -> [(idx, score), ...]
        # key=lambda x: x[1] 表示按得分排序
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # 取前 topk 个节点
        topk_indices = [i for i, _ in sorted_indices[:topk]]
        reranked_nodes = [nodes[i] for i in topk_indices]

        # 将重排分数写入 metadata
        for i, idx in enumerate(topk_indices):
            reranked_nodes[i]["node"]["metadata"]["rerank_score"] = float(scores[idx])

        print(f"Rerank 完成，返回 Top-{topk} 文档，共{len(reranked_nodes)}个文档")

        # if len(reranked_nodes)<50:
        #     import pdb;pdb.set_trace()

        return reranked_nodes
    
    def query_recheck(self, nodes, subquery):
        import os
        base_url = self.check_url
        base_path = self.image_dir + os.sep
        llm = VLM(self.check_model_name, base_url)
        # llm = VLM("gpt-4o")

        rechecked_nodes = []
        for node in nodes:
            # 拼文件路径
            file_name = node["node"]["metadata"]["file_name"]
            img_path = os.path.join(base_path, file_name)

            # 生成 prompt
            prompt = recheck_prompt.format(
                subquery=subquery
            ).strip()


            # 模型推理
            result = llm.generate(query=prompt, image=img_path)
            # print(result)


            # 尝试解析模型输出
            try:
                json_str = extract_json_from_response(result)
                data = json.loads(json_str)
                score = float(data.get("score", 0))
            except (ValueError, TypeError, AttributeError):
                # 跳过无法解析为 JSON 的输出
                print("Invalid query_recheck JSON output")
                continue

            # 保留有用的节点
            # 使用 score 筛选，>=7 视为有用
            if score >= 7:
                # 可选：记录 rationale 方便追踪
                # node["node"]["metadata"]["recheck_rationale"] = data.get("rationale", "")
                # node["node"]["metadata"]["recheck_score"] = data.get("score", 0)
                rechecked_nodes.append(node)

        # 输出保留的数量
        print(f"[query_recheck] Kept {len(rechecked_nodes)} of {len(nodes)} nodes")

        return rechecked_nodes
    

    def search(self, query, port=8111, metadata_filters=None):
        """Query search_url; port is retained for compatibility with older callers."""
        url = self.search_url
        data = {"query": query, "metadata_filters": metadata_filters}
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        return response.json()

    def run(self, question, return_history: bool = False):
        """Return (answer, evidence_pages), optionally followed by the step history.

        max_steps limits tool calls. One final model call may produce an answer
        after the tool budget is exhausted. The instance is used sequentially.
        """
        if self.max_steps < 0:
            raise ValueError("max_steps must be nonnegative")

        all_images = []
        self.image_raw = []
        self.image_input = []
        self.image_path = []
        task_type = self.task_type

        # 初始化当前问题的推理历史
        history = []
        history.append({"query": question})
        self.last_history = history

        if task_type=="numerical_reasoning":
            specific_prompt = specific_prompt_numerical
        elif task_type=="table_generation":
            specific_prompt = specific_prompt_table
        elif task_type=="chart_generation":
            specific_prompt = specific_prompt_chart
        elif task_type=="fact_verify":
            specific_prompt = specific_prompt_fact
        elif task_type=="knowledge_reasoning":
            specific_prompt = specific_prompt_knowledge
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        prompt_ins = prompt_tooluse.replace('{specific_prompt}', specific_prompt)

        prompt = prompt_ins.replace('{question}', question)
        messages = [dict(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        )]

        for step in range(self.max_steps + 1):
            ## assistant
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
            )
            if not response.choices:
                raise ValueError("Generation returned no choices")
            response_content = response.choices[0].message.content
            if not isinstance(response_content, str) or not response_content.strip():
                raise ValueError("Generation returned an empty or non-text response")

            print("response_content: ", response_content)

            messages.append(dict(
                role="assistant",
                content=[{
                    "type": "text",
                    "text": response_content
                }]
            ))

            ## think
            pattern = r'<thinking>(.*?)</thinking>'
            match = re.search(pattern, response_content, re.DOTALL)
            thought = match.group(1) if match else ""  # Some models omit the reasoning tag.
            # Only parse actions after the reasoning block; examples inside it are not tool calls.
            action_text = response_content[match.end():] if match else response_content
            ## action
            pattern = r'<(search|answer)>(.*?)</\1>'
            match = re.search(pattern, action_text, re.DOTALL)
            if match:
                content = match.group(2).strip()  # Return only the content inside the tags
                action = match.group(1)
            else:
                content = ''
                raise ValueError("Model response has no valid action tag")

            # 记录本轮推理内容和动作
            assistant_step_item = {
                "role": "assistant",
                "think": thought,
                "action": action,
                "action_content": content,
                "raw_response": response_content,
            }
            history.append(assistant_step_item)

            ## whether end
            if action == "answer":
                if not content:
                    raise ValueError("Model returned an empty answer")
                # 单独记录最终答案，便于从历史中提取
                history.append(
                    {
                        "final_answer": content,
                    }
                )
                # 按 return_history 参数决定是否返回推理历史
                if return_history:
                    self.last_history = history
                    return content, all_images, history
                else:
                    self.last_history = history
                    return content, all_images
                    
            elif step == self.max_steps:
                fallback_answer = "Sorry, I can not retrieval something about the question."
                history.append(
                    {
                        "final_answer": fallback_answer,
                        "reason": "max_steps_exceeded",
                    }
                )
                if return_history:
                    self.last_history = history
                    return fallback_answer, all_images, history
                else:
                    self.last_history = history
                    return fallback_answer, all_images


            ## action
            if action == 'search':
            
                try:
                    search_args = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError("Search action must contain a JSON object") from exc
                if not isinstance(search_args, dict):
                    raise ValueError("Search action must contain a JSON object")
                search_query = search_args.get("query")
                if not isinstance(search_query, str) or not search_query.strip():
                    raise ValueError("Search action requires a nonempty query string")
                ticker = search_args.get("company_ticker")
                year = search_args.get("year")
                metadata_filters = None
                if ticker and year is not None:
                    metadata_filters = {"file_name": f"{ticker}_10K_{year}"}

                # 按 metadata 过滤，并重排取前 50 个候选页面
                print(metadata_filters)
                search_result = self.search(query=search_query, metadata_filters=metadata_filters)

                # 处理指定公司或年份未命中报告的情况
                if not search_result["source_nodes"] and metadata_filters:
                    print("不存在{str}的文件".format(str=(metadata_filters or {}).get("file_name", "unfiltered corpus")))
                    # 未命中时保留公司过滤，移除年份限制
                    weak_metadata_filters = {"file_name": str(ticker or "") + "_10K_"}
                    search_result = self.search(query=search_query, metadata_filters=weak_metadata_filters)  
                    # 仍未命中时移除 metadata 过滤，在完整语料中检索
                    if not search_result["source_nodes"]:
                        print("不存在{str}的文件".format(str=weak_metadata_filters["file_name"]))
                        search_result = self.search(query=search_query)
                reranked_nodes = self.rerank(search_query, search_result, topk=50)
                # 分批复核候选页面，最多向模型提供 4 页证据
                rechecked_nodes = []
                for i in range(0, len(reranked_nodes), 10):
                    batch = reranked_nodes[i:i+10]
                    batch_checked = self.query_recheck(batch, search_query)
                    rechecked_nodes.extend(batch_checked)
                    # 达到证据页数要求后停止检查当前子查询
                    if len(rechecked_nodes) >= 4: # 每次检索保留 4 页证据，与训练设置一致
                        print(f"[query_recheck] 已筛选出 {len(rechecked_nodes)} 个节点，提前结束子查询")
                        break

                search_result_json = rechecked_nodes[0:4]
                image_path_list = [item["node"]["metadata"]["file_name"] for item in search_result_json[0:4]]


                
                image_input = [img for img in dict.fromkeys(image_path_list) if img not in all_images]
                image_path_list = image_input
                base_path = self.image_dir
                image_path_list_input = [os.path.join(base_path, image) for image in image_path_list]

                # 读取
                user_content = []  # 用来存放所有图片的输入
                for image_path in image_path_list_input[0:4]: # 每次检索最多向模型提供 4 张图片
                    with Image.open(image_path) as image_file:
                        image_raw = image_file.copy()
                    image_input_, img_base64 = self.process_image(image_raw)
                    # 添加到 user_content（每张图片一个字典）
                    user_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': img_base64

                        }
                    })

                    self.image_raw.append(image_raw)
                    self.image_input.append(image_input_)
                    self.image_path.append(image_path)
                # 无新增证据图片时返回文本提示
                if(len(image_path_list_input)==0):
                    print("[search] No new evidence images found.")
                    user_content.append({
                        'type': 'text',
                        'text': 'No images were found.' 
                    })

                history.append(
                    {
                        "role": "tool",
                        "tool": "search",
                        "search_query": search_query,
                        "metadata_filters": metadata_filters,
                        "selected_images": image_path_list,
                    }
                )

                all_images += image_path_list


            if step + 1 == self.max_steps:
                user_content.append({
                    'type': 'text',
                    'text': 'please answer the question now with answer in <answer> ... </answer>' 
                })
            messages.append(dict(
                role='user',
                content=user_content
            ))


if __name__ == '__main__':

    
    agent = ReactAgent(task_type = "table_generation")
    query = "Extract the Advanced Micro Devices’s [net income, total assets, total liabilities, and total stockholders’ equity] for 2022 and 2023. Display the data in a markdown table. Output the results in a markdown-formatted table using ‘Item’, ‘2022’, and ‘2023’ as column headers."
    # query = "JPMorgan Chase’s textual disclosures for 2019 and 2020 both highlight management judgments on allowance for credit losses but differ in that the 2020 disclosures emphasize CECL adoption and pandemic-related uncertainty."
    # generator = agent.run(query)
    # while True:
    #     print(next(generator))
    answer, images, history = agent.run(query, return_history=True)

    print("最终答案:", answer)
    print("用到的图片:", images)
    # print("对话 history 样例：")
    # for h in history:
    #     print(h)

    # 组织一个 sample，写入 test.jsonl
    sample = {
        "question": query,
        "response": answer,
        "history": history,
        "images": images,
    }

    # 单样本示例使用覆盖写入；追加样本时使用模式 "a"
    with open("test.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("已写入 test.jsonl")
