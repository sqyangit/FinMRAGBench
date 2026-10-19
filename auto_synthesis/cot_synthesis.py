
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import json
from tqdm import tqdm

from io import BytesIO
from openai import OpenAI
from PIL import Image, ImageDraw
import base64
import math

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.schema import NodeWithScore, ImageNode
import sys
import requests
import time

from transformers import AutoModel
sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))
from stock_map import decomposition_prompt, recheck_prompt, extra_metadata_prompt, final_judge_prompt ,ticker_to_name_semiconductors, finance_metrics, industry_to_dict

raw_image_dir = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img')
search_engine_url = "http://127.0.0.1:8111/search"
search_url = search_engine_url

from cot_prompt import numerical_cot_prompt, numerical_cot_prompt_v2, numerical_cot_prompt_v3, chart_cot_promt, chart_cot_promt_v2, table_cot_pormpt, fact_cot_pormpt, fact_cot_pormpt_v2, knowledge_cot_pormpt

# 默认任务提示词；命令行入口根据 task_type 覆盖
prompt_inst = table_cot_pormpt  # Overridden by task_type in the CLI below.

# specific_answer  = """
# """
# prompt_inst = prompt_inst + specific_answer

prompt_user_start = """Question: {question}"""


def extract_json(response):
    response = response.replace("```json", "").replace("```", "")
    response = response.replace("```\n", "").replace("\n```", "")
    response_json = json.loads(response)
    return response_json

def extract_json_from_response(response):
    """Extract json data from API response"""

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response, re.IGNORECASE)

    if match:
        content = match.group(1).strip()
        return content

    return response.strip()

# 行业语料检索
def search(query, ip="127.0.0.1", port=8002, metadata_filters=None):
    url = search_url  # CLI host and port are assigned below before collection.
    data = {"query": query, "metadata_filters": metadata_filters}
    response = requests.post(url, json=data, timeout=120)
    response.raise_for_status()
    return response.json()

def rerank(query, search_result, topk=10):
    image_list = []
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
    nodes = search_result["source_nodes"]
    if not nodes:
        return []

    # 取前 300 个候选文档
    for node in nodes[:300]:
        filename = node["node"]["metadata"]["file_name"]
        image_list.append(base_path + filename)

    image_pairs = [[query, doc] for doc in image_list]
    scores = reranker.compute_score(image_pairs, max_length=2048, doc_type="image")

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
        # import pdb;pdb.set_trace()

    return reranked_nodes

from llm_local import VLM
def query_recheck(nodes, subquery):
    import os
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
    llm = VLM("Qwen/Qwen3-VL-8B-Instruct", base_url)
    # llm = VLM("gpt-5.1")

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
        # import pdb;pdb.set_trace()

        # 尝试解析模型输出
        try:
            json_str = extract_json_from_response(result)
            data = json.loads(json_str)
        except Exception:
            # 跳过无法解析为 JSON 的输出
            print("Invalid query_recheck JSON output")
            continue

        # 保留有用的节点
        # 使用 score 筛选，>=7 视为有用
        if data.get("score", 0) >= 7:
            # 可选：记录 rationale 方便追踪
            # node["node"]["metadata"]["recheck_rationale"] = data.get("rationale", "")
            # node["node"]["metadata"]["recheck_score"] = data.get("score", 0)
            rechecked_nodes.append(node)

    # 输出保留的数量
    print(f"[query_recheck] Kept {len(rechecked_nodes)} of {len(nodes)} nodes")

    return rechecked_nodes


class Model_Role:
    def __init__(self, model_name):
        self.model = model_name or "gpt-5.1"
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""), # gpt
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

    def extract_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join([c["text"] for c in content if c.get("type") == "text"])
        return ""

    def save_usage_log(self, content, completion, max_chars=100):
        model_name = self.model
        log_dir = os.path.join(PROJECT_ROOT, 'results/api_use')

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        log_input = self.extract_text(content)
        log_output = completion.choices[0].message.content

        short_input = log_input[:max_chars]
        short_output = log_output[:max_chars]

        print(model_name + "/Prompt tokens:", input_tokens)
        print(model_name + "/Completion tokens:", output_tokens)
        print(model_name + "/Total tokens:", total_tokens)

        record = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/{model_name}_used.jsonl"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def generate(self, messages):
        # 同步调用模型接口
        return self.generate_vlm(messages)

    def generate_llm(self, messages):
        # 兼容旧接口，转到统一实现
        return self.generate_vlm(messages)

    def generate_vlm(self, messages):
        retries = 5
        while retries > 0:
            try:
                # resp = chatgpt_api(
                #     prompt=messages, model=self.model, asyn_request=False, logging=False
                # )
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # reasoning={"effort": "none"} # 仅在服务支持 reasoning 参数时启用
                    # temperature=1,
                    # top_p=0.95
                )
                # 将空响应或接口异常交由重试逻辑处理
                # assert 'think' in response
                assert resp.choices and resp.choices[0].message.content
                if not resp:
                    raise AssertionError

                if os.environ.get("FINMRAG_LOG_USAGE") == "1":
                    self.save_usage_log(messages[0]["content"], resp)


                return resp.choices[0].message.content
            except Exception as e:
                print(e)
                retries -= 1
                time.sleep(5)
        return None


class MMRAG:
    def __init__(
        self,
        dataset="search_engine/coupus",
        query_file="rag_dataset.json",
        experiment_type="cot",
        workers_num=1,
        topk=10,
        output_file_name="output.jsonl",
    ):
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.query_file = query_file
        # self.dataset_dir = os.path.join('./', dataset)
        self.dataset_dir = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport')
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # retrieval only
        if experiment_type == "cot":
            self.eval_func = self.cot_collect
            self.output_file_name = output_file_name

        self.output_file_path = os.path.join(
            self.results_dir, self.output_file_name.replace("/", "-")
        )

        self.max_pixels = 512 * 28 * 28 * 2
        self.min_pixels = 256 * 28 * 28

        self.llm = Model_Role(model_name="gpt-5.1")
        self.vlm = Model_Role(model_name="gpt-5.1")

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

    def cot_collect(self, sample):
        query = sample["question"]
        print("query: " + query)
        # reference_images = [
        #     f"{raw_image_dir}/" + evidence_page
        #     for evidence_page in sample["evidence_pages"]
        # ]
        # reference_answer = sample["answer"]
        all_images = []
        history = [{"query": query}]
        messages = []
        # messages.append({
        #     "role": "system",
        #     "content": [
        #         {"text": prompt_inst}
        #     ]
        # })
        messages.append(
            dict(
                role="system",
                content=[
                    {
                        "type": "text",
                        "text": prompt_inst,
                    }
                ],
            )
        )
        # messages.append({
        #     "role": "user",
        #     "content": [
        #         {"text": prompt_user_start.replace('{question}', query)}
        #     ]
        # })
        messages.append(
            dict(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": prompt_user_start.replace('{question}', query),
                    }
                ],
            )
        )        

        try_times = 6 # 工具交互轮数预算
        while True:
            try_times -= 1
            if try_times < 0:
                print("无效样本")
                return None
            cnt = 0
            while True:
                try:
                    cnt += 1  # Count API failures too, so retries are bounded.
                    response = self.vlm.generate(messages)
                    if response is None:
                        return None
                    print(response)
                    response_json = extract_json(response)
                    break
                except Exception as e:
                    time.sleep(3)
                    if cnt < 5: # 最多尝试 5 次生成并解析 JSON 响应
                        continue
                    else:
                        # response_json = None
                        return None # 连续失败后丢弃当前轨迹

            # print(response_json)
            # import pdb;pdb.set_trace()



            if "think" in response_json and "answer" in response_json:
                history.append(response_json)
                answer = response_json["answer"]
                break
            elif "think" in response_json and "python" in response_json:
                python_code = response_json["python"]

                # import pdb;pdb.set_trace()
                # Model-generated code runs in-process; use an isolated research environment.
                local_env = {}

                try:
                    # Functions and comprehensions must see the generated imports and variables.
                    exec(python_code, local_env, local_env)

                    # 将末行指定的变量作为执行结果
                    # 约定生成代码的最后一行是结果变量名，例如 roe
                    result = None
                    if python_code.strip().split("\n")[-1] in local_env:
                        last_var = python_code.strip().split("\n")[-1]
                        result = local_env[last_var]
                    else:
                        # 末行不是变量名时，返回执行命名空间中的变量
                        result = str({key: value for key, value in local_env.items() if key != "__builtins__"})
                except Exception as e:
                    result = f"Python execution error: {e}"

                # 记录 assistant 的原始 Python 工具调用响应
                messages.append(
                    dict(
                        role="assistant",
                        content=[
                            {
                                "type": "text",
                                "text": response
                            }
                        ]
                    )
                )
                history.append(response_json)

                # 将 Python 执行结果加入下一轮模型输入
                messages.append(
                    dict(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": f"Here is the python execution result: {result}. Please continue your reasoning."
                            }
                        ]
                    )
                )
                user_response = {
                    "tool_response": f"Here is the python execution result: {result}. Please continue your reasoning."
                }
                history.append(user_response)

            elif "think" in response_json and "search" in response_json:
                # search_query = response_json["search"]
                # queries = [search_query]
                search_json = response_json["search"]
                # search_result = requests.get(search_engine_url, params={"queries": queries})

                # import pdb;pdb.set_trace()
                # 获取metadata
                try:
                    # ========= 旧版本，keyword版本 ==========
                    # keyword_data = response_json.get("keyword", {})  # 防止 keyword 不存在
                    # ticker = keyword_data.get("company_ticker")
                    # year = keyword_data.get("year")

                    # ========= search是json的版本 ===========
                    search_query = search_json.get("query")
                    industry = search_json.get("industry")
                    company_name = search_json.get("company_name")
                    ticker = search_json.get("company_ticker")
                    year = search_json.get("year")

                    # 检查 ticker 和 year 是否有效
                    if ticker is not None and year is not None:
                        # year 如果是 int 就转成 str
                        if isinstance(year, int):
                            year = str(year)
                        elif not isinstance(year, str):
                            # 如果是其他类型，转成字符串避免报错
                            year = str(year)

                        metadata_filters = {"file_name": f"{ticker}_10K_{year}"}
                    else:
                        print("缺少 company_ticker 或 year 字段")
                        metadata_filters = None
                except Exception as e:
                    print(f"有异常: {e}")
                    metadata_filters = None

                # 按 metadata 过滤，并重排取前 50 个候选页面
                print(metadata_filters)
                search_result = search(query=search_query, metadata_filters=metadata_filters)

                # 处理指定公司或年份未命中报告的情况
                if(len(search_result["source_nodes"])==0):
                    print("不存在{str}的文件".format(str=(metadata_filters or {}).get("file_name", "unfiltered corpus")))
                    end_flag = 1
                    return None

                reranked_nodes = rerank(search_query, search_result, topk=50)

                # 分批复核候选页面，最多向模型提供 4 页证据
                rechecked_nodes = []
                for i in range(0, len(reranked_nodes), 10):
                    batch = reranked_nodes[i:i+10]
                    batch_checked = query_recheck(batch, search_query)
                    rechecked_nodes.extend(batch_checked)
                    # 达到证据页数要求后停止检查当前子查询
                    if len(rechecked_nodes) >= 4:
                        print(f"[query_recheck] 已筛选出 {len(rechecked_nodes)} 个节点，提前结束子查询")
                        break

                search_result_json = rechecked_nodes[0:4]
                image_path_list = [item["node"]["metadata"]["file_name"] for item in search_result_json[0:4]]


                # def select_element(A, B, C):
                #     # 检查A中元素是否在B中且不在C中的第一个元素
                #     for a in A:
                #         if a in B and a not in C:
                #             return a
                #     # 如果没有找到这样的元素，则检查A中第一个不在C中的元素
                #     for a in A:
                #         if a not in C:
                #             return a
                #     # 如果所有元素都在C中，则返回None
                #     return None

                # image_input = select_element(
                #     image_path_list, reference_images, all_images
                # )
                # if image_input is None:
                #     return None
                # image_path_list = [image_input]

                # 取出 image_path_list 里有、但 all_images 里没有的元素
                image_input = [img for img in image_path_list if img not in all_images]
                if image_input is None:
                    return None
                image_path_list = image_input
                base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img')
                image_path_list_input = [os.path.join(base_path, image) for image in image_path_list]

                # 读取
                user_content = []  # 用来存放所有图片的输入
                for image_path in image_path_list_input[0:4]: # 每次工具调用最多输入 4 张证据图片
                    image_raw = Image.open(image_path)
                    image_input_, img_base64 = self.process_image(image_raw)
                    # 添加到 user_content（每张图片一个字典）
                    user_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': img_base64
                        }
                    })
                # assistant
                history.append(response_json)
                # messages.append({"role": "assistant", "content": [{"text": response}]})
                messages.append(
                    dict(
                        role="assistant",
                        content=[
                            {
                                "type": "text",
                                "text": response,
                            }
                        ]
                    )
                )
                # user
                images_content = [{"image": image_path} for image_path in image_path_list[:5]]
                images_content += [
                    {
                        "text": "Here are the search results. Please analyze them and provide your next think."
                    }
                ]
                if try_times == 1:
                    user_content.append({
                        'type': 'text',
                        'text': 'Here are all the search results we have gathered. Please analyze them and provide your final answer based on all available results.' 
                    })
                else:
                    user_content.append({
                        'type': 'text',
                        'text': 'Here are the search results. Please analyze them and provide your next think.' 
                    })
                # messages.append({"role": "user", "content": images_content})
                messages.append(
                    dict(
                        role="user",
                        content=user_content, # user_content用来传图像base64，images_content来存储图像路径
                    )
                )
                history.append(images_content)
                all_images += image_path_list


        sample["history"] = history
        # sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        sample["response"] = answer
        print(f"answer: {answer}")

        sample["recall_results"] = dict(
            source_nodes=[
                NodeWithScore(
                    node=ImageNode(image_path=image, metadata=dict(file_name=image)),
                    score=None,
                ).to_dict()
                for image in all_images
            ],
            response=None,
            metadata=None,
        )
        return sample

    def eval_dataset(self):
        eval_func = self.eval_func

        # rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        # rag_dataset_path = "<PROJECT_PATH>"
        rag_dataset_path = os.path.join(os.path.join(PROJECT_ROOT, 'auto_synthesis'),self.query_file)

        # 按任务类型组织输入问题和输出轨迹目录
        rag_dataset_path = f"{PROJECT_ROOT}/auto_synthesis/industry_data/{task_type}/filter/"+self.query_file
        self.output_file_path = f"{PROJECT_ROOT}/auto_synthesis/industry_data/{task_type}/cot/"+self.output_file_name
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        
        with open(rag_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data["result"]

        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    results.append(json.loads(line.strip()))
            uid_already = [item["id"] for item in results]
            data = [item for item in data if item["id"] not in uid_already]

        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None:
                    continue
                with open(self.output_file_path, "a", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                results = []
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing"
                ):
                    result = future.result()
                    results.append(result)
                    if len(results) >= 3:
                        with open(self.output_file_path, "a", encoding="utf-8") as f:
                            for res in results:
                                if res is None:
                                    continue
                                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        results = []
                if results:
                    with open(self.output_file_path, "a", encoding="utf-8") as f:
                        for res in results:
                            if res is None:
                                continue
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        return self.output_file_path


def arg_parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="example", help="The name of dataset")
    parser.add_argument("--query_file",type=str,default="rag_dataset.json",help="The name of anno_file",)
    parser.add_argument("--experiment_type", type=str, default="cot", help="The type of experiment")
    parser.add_argument("--workers_num", type=int, default=10, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")

    parser.add_argument("--industry", type=str, default='Semiconductors', help="")
    parser.add_argument("--task_type", type=str, default='numerical_reasoning', help="")
    parser.add_argument("--retriever_ip", type=str, default="127.0.0.1", help="") # table 127.0.0.1 chart 127.0.0.1
    parser.add_argument("--retriever_port", type=int, default=8001, help="")
    parser.add_argument("--small_model_ip", type=str, default="127.0.0.1", help="") # table 127.0.0.1 chart 127.0.0.1
    parser.add_argument("--small_model_port", type=int, default=8001, help="")
    parser.add_argument("--output_file_name", type=str, default="output.jsonl", help="")

    args = parser.parse_args()
    return args



# pdb.set_trace = lambda: None
if __name__ == "__main__":
    args = arg_parse()
    base_url = "http://"+args.small_model_ip+":"+str(args.small_model_port)+"/v1"
    search_url = "http://"+args.retriever_ip+":"+str(args.retriever_port)+"/search"
    
    industry_name = args.industry
    task_type = "fact_verify" if args.task_type == "explainable_fact_verification" else args.task_type
    prompt_inst = {"numerical_reasoning": numerical_cot_prompt_v3,
                   "table_generation": table_cot_pormpt,
                   "chart_generation": chart_cot_promt_v2,
                   "fact_verify": fact_cot_pormpt_v2,
                   "knowledge_reasoning": knowledge_cot_pormpt}[task_type]

    reranker = AutoModel.from_pretrained(
        os.environ.get('FINMRAG_RERANK_MODEL', 'jinaai/jina-reranker-m0'),
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    is_rerank = True

    mmrag = MMRAG(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        workers_num=args.workers_num,
        topk=args.topk,
        output_file_name=args.output_file_name,
    )
    mmrag.eval_dataset()
