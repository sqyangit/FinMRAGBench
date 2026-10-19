
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import requests
import json
from transformers import AutoModel
import sys
import time
import re
import random
from tqdm import tqdm

# from stock_map import *
from stock_map import decomposition_prompt, recheck_prompt, extra_metadata_prompt, subquery_judge_prompt, final_judge_prompt ,ticker_to_name_semiconductors, finance_metrics, industry_to_dict

sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))

from llm_local import VLM
from synthesis_prompt import numerical_reasoning_syn_prompt, numerical_reasoning_syn_prompt_text, chart_generation_syn_prompt_text, table_generation_syn_prompt_text2, table_generation_syn_prompt_text1, fact_verify_syn_prompt_text, knowledge_reasoning_syn_prompt_text


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
            response = llm.generate(query=query, image=image_list, max_new_tokens=8192)
            # 事实验证任务可启用答案及解释字段检查
            # assert 'answer' in response and 'explanation' in response
            assert "errorCode" not in response
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


def search(query, ip="127.0.0.1", port=8002, metadata_filters=None):
    url = f"http://{ip}:{port}/search"
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


def query_decomposition(query: str):
    """
    用大模型做原子化分解：公司、年份、复合指标拆分。暂时不做同义词扩展。
    失败时回退为 [query]。
    """

    # 查询分解的 few-shot 示例
    examples = [
        # 跨公司分解
        {
            "original_query": "Among Intel, AMD and NVIDIA, what are their revenue in 2023?",
            "subqueries": [
                "Intel 2023 revenue",
                "AMD 2023 revenue",
                "NVIDIA 2023 revenue"
            ]
        },
        # 跨年份分解
        {
            "original_query": "How much did NVIDIA's operating cash flow between 2021 and 2023?",
            "subqueries": [
                "NVIDIA 2021 operating cash flow",
                # "NVIDIA 2022 operating cash flow",
                "NVIDIA 2023 operating cash flow"
            ]
        },
        # 复合指标分解（ROA = Net Income / Average Total Assets）
        {
            "original_query": "How much did NVIDIA's ROA in FY2023",
            "subqueries": [
                "NVIDIA FY2023 net income",
                "NVIDIA FY2022 total assets",
                "NVIDIA FY2023 total assets"
            ]
        },
    ]

    # 拼接 Example 字符串
    example_texts = []
    for i, ex in enumerate(examples, start=1):
        example_texts.append(
            f"Example {i}:\n"
            f"Original query: {ex['original_query']}\n"
            f"Subqueries: {json.dumps(ex['subqueries'], ensure_ascii=False)}"
        )
    examples_block = "\n\n".join(example_texts)

    # 构造查询分解提示词
    prompt = decomposition_prompt.format(
        examples_block=examples_block, 
        query=query
    ).strip()

    # print(decomposition_prompt)

    # import pdb; pdb.set_trace()
    llm = VLM("gpt-5.1")
    raw = safe_generate(llm, query=prompt, image_list='', max_retries=5, wait_seconds=5)
    # print("raw_response:", raw)
    if not raw:
        return [query]

    try:
        content = extract_json_from_response(raw)
        obj = json.loads(content)
        subs = obj.get("subqueries", None)
        if not subs or not isinstance(subs, list):
            return [query]
        # 清洗子查询并去重，保留原顺序
        cleaned = []
        seen = set()
        for s in subs:
            if not isinstance(s, str):
                continue
            t = re.sub(r"\s+", " ", s.strip())
            if t and t.lower() not in seen:
                seen.add(t.lower())
                cleaned.append(t)
        return cleaned if cleaned else [query]
    except Exception:
        return [query]


# 返回跟子问题相关度高于一定阈值的节点
# 根据页面与子查询的相关性得分筛选证据
def query_recheck(nodes, subquery, model_name, base_url):
    import os
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img')
    llm = VLM(model_name, base_url)
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


def extra_metadata(subquery, industry):
    llm = VLM("gpt-5.1")
    ticker_map = industry_to_dict[industry]

    prompt = extra_metadata_prompt.format(
        industry=industry,
        ticker_map=json.dumps(ticker_map, indent=2),
        subquery=subquery
    )

    response = safe_generate(llm, query=prompt, image_list='')
    # import pdb;pdb.set_trace()

    try:
        json_str = extract_json_from_response(response)
        data = json.loads(json_str)
    except Exception:
        # 跳过无法解析为 JSON 的输出
        print("Invalid extra_metadata JSON output")
        data = {}

    ticker = data.get("ticker", "")
    year = data.get("year", "")
    return ticker, year


def subquery_judge(subquery, nodes):
    """
    对单个子查询的已召回页面进行充分性判断：
    - 输入：子查询文本 subquery，与该子查询对应的图像列表 imagelist
    - 输出：("YES"/"NO", rationale 一句话)
    """
    llm = VLM("gpt-5.1")
    text_base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/text_clean/')
    imagelist = [node["node"]["metadata"]["file_name"] for node in nodes]
    filelist = [os.path.basename(image) for image in imagelist]

    # 读取 image 对应的 OCR 文本文件
    textlist = [os.path.join(text_base_path, os.path.splitext(f)[0] + ".txt") for f in filelist]
    text_contents = []
    for text_path in textlist:
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                text_contents.append(content)
        except FileNotFoundError:
            print(f"文件未找到: {text_path}")
            text_contents.append("")  # 保持长度一致

    # 将文件名与文本绑定并加编号，利于模型定位
    text_block = "\n\n".join(
        f"### File {i+1}: {fname}\n{text}"
        for i, (fname, text) in enumerate(zip(filelist, text_contents))
    )

    prompt = subquery_judge_prompt.format(
        subquery=subquery,
        text_block=text_block
    ).strip()

    # 调用模型生成
    response = safe_generate(llm, query=prompt, image_list='')

    # 尝试解析 JSON 输出
    try:
        json_str = extract_json_from_response(response)
        data = json.loads(json_str)
    except Exception:
        print("Invalid subquery_judge JSON output")
        data = {}

    judge = data.get("judge", "")
    rationale = data.get("rationale", "")
    return judge, rationale

# 检查最终筛选的页面是否足以回答
def final_judge(query, subquerylist, imagelist):
    llm = VLM("gpt-5.1")
    text_base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/text_clean/')
    filelist = [os.path.basename(image) for image in imagelist]
    
    # 读取 image 对应的 OCR 文本文件
    textlist = [os.path.join(text_base_path, os.path.splitext(f)[0] + ".txt") for f in filelist]
    text_contents = []
    for text_path in textlist:
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                text_contents.append(content)
        except FileNotFoundError:
            print(f"文件未找到: {text_path}")
            text_contents.append("")  # 保持长度一致

    # 构造 prompt
    prompt = final_judge_prompt.format(
        query=query,
        subqueries_block="\n".join(f"- {sq}" for sq in subquerylist),
        text_block="\n\n---\n\n".join(text_contents)
    ).strip()

    # 调用模型生成
    response = safe_generate(llm, query=prompt, image_list='')
    # import pdb;pdb.set_trace()

    # 尝试解析 JSON 输出
    try:
        json_str = extract_json_from_response(response)
        data = json.loads(json_str)
    except Exception:
        print("Invalid final_judge JSON output")
        data = {}

    judge = data.get("judge", "")
    rationale = data.get("rationale", "")
    return judge, rationale

def synthesis_gpt(system_prompt, imagelist):
    llm = VLM("gpt-5.1")
    text_base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/text_clean/')
    filelist = [os.path.basename(image) for image in imagelist]
    
    # 读取image对应的ocr文件
    textlist = [os.path.join(text_base_path, os.path.splitext(f)[0] + ".txt") for f in filelist]
    text_contents = []
    for text_path in textlist:
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                text_contents.append(content)
        except FileNotFoundError:
            print(f"文件未找到: {text_path}")
            text_contents.append("")  # 保持长度一致

    text_block = "\n\n".join(
        f"### File {i+1}: {fname}\n{text}"
        for i, (fname, text) in enumerate(zip(filelist, text_contents))
    )

    if task_type == "table_generation":
        system_prompt = system_prompt.format(
            company_name=companies_readable,
            year=year,
            text_block=text_block
        )
    else:
        system_prompt = system_prompt.format(
            text_block=text_block
        )
    # import pdb;pdb.set_trace()
    # user_prompt = (
    #     system_prompt
    #     + f"\nAll images are from annual reports of publicly listed companies. Each file is named with the company name, year, and page number as follows: [{', '.join(filelist)}], corresponding one-to-one with the images. When annotating evidence_pages, you must use the data and page numbers from the above list, not the page numbers shown within the images.The image file names from the list should not appear in the question."
    # )
    user_prompt = (
        system_prompt
        + ""
    )
    response = safe_generate(llm, query=user_prompt, image_list='')
    # import pdb;pdb.set_trace()
    return response


def natural_join(items):
    """把 ['A','B','C'] 变成 'A, B, and C'；两个元素时 'A and B'；一个元素直接返回。"""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", type=str, default='Semiconductors', help="")
    parser.add_argument("--task_type", type=str, default='numerical_reasoning', help="")
    parser.add_argument("--retriever_ip", type=str, default="127.0.0.1", help="") # fact 127.0.0.1 knowledge 127.0.0.1
    parser.add_argument("--retriever_port", type=int, default=8001, help="")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="")
    parser.add_argument("--small_model_ip", type=str, default="127.0.0.1", help="") # fact 127.0.0.1 knowledge 127.0.0.1
    parser.add_argument("--small_model_port", type=int, default=8001, help="")
    # parser.add_argument("--large_model", type=str, default="gpt-5.1", help="")
    # parser.add_argument("--rerank_model", type=str, default="jinaai/jina-reranker-m0", help="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()

    base_url = "http://"+args.small_model_ip+":"+str(args.small_model_port)+"/v1"
    recheck_model_name = args.small_model

    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
    text_base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/text_clean/')

    reranker = AutoModel.from_pretrained(
        os.environ.get('FINMRAG_RERANK_MODEL', 'jinaai/jina-reranker-m0'),
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    is_rerank = True

    for i in tqdm(range(10), desc="生成问题中", ncols=100):
        end_flag = 0
        # 每次生成5个问题
        print(f"\n======================== 第 {i+1} 次生成 ===========================")


        # ===============同公司跨年份===============
        # industry_name = args.industry
        # industry_map = industry_to_dict[industry_name]
        # company_tickers = random.choice(list(industry_map.keys()))
        # company_name = industry_map.get(company_tickers, company_tickers)
        # company_name = company_name.split(", Inc.")[0].split(" Inc.")[0].split(" Corporation")[0].split(" Incorporated")[0]

        # 数值推理的可选年份采样配置
        # years = sorted(random.sample(range(2020, 2025), 2))

        # 图表生成的可选年份配置：在连续四年范围内采样
        # while True:
        #     years = sorted(random.sample(range(2020, 2025), 2))
        #     if years[1] - years[0] <= 2: # 年份跨度最多为两年，例如 2020 至 2022 年
        #         break
        # # ---------


        # 表格生成的可选年份配置：选择相邻两年
        # start = random.choice(range(2020, 2024))  # 2024 后面没有相邻年份
        # years = [start, start + 1]
        # #---------


        # year = [str(y) for y in years]
        # year_str = f"'{year[0]}' and '{year[1]}'" # 给table generation准备的
        # metric = random.choice(finance_metrics)
        # metadata_filters = {"file_name": [company_tickers + '_10K_' + year[0], company_tickers + '_10K_' + year[1]]}

        # query = f"How much did {company_name}'s {metric} between {year[0]} and {year[1]}?"
        # print(query)

        # 任务类型
        # numerical_reasoning, chart_generation, table_generation, fact_verify
        # task_type = "knowledge_reasoning" 
        task_type = args.task_type

        cross_type = "cross_company" # cross_year cross_company
        # numerical_reasoning_syn_prompt_text, chart_generation_syn_prompt_text, table_generation_syn_prompt_text, fact_verify_syn_prompt_text

        if task_type == "fact_verify":
            system_prompt = fact_verify_syn_prompt_text 
        elif task_type == "knowledge_reasoning":
            system_prompt = knowledge_reasoning_syn_prompt_text
        elif task_type == "numerical_reasoning":
            system_prompt = numerical_reasoning_syn_prompt_text
        elif task_type == "chart_generation":
            system_prompt = chart_generation_syn_prompt_text
        elif task_type == "table_generation":
            system_prompt = table_generation_syn_prompt_text2
            if cross_type == "cross_year_extra":
                system_prompt = table_generation_syn_prompt_text1
            elif cross_type == "cross_year_cal":
                system_prompt = table_generation_syn_prompt_text2
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        



        # ===============同年份跨公司===============
        industry_name = args.industry
        industry_map = industry_to_dict[industry_name]
        company_tickers = random.sample(list(industry_map.keys()), 2) # 每个跨公司问题采样两家公司
        company_names = [industry_map.get(t, t) for t in company_tickers]
        companies_readable = natural_join(
            [
                name.split(", Inc.")[0]
                .split(" Inc.")[0]
                .split(" Corporation")[0]
                .split(" Incorporated")[0]  # 简化显示名
                for name in company_names
            ]
        )
        year = str(random.randint(2020, 2024)) # 包含头尾
        metric = random.choice(finance_metrics)
        metadata_filters = {"file_name": [company + "_10K_" + year for company in company_tickers]}

        query = f"Among {companies_readable}, what are their {metric} in {year}?"
        print(query)

        subquery_list = query_decomposition(query)
        print(subquery_list)

        # ===============搜索===============
        # 按 metadata 检索候选页面，再按相关性重排
        # 按子查询分配证据页数，并限制每份报告的最终保留页数

        # 单一查询
        # search_result = search(query=query, metadata_filters=metadata_filters)
        # # rerank
        # if is_rerank:
        #     reranked_nodes = rerank(query, search_result)
        # else:
        #     reranked_nodes = search_result["source_nodes"]

        # 多查询合并结果
        all_results = []
        # 子查询越多，每个子查询保留的证据页数越少
        if len(subquery_list)==2 or len(subquery_list)==1:
            rechecked_num = 5
        elif len(subquery_list)==3 or len(subquery_list)==4:
            rechecked_num = 3
        elif len(subquery_list)>=5 and len(subquery_list)<=7:
            rechecked_num = 2
        elif len(subquery_list)>=8:
            rechecked_num = 1

        for sub_q in subquery_list:

            ticker, year = extra_metadata(sub_q, industry_name)
            if ticker == "" or year == "":
                sub_metadata_filters = metadata_filters
                print(f"ticker:{ticker}, year:{year}. 有一个是空的")
            else:
                sub_metadata_filters = {"file_name": ticker + "_10K_" + year}

            print(sub_metadata_filters)
            search_result = search(query=sub_q, metadata_filters=sub_metadata_filters, ip=args.retriever_ip, port=args.retriever_port)

            # 处理指定公司或年份未命中报告的情况
            if(len(search_result["source_nodes"])==0):
                print("不存在{str}的文件".format(str=sub_metadata_filters["file_name"]))
                end_flag = 1
                break

            # rerank
            if is_rerank:
                reranked_nodes = rerank(sub_q, search_result, topk=50)
            else:
                reranked_nodes = search_result["source_nodes"]

            # 使用页面复核模型检查重排后的候选证据
            # 每批检查 10 页，累计足够证据后停止
            rechecked_nodes = []
            for i in range(0, len(reranked_nodes), 10):
                batch = reranked_nodes[i:i+10]
                batch_checked = query_recheck(batch, sub_q, recheck_model_name, base_url)
                rechecked_nodes.extend(batch_checked)
                # 达到证据页数要求后停止检查当前子查询
                if len(rechecked_nodes) >= rechecked_num:
                    print(f"[query_recheck] 已筛选出 {len(rechecked_nodes)} 个节点，提前结束子查询")
                    break

            # judge每个子查询召回的页面信息是否充足足以回答该子查询
            # import pdb;pdb.set_trace()
            judge_sufficient, rationale_sufficient = subquery_judge(sub_q, rechecked_nodes[0:rechecked_num])
            if judge_sufficient.upper() == "YES":
                print("该子查询筛选的页面充足")
            else:
                print("该子查询筛选的页面不足，直接break")
                end_flag = 2
                break
            
            all_results.extend(rechecked_nodes[0:rechecked_num])

        
        if end_flag == 1:
            end_flag = 0
            print("年报不存在，这次循环就过了吧")
            continue
        elif end_flag == 2:
            end_flag = 0
            print("子查询都不行了，这次循环就过了吧")
            continue


        # ===============后处理：每个文件最多保留n个==============
        image_list = []
        file_quota = {fname: 0 for fname in metadata_filters["file_name"]}  # 记录每个文件已保留数量
        max_per_file = 10  # 每份报告最多保留 10 页证据

        # import pdb;pdb.set_trace()
        for node in all_results[:20]:
            filename = node["node"]["metadata"]["file_name"]
            # 找到对应的 file_name
            for target_prefix in metadata_filters["file_name"]:
                if filename.startswith(target_prefix):
                    if file_quota[target_prefix] < max_per_file:
                        image_list.append(base_path + filename)
                        file_quota[target_prefix] += 1
                        print(filename)
                    break
        # 去重保持顺序
        image_list = list(dict.fromkeys(image_list))
        # import pdb;pdb.set_trace()


        # ===============判断筛选的页面是否充足============== （已废弃，改为子查询时就判断）

        # 可选的整题证据充分性检查；当前流程在子查询阶段检查证据
        # judge_sufficient, rationale_sufficient = final_judge(query, subquery_list, image_list)
        # if judge_sufficient.upper() == "YES":
        #     print("筛选的页面充足，开始合成")
        # else:
        #     print("筛选的页面不足，直接下一个")
        #     continue


        # ===============合成===============

        response = synthesis_gpt(system_prompt, image_list)
        # print(response)

        response_new = extract_json_from_response(response)
        response_json = json.loads(response_new)
        # print(response_json)



        # ===============存json===============
        # 将所挑选的页面存入，之后用于过滤掉 标注的evidence page并非是select page子集的数据
        filename_list = [os.path.basename(image) for image in image_list]
        # for item in response_json["result"]:
        #     item["industry"] = industry_name
        #     item["select_company"] = company_tickers
        #     item["select_year"] = year
        #     item["select_metric"] = metric
        #     item["subqueries"] = subquery_list
        #     item["select_pages"] = filename_list
        #     item["page_sufficient"] = rationale_sufficient

        # 降一级写入
        for item in response_json["result"]:
            item["synthesis_metadata"] = {
                "industry": industry_name,
                "select_company": company_tickers,
                "select_year": year,
                "select_metric": metric,
                "subqueries": subquery_list,
                "select_pages": filename_list,
                # "page_sufficient": rationale_sufficient
            }
            
            

        # 追加写入json文件
        json_file = f"industry_data/{task_type}/{cross_type}/questions_first_{industry_name}.json"
        # 确保目录存在
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        existing_data = {"result": []}
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = {"result": []}

        if "result" in response_json:
            existing_data["result"].extend(response_json["result"])
        else:
            print("Warning: response_json 中没有 'result' 字段")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"第 {i+1} 次结果写入完成，共计 {len(existing_data['result'])} 条记录")
