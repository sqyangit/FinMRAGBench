
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import json
import sys
import time
import re
from tqdm import tqdm

sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))
from llm_local import VLM
from stock_map import rewrite_prompt


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
            # 任务输出格式检查
            assert 'rewritten_flag' in response and 'new_question' in response
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


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def filter_questions(INPUT, OUTPUT):
    data = load_json(INPUT)
    results = data.get("result", [])
    if not isinstance(results, list):
        raise ValueError("`result` 字段应为列表")

    cleaned = []
    for idx, item in enumerate(results, start=1):
        meta = item.get("synthesis_metadata") or {}
        evidence_pages = item.get("evidence_pages", []) or []
        select_pages = meta.get("select_pages", []) or []
        if not isinstance(evidence_pages, list):
            evidence_pages = []
        if not isinstance(select_pages, list):
            select_pages = []

        # 检查是否所有 evidence 都在 select 里
        if evidence_pages and all(p in select_pages for p in evidence_pages):
            # 合法样本：保留并编号
            new_item = dict(item)
            new_item["id"] = len(cleaned) + 1
            cleaned.append(new_item)
        else:
            # 非法样本：跳过
            print(f"跳过不合法样本（id={idx}）：部分 evidence 不在 select_pages 中")

    out = {"result": cleaned}
    save_json(out, OUTPUT)
    print(f"处理完成：共保留 {len(cleaned)} 条，有效数据已写入 {OUTPUT}")


def rewrite_questions(INPUT, OUTPUT):
    data = load_json(INPUT)
    data = data.get("result", [])
    if not isinstance(data, list):
        raise ValueError("`result` 字段应为列表")
    
    # 读取已保存样本 ID，用于断点续跑
    if os.path.exists(OUTPUT):
        results = []
        with open(OUTPUT, "r") as f:
            for line in f:
                results.append(json.loads(line.strip()))
        uid_already = [item["id"] for item in results]
        data = [item for item in data if item["id"] not in uid_already]

    rewritten = []
    for idx, item in enumerate(tqdm(data, desc="Rewriting questions"), start=1):
        meta = item.get("synthesis_metadata") or {}
        question = item.get("question", "")
        pages = meta.get("select_pages", []) or []

        # 所使用页面对应的公司、年份
        tickers = set()
        periods = set()

        for page in pages:
            parts = page.split("_")
            if len(parts) >= 3:
                tickers.add(parts[0])   # 股票代码
                periods.add(parts[2])   # 年份

        ticker = ", ".join(sorted(tickers))
        period = ", ".join(sorted(periods))

        # 构造 prompt
        prompt = rewrite_prompt.format(
            ticker=ticker,
            period=period,
            question=question
        )
        # import pdb;pdb.set_trace()

        # 调用大模型生成
        llm = VLM("gpt-4o")
        response = safe_generate(llm, query=prompt, image_list='')
        response_new = extract_json_from_response(response)

        # 尝试解析 JSON
        try:
            response_json = json.loads(response_new)
        except Exception as e:
            print(f"[Warning] 第 {idx} 条生成的响应无法解析为 JSON：{e}")
            response_json = {}

        # 从响应中提取字段
        rewritten_flag = response_json.get("rewritten_flag", "")
        new_question = response_json.get("new_question", "")
        if isinstance(rewritten_flag, (int, float, bool)):
            rewritten_flag = str(rewritten_flag)

        # 更新原始 item
        item["rewritten_flag"] = rewritten_flag
        item["new_question"] = new_question

        # 立即写入输出文件（逐条追加）
        with open(OUTPUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 保存到结果列表
        rewritten.append(item) # 保留内存副本；结果已逐条写入文件

        if rewritten_flag == "1":
            print(f"[{idx}] 完成问题重写：{new_question}")
        elif rewritten_flag == "0":
            print(f"[{idx}] 无需重写：{question}")
        else:
            print(f"[{idx}] 重写失败：{question}")

    # 输出结果
    # output_data = {"result": rewritten}
    # with open(OUTPUT, "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\n所有结果已写入: {OUTPUT}")


def finalize_output(path_in, path_out=None):
    """将逐行 JSON 文件封装为标准 JSON 格式 { "result": [...] }"""
    if path_out is None:
        path_out = path_in.replace(".json", "_final.json")

    with open(path_in, "r", encoding="utf-8") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump({"result": items}, f, ensure_ascii=False, indent=4)

    print(f"封装完成：{path_out} （共 {len(items)} 条）")

    


if __name__ == "__main__":

    # 过滤证据页不属于候选页面集的样本
    # questions_first_path = "questions_first_merged.json"
    # questions_filtered_path = "questions_filtered.json"
    # filter_questions(questions_first_path, questions_filtered_path)

    # 批量过滤
    task_type = "table_generation"
    src_dir = f"{PROJECT_ROOT}/auto_synthesis/industry_data/{task_type}/cross_year"
    dst_dir = f"{PROJECT_ROOT}/auto_synthesis/industry_data/{task_type}/filter"
    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        if not filename.endswith(".json"):
            continue

        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)

        print(f"Filtering {src_file} → {dst_file}")
        filter_questions(src_file, dst_file)

    print("全部处理完成！")

    # ============改写问题============
    # 改写已通过证据检查的问题，补充检索所需的公司名称和年份
    # questions_rewritten_path = "questions_rewritten.json"
    # rewrite_questions(questions_filtered_path, questions_rewritten_path)

    # 封装为标准 JSON 格式
    # finalize_output(questions_rewritten_path)


