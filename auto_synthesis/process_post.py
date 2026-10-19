#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def extract_result_items(input_file, output_file, n=2):
    """
    从 JSON 文件中提取 result 字段的前 n 个元素，
    并保持原有的 result 包裹形式保存到新的文件。

    参数:
        input_file (str): 输入 JSON 文件路径
        output_file (str): 输出 JSON 文件路径
        n (int): 要提取的数量，默认为 2
    """
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 result 前 n 项
    subset = {"result": data["result"][:n]}

    # 保存到新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    print(f"已成功提取 result 的前 {n} 个项并保存到 {output_file}")

def load_items(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种结构：{"result": [...]} 或直接是 [...]
    if isinstance(data, dict) and "result" in data:
        items = data["result"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("无法识别的输入结构：应为列表或包含 'result' 的字典。")
    if not isinstance(items, list):
        raise ValueError("'result' 应为列表。")
    return items

def pick_cot_answer(ann_meta: dict):
    """按 vote_number（1-based）从 answer_raw 中选 cot_answer；容错处理"""
    if not isinstance(ann_meta, dict):
        return None
    answers = ann_meta.get("answer_raw") or []
    vote_number = ann_meta.get("vote_number")

    # 默认按 1-based 取值；异常时尽量回退
    idx = None
    if isinstance(vote_number, int):
        # 视为 1-based
        idx = vote_number - 1
        # 兼容从 0 开始的候选答案索引
        if not (0 <= idx < len(answers)) and 0 <= vote_number < len(answers):
            print("警告：vote_number 可能是 0-based")
            idx = vote_number

    if idx is not None and 0 <= idx < len(answers):
        # print(f"vote_number={vote_number}，取 answers[{idx}]")
        return answers[idx]

    # 回退策略：若没取到，尝试取第一项
    if answers:
        print("警告：vote_number 取值异常，取 answers[0]")
        return answers[0]
    
    print("警告：answer_raw 为空")
    return None

def transform(items):
    out = []
    for it in items:
        ann = (it or {}).get("annotation_metadata") or {}
        syn = (it or {}).get("synthesis_metadata") or {}
        out.append({
            "id": it.get("id"),
            "question": it.get("question"),
            "evidence_pages": it.get("evidence_pages") or [],
            "select_pages": syn.get("select_pages") or [],
            "cot_answer": pick_cot_answer(ann),
            "answer": ann.get("answer_final"),  # 用最终答案
            "industry": syn.get("industry") # 保留来源行业信息
        })
    return out

def main():
    parser = argparse.ArgumentParser(description="Post-process demo.json")
    parser.add_argument("--input", default="questions_rationale_labeled_merged_final.json", help="输入 JSON 文件路径，例如 demo.json")
    # parser.add_argument("--out_jsonl", default="processed.jsonl", help="输出 JSONL 文件名")
    parser.add_argument("--out_json",  default="rationale_data_processed.json", help="输出 JSON（整表）文件名")
    args = parser.parse_args()

    src = Path(args.input)
    items = load_items(src)
    processed = transform(items)

    # 写 JSONL
    # with Path(args.out_jsonl).open("w", encoding="utf-8") as f:
    #     for row in processed:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 写整表 JSON
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=4)

    print(f"完成：{args.out_json}；条目数={len(processed)}")

if __name__ == "__main__":
    # extract_result_items("questions_labeled_final.json", "demo.json", n=2)
    main()
