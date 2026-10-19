"""Run benchmark inference and page-level retrieval evaluation.

MMRAG selects an experiment, reads annotations, and writes resumable JSONL
results. Iterative reasoning and tool execution live in the agent modules.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import time
import warnings

import requests
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))

from inference_prompt import (
    chart_generation_infer_prompt2,
    fact_verify_infer_prompt,
    numerical_reasoning_infer_prompt,
    table_generation_infer_prompt,
    text_reasoning_infer_prompt1,
)
from utils.overall_evaluator import eval_search2


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
                raise RuntimeError("Generation failed after all retries") from e


class MMRAG:
    def __init__(self,
                dataset='FinanceReport',
                query_file='numerical_reasoning.json',
                experiment_type = 'vanilla_rag',
                generate_vlm='Qwen/Qwen3-VL-8B-Instruct',
                embed_model_name='BAAI/bge-m3',
                embed_model_name_vl=None, # openbmb/VisRAG-Ret vidore/colqwen2-v1.0
                embed_model_name_text=None, # nvidia/NV-Embed-v2 BAAI/bge-m3
                workers_num = 1,
                topk=10,
                has_filter=False,
                search_port=8111,
                base_url=None,
                check_url=None,
                search_url=None,
                check_vlm="Qwen/Qwen3-VL-8B-Instruct",
                max_steps=10
                ):
        if experiment_type not in ('vanilla_rag', 'gold_rag', 'react', 'tooluse', 'vidorag'):
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        if workers_num < 1 or topk < 1 or max_steps < 0:
            raise ValueError("workers_num and topk must be positive; max_steps must be nonnegative")
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.query_file = query_file
        task_type = Path(self.query_file).stem
        if task_type == 'explainable_fact_verification':
            task_type = 'fact_verify'
        print("task_type:", task_type)
        self.task_type = task_type
        if task_type not in ('fact_verify', 'numerical_reasoning', 'table_generation',
                             'chart_generation', 'knowledge_reasoning'):
            raise ValueError(f"Unsupported task type inferred from query_file: {task_type}")

        self.dataset_dir = os.path.join(PROJECT_ROOT, 'FinMRAGAgent', 'data', dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(PROJECT_ROOT, "results", experiment_type)

        self.search_url = search_url or f"http://127.0.0.1:{search_port}/search"
        self.port = search_port
        self.has_filter = has_filter
        self.generate_vlm = generate_vlm
        self.generate_vlm_name = self.generate_vlm.split('/')[-1]
        self.base_url = base_url or 'http://127.0.0.1:8001/v1'
        self.check_url = check_url or 'http://127.0.0.1:8001/v1'
        self.vlm = None
        if experiment_type in ('vanilla_rag', 'gold_rag', 'vidorag'):
            from llms.llm import LLM
            self.vlm = LLM(model_name=generate_vlm, base_url=base_url)

        # jina reranker 
        self.reranker_name = "jina-reranker-m0"
        # Agent classes load their own rerankers; vanilla/gold RAG do not need one.
        self.reranker = None

        # Retrieval is served over HTTP. Keep legacy embedding arguments for scripts
        # that still pass them, without loading unused local indexes or GPU models.
        self.search_engine = None
        if (embed_model_name != 'BAAI/bge-m3' or embed_model_name_vl is not None
                or embed_model_name_text is not None):
            warnings.warn("Embedding arguments do not configure the HTTP search service; "
                          "set its model in search_engine_api.py instead.", UserWarning)

        if experiment_type in ('vanilla_rag', 'gold_rag'):
            self.eval_func = self.vanilla_rag
        elif experiment_type == 'vidorag':
            try:
                from vidorag_agents import ViDoRAG_Agents
            except ModuleNotFoundError as exc:
                if exc.name in ('agent', 'agent.agent_prompt', 'agent.map_dict'):
                    raise RuntimeError("ViDoRAG prompt/mapping modules are missing. "
                                       "See the README adaptation instructions.") from exc
                raise
            self.agents = ViDoRAG_Agents(self.vlm)
            self.eval_func = self.vidorag
        else:
            if experiment_type == 'react':
                from react_agent import ReactAgent
                agent_class = ReactAgent
                self.eval_func = self.react
            else:
                from tooluse_agent import ToolAgent
                agent_class = ToolAgent
                self.eval_func = self.tooluse
            self.agents = agent_class(
                base_url=self.base_url, search_url=self.search_url,
                model_name=generate_vlm, check_url=self.check_url,
                task_type=self.task_type, image_dir=self.img_dir,
                check_model_name=check_vlm, max_steps=max_steps,
            )

        self.output_file_name = f'{experiment_type}_{self.generate_vlm_name}_{task_type}.jsonl'
        self.output_file_path = os.path.join(
            self.results_dir, self.generate_vlm_name, self.output_file_name,
        )

    # Standalone retrieval helpers retained for script compatibility.
    # CLI experiments use the corresponding helpers inside their agent classes.
    def rerank(self, query, search_result, topk=10):
        image_list = []
        base_path = self.img_dir + os.sep
        nodes = search_result['source_nodes']
        if not nodes:
            return []
        if self.reranker is None:
            from transformers import AutoModel
            self.reranker = AutoModel.from_pretrained(
                os.environ.get('FINMRAG_RERANK_MODEL', 'jinaai/jina-reranker-m0'),
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).to('cuda').eval()

        # 取前 50 个候选文档
        for node in nodes[:50]:
            filename = node['node']['metadata']['file_name']
            image_list.append(base_path + filename)

        image_pairs = [[query, doc] for doc in image_list]
        scores = self.reranker.compute_score(image_pairs, max_length=2048, doc_type="image")
        if len(scores) != len(nodes[:50]):
            print(f"[警告] 打分数量({len(scores)})与候选数量({len(nodes[:50])})不一致，可能有异常。")

        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        topk_indices = [i for i, _ in sorted_indices[:topk]]
        reranked_nodes = [nodes[i] for i in topk_indices]

        # 将重排分数写入 metadata
        for i, idx in enumerate(topk_indices):
            reranked_nodes[i]['node']['metadata']['rerank_score'] = float(scores[idx])

        print(f"Rerank 完成，返回 Top-{topk} 文档，共{len(reranked_nodes)}个文档")

        return reranked_nodes
      
    def query_recheck(self, nodes, subquery, golden_company_list, model_name="Qwen/Qwen3-VL-8B-Instruct", base_url="http://127.0.0.1:8004/v1", max_pages_per_company=5):
        import os
        import sys
        import json
        import re

        base_path = self.img_dir + os.sep

        sys.path.append(os.path.join(PROJECT_ROOT, 'inference'))
        sys.path.append(os.path.join(PROJECT_ROOT, 'auto_synthesis'))
        from llm_local import VLM
        from stock_map import recheck_prompt

        def extract_company_from_filename(fname):
            # 期望格式类似 "MCD_10K_2024.jpg"
            # 取第一个下划线前的部分为 ticker；若无下划线，则取文件名（去扩展名）
            if not fname:
                return ""
            base = os.path.basename(fname)
            # remove extension
            base_no_ext = os.path.splitext(base)[0]
            # split on underscore
            parts = base_no_ext.split("_", 1)
            return parts[0].upper()

        # 标准化 golden list 为大写集合，便于快速查找
        golden_set = {c.upper() for c in golden_company_list}

        llm = VLM(model_name, base_url)

        rechecked_nodes = []
        # 计数每个公司的已保留页数
        kept_counts = {c: 0 for c in golden_set}

        # 公司列表为空时仅按相关性得分筛选，不限制公司和每家公司页数
        limit_per_company = max_pages_per_company if golden_set else None
        last_idx = 0
        for idx, node in enumerate(nodes, start=1):
            last_idx = idx
            # 如果所有公司都达到上限，提前结束
            if golden_set and all(count >= limit_per_company for count in kept_counts.values()):
                break

            file_name = node.get("node", {}).get("metadata", {}).get("file_name", "")
            if not file_name:
                # 无文件名就跳过
                continue

            company = extract_company_from_filename(file_name)

            # 如果指定了 golden list 且当前文件不在其中，跳过
            if golden_set and company not in golden_set:
                continue

            # 如果该公司已经达到上限，跳过
            if limit_per_company is not None and kept_counts.get(company, 0) >= limit_per_company:
                continue

            img_path = os.path.join(base_path, file_name)

            prompt = recheck_prompt.format(subquery=subquery).strip()

            # 模型推理
            try:
                result = llm.generate(query=prompt, image=img_path)
            except Exception as e:
                print(f"[query_recheck] LLM generation error for {file_name}: {e}")
                continue

            # 尝试解析模型输出为 JSON
            try:
                
                json_str = extract_json_from_response(result)
                data = json.loads(json_str)
            except Exception:
                print(f"[query_recheck] Invalid JSON output for {file_name}")
                continue

            score = data.get("score", 0)
            if score >= 7:
                # 将相关性得分及理由写入 metadata
                meta = node.setdefault("node", {}).setdefault("metadata", {})
                meta["recheck_score"] = score
                if "rationale" in data:
                    meta["recheck_rationale"] = data.get("rationale", "")

                rechecked_nodes.append(node)
                # 增加该公司计数（如果在 golden_set 中）
                if company in kept_counts:
                    kept_counts[company] += 1

        # 打印统计（显示哪些公司还没达到上限）
        if golden_set:
            for c in sorted(golden_set):
                print(f"[query_recheck] Kept {kept_counts.get(c,0)} / {max_pages_per_company} pages for company {c}")

        print(f"[query_recheck] Last processed page index: {last_idx}")
        print(f"[query_recheck] Kept {len(rechecked_nodes)} of {len(nodes)} nodes total")
        return rechecked_nodes

    # POST
    def search(self, query, port=8111, metadata_filters=None):
        """Query search_url; port is retained for compatibility with older callers."""
        url = self.search_url
        data = {
            "query": query,
            "metadata_filters": metadata_filters
        }
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        return response.json()

    def vanilla_rag(self,sample):
        
        base_path = self.img_dir + os.sep
        llm = self.vlm
        query = sample['question']
        print("question: ", query)


        # ============ vanilla retrieval ============
        if self.experiment_type == "vanilla_rag":
            # api search
            recall_results = self.search(query, port=self.port)
            recall_results['source_nodes'] = recall_results['source_nodes'][:self.top_k]

            sample['recall_results'] = recall_results
            sample['recall_pages'] = [node['node']['metadata']['file_name'] for node in recall_results['source_nodes']]
            golden_image_list = [base_path + node['node']['metadata']['file_name'] for node in recall_results['source_nodes']]

        # ============ golden evidence ============
        if self.experiment_type == "gold_rag":
            # no search
            recall_results = {}
            recall_results['source_nodes'] = []

            sample['recall_results'] = recall_results
            sample['recall_pages'] = sample['evidence_pages']
            golden_image_list = [base_path + node for node in sample['evidence_pages']]

        task_prompts = {
            'table_generation': table_generation_infer_prompt,
            'knowledge_reasoning': text_reasoning_infer_prompt1,
            'numerical_reasoning': numerical_reasoning_infer_prompt,
            'chart_generation': chart_generation_infer_prompt2,
            'fact_verify': fact_verify_infer_prompt,
        }
        task_system_prompt = task_prompts[self.task_type]
        if self.experiment_type == "vanilla_rag":
            user_prompt = task_system_prompt + f"\n# Retrieved Documents\nEach retrieved image file is named with the company name, year, and page number, as follows: [{', '.join(sample['recall_pages'])}], corresponding one-to-one with the provided images.\n# Question: " + query
        
        if self.experiment_type == "gold_rag":
            # user_prompt = task_system_prompt + f"\n# Golden Documents\nEach retrieved image file is named with the company name, year, and page number, as follows: [{', '.join(sample['recall_pages'])}], corresponding one-to-one with the provided images.\n# Question: " + query
            
            user_prompt = task_system_prompt + f"\n# Question: " + query


        # import pdb;pdb.set_trace()
        
        if "Qwen" in self.generate_vlm or "InternVL" in self.generate_vlm or "Llama" in self.generate_vlm or "MiniCPM" in self.generate_vlm:
            response = llm.generate(query=user_prompt, image=golden_image_list)
        else:
            # 为远程模型调用设置重试次数
            if "gemini" in self.generate_vlm:
                max_try = 1
            else :
                max_try = 5

            response = safe_generate(llm, query=user_prompt, image_list=golden_image_list, max_retries=max_try)

            
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Generation returned an empty or non-text response")
        # sample['eval_result'] = self.evaluator.evaluate(query, sample['answer'], str(answer))
        sample['response'] = response

        return sample

    def vidorag(self,sample):
        # query = sample['query']

        base_path = self.img_dir + os.sep
        llm = self.vlm
        query = sample['question']
        print(query)



        # recall_results = self.search(query ,port=self.port, metadata_filters=None)
        # recall_results['source_nodes'] = recall_results['source_nodes'][:10]
        # sample['recall_results'] = recall_results
        # <PROJECT_PATH>
        
        retrieval_path = f"{PROJECT_ROOT}/results/hrps_rag/Qwen3-VL-8B-Instruct-company+year/hrps_rag_Qwen3-VL-8B-Instruct_{self.task_type}.jsonl"
        start = time.perf_counter()
        golden_image_list_raw = None
        with open(retrieval_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())

                # 判断 question 字段
                if item.get("question") == query:
                    golden_image_list_raw = item.get("recall_pages", [])
                    break  # 找到后退出循环

        if golden_image_list_raw is None:
            raise ValueError(f"No saved ViDoRAG retrieval result for question: {query}")

        end = time.perf_counter()
        print(f"direct search 耗时: {end - start:.6f} 秒")

        golden_image_list = [base_path + path for path in golden_image_list_raw[0:10]]

        # candidate_image = [os.path.join(self.img_dir, os.path.basename(node['node']['metadata'].get('file_name',node['node']['metadata'].get('filename')).replace('.txt','.jpg'))) for node in recall_results['source_nodes']]
        candidate_image = golden_image_list
        if 'gmm' not in self.experiment_type:
            candidate_image = candidate_image[:10]
            # candidate_image = candidate_image[:100]
        
        # import pdb;pdb.set_trace()
        try:
            answer = self.agents.run_agent(query, candidate_image, self.task_type)
            if answer is None or answer == '':
                raise ValueError('ViDoRAG returned no answer')
        except Exception as e:
            print(e)
            sample.update(response="", status="error", error=str(e), recall_results={"source_nodes": []})
            return sample
        # import pdb;pdb.set_trace()
        # sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        # sample['eval_result'] = self.evaluator.evaluate(query, sample['answer'], str(answer))
        sample['response'] = answer
        sample['recall_results'] = self._recall_results(candidate_image)
        return sample

    @staticmethod
    def _recall_results(images):
        from llama_index.core.schema import NodeWithScore, ImageNode
        return dict(
            source_nodes=[
                NodeWithScore(node=ImageNode(image_path=image, metadata=dict(file_name=image)),
                              score=None).to_dict()
                for image in images
            ],
            response=None,
            metadata=None,
        )

    def _run_iterative_agent(self, sample):
        """Normalize the shared (answer, pages, history) result of both agents."""
        try:
            answer, all_images, history = self.agents.run(sample['question'], return_history=True)
        except Exception as exc:
            # Preserve evidence from completed search steps if a later step fails.
            history = getattr(self.agents, 'last_history', [])
            all_images = list(dict.fromkeys(
                image for step in history for image in step.get('selected_images', [])
            ))
            answer = ""
            sample.update(status="error", error=str(exc))
        sample['response'] = answer
        sample['history'] = history
        sample['recall_results'] = self._recall_results(all_images)
        return sample

    def react(self, sample):
        return self._run_iterative_agent(sample)

    def tooluse(self, sample):
        return self._run_iterative_agent(sample)

    def eval_dataset(self):
        # import pdb;pdb.set_trace()
        def eval_func(item):
            try:
                result = self.eval_func(item)
                if not isinstance(result, dict):
                    raise ValueError("Inference did not return a sample record")
                return result
            except Exception as exc:
                # Keep failed samples in the output so task averages retain all IDs.
                print(f"Sample {item.get('id')} failed: {exc}")
                result = dict(item, response="", status="error", error=str(exc))
                result.setdefault("recall_results", {"source_nodes": []})
                return result
        # 读取基准测试标注
        # rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        # 从仓库的 data_bench 目录读取标注
        rag_dataset_path = os.path.join(os.path.join(PROJECT_ROOT, 'data_bench'), self.query_file)

        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        Path(self.output_file_path).touch(exist_ok=True)

        with open(rag_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data = data['examples']
        data = data['result']
        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    results.append(json.loads(line.strip()))
            uid_already = {item['id'] for item in results}
            data = [item for item in data if item['id'] not in uid_already]
            
        def write_result(result):
            # Persist each completed sample so interruptions do not lose a batch.
            with open(self.output_file_path, "a", encoding="utf-8") as output:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Agent instances retain per-question state and must not be shared by workers.
        if self.workers_num == 1 or self.experiment_type in ('react', 'tooluse', 'vidorag'):
            for item in tqdm(data):
                write_result(eval_func(item))
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    write_result(future.result())

        return self.output_file_path

    def eval_overall(self):
        data = []
        with open(self.output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data.append(json.loads(line.strip()))
        results = eval_search2(data)
        with open(self.output_file_path.replace(".jsonl", "_eval.json"), "w", encoding="utf-8") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='FinanceReport', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='numerical_reasoning.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='vanilla_rag', choices=['vanilla_rag', 'gold_rag', 'react', 'tooluse', 'vidorag'], help="The type of experiment")
    parser.add_argument("--embed_model_name", type=str, default='BAAI/bge-m3', help="Legacy argument; configure embeddings on the search service")
    parser.add_argument("--workers_num", type=int, default=1, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    parser.add_argument("--embed_model_name_vl", type=str, default=None, help="Legacy argument; unused by HTTP retrieval")
    parser.add_argument("--embed_model_name_text", type=str, default=None, help="Legacy argument; unused by HTTP retrieval")

    # parser.add_argument("--retriever_ip", type=str, default="127.0.0.1", help="") # table 127.0.0.1 chart 127.0.0.1
    # parser.add_argument("--retriever_port", type=int, default=8001, help="")

    parser.add_argument("--generate_ip", type=str, default="127.0.0.1", help="") # table 127.0.0.1 chart 127.0.0.1
    parser.add_argument("--generate_port", type=int, default=8001, help="")
    parser.add_argument("--generate_vlm", type=str, default='Qwen/Qwen3-VL-8B-Instruct', help="The name of VLM model")

    parser.add_argument("--check_ip", type=str, default="127.0.0.1", help="") # table 127.0.0.1 chart 127.0.0.1
    parser.add_argument("--check_port", type=int, default=8001, help="")    

    parser.add_argument("--check_vlm", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name served by the page-filtering endpoint")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Maximum tool steps; permits one final answer-only model call")
    parser.add_argument("--has_filter", type=str2bool, nargs="?", const=True, default=False, help="Legacy argument; agents filter pages internally")
    parser.add_argument("--search_ip", type=str, default="127.0.0.1", help="")
    parser.add_argument("--search_port", type=int, default=8111, help="The port of search engine")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()

    generate_url = "http://"+args.generate_ip+":"+str(args.generate_port)+"/v1"
    check_url = "http://"+args.check_ip+":"+str(args.check_port)+"/v1"
    search_url = "http://"+args.search_ip+":"+str(args.search_port)+"/search"

    mmrag = MMRAG(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        embed_model_name=args.embed_model_name,
        workers_num=args.workers_num,
        topk=args.topk,
        embed_model_name_vl=args.embed_model_name_vl,
        embed_model_name_text=args.embed_model_name_text,
        generate_vlm=args.generate_vlm,
        has_filter=args.has_filter,
        search_port=args.search_port,
        base_url = generate_url,
        check_url = check_url,
        search_url = search_url,
        check_vlm=args.check_vlm,
        max_steps=args.max_steps,
    )
    # import pdb;pdb.set_trace()

    mmrag.eval_dataset()
    mmrag.eval_overall()

    # mmrag.eval_overall_type_wise()
