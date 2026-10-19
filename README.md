# [ACL 2026] FinMRAGBench

This repository contains the code and benchmark annotations for our paper, **FinMRAGBench: A Realistic and Complex Benchmark for Multi-Modal RAG in Financial Document Analysis**, published in *Findings of the Association for Computational Linguistics: ACL 2026*.

[Paper](https://aclanthology.org/2026.findings-acl.187/) · [PDF](https://aclanthology.org/2026.findings-acl.187.pdf)

FinMRAGBench covers five financial document analysis tasks, with questions spanning multiple pages and documents. We also provide FinMRAGAgent, which uses retrieval and Python computation in iterative reasoning, along with scripts for question-answer synthesis and chain-of-thought (CoT) trajectory collection.

## Repository structure

```text
FinMRAGAgent/       Page indexing, search API, agents, and inference experiments
  llms/            Model clients and visual embedding adapters
  utils/           Retrieval metrics, image helpers, and format conversion
  scripts/         PDF/page preparation utilities
inference/         Answer-generation prompts and model clients
evaluation/        Task-specific answer evaluation
auto_synthesis/    QA synthesis, filtering, annotation, CoT collection, SFT conversion
data_bench/        The five benchmark annotation files and their schema
tests/             Entry-point and agent regression checks with mocked model services
```

Run the commands below from the specified directories. The `.sh` files are experiment templates; set the models, devices, and ports for your environment before using them.

## Benchmark data

| Task | Annotation file | Samples |
| --- | --- | ---: |
| Explainable Fact Verification | `explainable_fact_verification.json` | 155 |
| Numerical Reasoning | `numerical_reasoning.json` | 213 |
| Table Generation | `table_generation.json` | 183 |
| Chart Generation | `chart_generation.json` | 182 |
| Knowledge-Intensive Reasoning | `knowledge_reasoning.json` | 154 |
| **Total** | | **887** |

The 887 questions reference 1,357 unique evidence pages from 460 documents. Of these questions, 787 require multiple pages and 667 require multiple documents. The full retrieval corpus used in the paper contains 723 filings and 96,549 pages, including pages beyond the annotated gold evidence.

See [data_bench/README.md](data_bench/README.md) for fields, answer formats, and loading examples. Sample IDs are unique within each task file, so use `(task, id)` when combining tasks. The inference entry point maps the fact-verification filename to the internal task name `fact_verify`.

The benchmark annotations are available in `data_bench/`. We will release the annual-report corpus resources and a source manifest soon. The manifest will list the industries, companies, and annual-report years used in the corpus, with access instructions provided in this repository.

When preparing page images, preserve the original page numbering and filenames to match the evidence annotations. Trained model weights and training trajectories are not included in this release.

## Environment and configuration

Use Python 3.10 or newer. GPU retrieval/inference requires a compatible PyTorch/CUDA environment.

```bash
python -m pip install -r requirements.txt
```

Install model-serving software such as vLLM separately in an environment compatible with your model and CUDA version. PDF conversion also requires Poppler.

The Python tool and chart evaluator execute model-generated code. Run these components in an isolated environment containing only the data and credentials required for the experiment.

Remote model clients use `OPENAI_API_KEY` and an OpenAI-compatible `OPENAI_BASE_URL`. Set these variables before running the scripts. The variables are listed in `.env.example`, which the scripts do not load automatically.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export FINMRAG_EMBED_MODEL="vidore/colqwen2-v1.0"
export FINMRAG_RERANK_MODEL="jinaai/jina-reranker-m0"
```

PowerShell equivalents:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:FINMRAG_EMBED_MODEL = "vidore/colqwen2-v1.0"
$env:FINMRAG_RERANK_MODEL = "jinaai/jina-reranker-m0"
```

The model variables accept a Hugging Face model ID or a local directory. Local generation and page-filtering services use the IP/port arguments in the inference and synthesis scripts. Their served model names must match the names requested by the clients. API usage logs are disabled by default; setting `FINMRAG_LOG_USAGE=1` records token counts only, without prompts or generated answers.

## Prepare the retrieval corpus

Arrange the corpus as follows:

```text
FinMRAGAgent/data/FinanceReport/
  pdf/                   Original report PDFs
  img/                   TICKER_10K_YEAR_PAGE.jpg
  text_clean/            Matching TICKER_10K_YEAR_PAGE.txt OCR files
  colqwen_ingestion/      Generated .node files
```

`FinMRAGAgent/scripts/pdf2images.py` includes a page-rendering example. The script counts PDF pages by default; enable the rendering block to create page images. QA synthesis also requires OCR text in `text_clean/`. An OCR pipeline is not included.

Build page embeddings:

```bash
cd FinMRAGAgent
python ingestion.py
```

The default is ColQwen2 with `colqwen_ingestion`. For ColPali or VisRAG-Ret, choose the corresponding model and output prefix in the script's main block. Existing `.node` files are skipped, so use a separate index directory for each model.

Start the search service from `FinMRAGAgent/`:

```bash
python search_engine_api.py --port 8111
```

`--industry` optionally restricts the loaded corpus using the mappings in `auto_synthesis/stock_map.py`. The API exposes `POST /search` with `query` and optional `metadata_filters`, and `GET /search?query=...`.

## Run inference

Start an OpenAI-compatible multimodal generation service separately, for example at `127.0.0.1:8001/v1`. Then run from `FinMRAGAgent/`:

```bash
python eval.py \
  --dataset FinanceReport \
  --query_file numerical_reasoning.json \
  --experiment_type vanilla_rag \
  --generate_vlm Qwen/Qwen3-VL-8B-Instruct \
  --generate_ip 127.0.0.1 --generate_port 8001 \
  --search_ip 127.0.0.1 --search_port 8111
```

| `--experiment_type` | Setting |
| --- | --- |
| `vanilla_rag` | Retrieve pages, then generate an answer |
| `gold_rag` | Generate using gold evidence pages |
| `react` | IRCOT-style iterative reasoning and search |
| `tooluse` | FinMRAGAgent with search and Python tools |
| `vidorag` | Optional ViDoRAG baseline; see requirements below |

For FinMRAGAgent, use `--experiment_type tooluse --generate_vlm <your-served-agent-model-name>`. Configure `--check_ip`, `--check_port`, and `--check_vlm` for the page-filtering model. The `--max_steps` option limits tool calls (default: 10), followed by one final opportunity for the model to answer. Both iterative agents use the image directory selected by `--dataset`. Inference results are written to `results/<experiment_type>/<model_name>/`; existing sample IDs are skipped when resuming.

Agent experiments process questions sequentially because the agent instance retains state between reasoning steps. Vanilla and gold RAG support parallel questions through `--workers_num`. Failed samples are saved with `status: error` and an empty response so they remain visible in the results and evaluation sample count.

`react` and `tooluse` rerank and filter pages during search. The `--has_filter` argument has no effect on vanilla RAG.

Retrieval runs through the search API. The embedding arguments in `eval.py` are retained for older launch scripts; configure the retrieval model in the search service.

The ViDoRAG prompt and image-layout mapping modules (`FinMRAGAgent/agent/agent_prompt.py` and `agent/map_dict.py`) are missing from this release. To run this baseline, adapt the prompts and mappings from the [upstream ViDoRAG project](https://github.com/Alibaba-NLP/ViDoRAG) to the financial tasks and interfaces in `vidorag_agents.py`. We are looking for the original experiment files and will add them if recovered. This baseline also requires saved retrieval results at the `retrieval_path` configured in `eval.py`.

## Evaluate generated answers

Each task has a separate evaluation script. Set `exp_type`, `model_type`, and the prediction path in the script's main block before running it. Predictions use JSONL records containing the annotation fields plus `response` and `recall_results`.

| Task | Script | Main reported metrics |
| --- | --- | --- |
| Numerical Reasoning | `evaluation/eval_numerical.py` | Token F1, exact match |
| Fact Verification | `evaluation/eval_explanation.py` | LLM judge score, HQR |
| Knowledge Reasoning | `evaluation/eval_knowledge.py` | LLM judge score, HQR |
| Table Generation | `evaluation/table_utils/eval_table.py` | RMS precision/recall |
| Chart Generation | `evaluation/chart_utils/eval_chart.py` | Executable code rate, value-match pass rate |

Run numerical/fact/knowledge scripts from `evaluation/`, table scripts from `evaluation/table_utils/`, and chart scripts from `evaluation/chart_utils/`. Numerical LLM judging is separately available in `FinMRAGAgent/llms/evaluator.py`. Retrieval metrics are in `FinMRAGAgent/utils/overall_evaluator.py`.

### Evaluation protocol

- Retrieval is evaluated with page-level Recall@10 in the paper. The scripts also provide hit rate, nDCG, MRR, and metrics over all retrieved pages. For iterative agents, report whether the score uses all accumulated evidence (`@all`) or a fixed top-10 budget.
- Numerical token F1 and exact match use the preprocessing in `evaluation/metrics/qa_metrics.py`: decimal rounding to one decimal place, lowercasing, punctuation and article removal, and whitespace normalization. Keep answer parsing and normalization consistent across models.
- Fact verification and knowledge reasoning use LLM judge scores from 1 to 10. HQR is the fraction of answers scoring at least 7. The separate numerical judge uses scores from 1 to 5, with a threshold of 4. The paper reports LJS as a percentage; convert raw mean scores using the corresponding scale and use the same judge model across compared runs.
- Table evaluation normalizes Markdown tables and computes RMS precision/recall. The numerical matching threshold is `0.01`.
- Chart evaluation reports executable code rate (ECR) and the pass rate from chart-data matching, which uses a 1% tolerance. It does not score chart styling or visual appearance.

Evaluate all samples in each task. Record the model versions, prompts, generation settings, retrieval budget, and judge configuration alongside the results. See the paper's evaluation appendix for metric definitions and the experimental protocol.

## QA and CoT data synthesis

The scripts in `auto_synthesis/` cover the following steps:

1. `synthesis.py`: select companies/years and financial metrics, decompose queries, retrieve and check evidence, and synthesize candidate questions.
2. `filter.py`: remove candidates with empty or inconsistent evidence sets; optionally rewrite ambiguous questions.
3. `annotation.py` and `llm_utils/`: generate candidate answers and select an answer through model-assisted voting.
4. `cot_synthesis.py`: collect multi-step JSON trajectories with `think`, `search`, `python`, and `answer` actions.
5. `cot2sft.py`: convert filtered trajectories into multimodal assistant/user messages with `<image>` placeholders and image paths.

For example, from `auto_synthesis/`:

```bash
python synthesis.py \
  --industry Restaurants --task_type numerical_reasoning \
  --retriever_ip 127.0.0.1 --retriever_port 8111 \
  --small_model_ip 127.0.0.1 --small_model_port 8001
```

Set the input/output paths and task options in the main blocks of the synthesis, filtering, and annotation scripts. Intermediate question files are stored under `auto_synthesis/industry_data/`. CoT collection reads filtered questions from `industry_data/<task>/filter/` and writes trajectories to `industry_data/<task>/cot/`.

Convert an already filtered trajectory file:

```bash
python cot2sft.py \
  --input industry_data/numerical_reasoning/final/train_filtered.jsonl \
  --task numerical_reasoning --output-name cot_numerical
```

This writes `auto_synthesis/data/cot_numerical.json` when run from `auto_synthesis/`. Update the local image paths for the training machine. The paper's data construction also includes expert review after automatic synthesis. Keep benchmark evaluation questions separate from training data.

## Tests

Run the entry-point and agent regression checks with:

```bash
python -m unittest discover -s tests -v
```

These tests use mocked model services and do not require a GPU or the report corpus. They check control flow and configuration; full benchmark runs require the resources described above.

## Citation

If you use this code or benchmark, please cite our paper:

```bibtex
@inproceedings{yang-etal-2026-finmragbench,
  title = "{F}in{MRAGB}ench: A Realistic and Complex Benchmark for Multi-Modal {RAG} in Financial Document Analysis",
  author = "Yang, Shouqing and Zhang, Qi and Yang, Yuhang and Xu, Ruikang and Hou, Yuwei and Jia, Zhulin and Gao, Lirong and Wang, Haobo and Chen, Jinglei and Wang, Jiexiang and Guo, Sheng and Zheng, Bo and Chen, Gang",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
  year = "2026",
  publisher = "Association for Computational Linguistics",
  pages = "3822--3867",
  doi = "10.18653/v1/2026.findings-acl.187",
  url = "https://aclanthology.org/2026.findings-acl.187/"
}
```

## Contact

For questions about the benchmark, code, or data, please contact Shouqing Yang at [sqyang@zju.edu.cn](mailto:sqyang@zju.edu.cn).

## Acknowledgments and license

The code builds on [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG), ColPali/ColQwen2, VisRAG-Ret, LlamaIndex, DePlot, and Pix2Struct. Third-party copyright and license headers are retained.

The code and benchmark annotations are released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for included third-party code. Model weights and annual-report resources are distributed separately under their respective terms.
