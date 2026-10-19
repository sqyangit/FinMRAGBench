# FinMRAGBench Data

This directory contains the 887 expert-verified question-answer annotations used
in **FinMRAGBench: A Realistic and Complex Benchmark for Multi-Modal RAG in
Financial Document Analysis**.

Paper: <https://aclanthology.org/2026.findings-acl.187/>

## Files

| File | Task | Samples |
| --- | --- | ---: |
| `explainable_fact_verification.json` | Explainable Fact Verification | 155 |
| `numerical_reasoning.json` | Numerical Reasoning | 213 |
| `table_generation.json` | Table Generation | 183 |
| `chart_generation.json` | Chart Generation | 182 |
| `knowledge_reasoning.json` | Knowledge-Intensive Reasoning | 154 |
| **Total** |  | **887** |

Across all tasks, 787 questions (88.73%) use evidence from multiple pages and
667 questions (75.20%) use evidence from multiple documents. The annotations
reference 1,357 unique evidence pages from 460 documents.

## Format

Each file is UTF-8 JSON with the following top-level structure:

```json
{
  "result": [
    {
      "id": 1,
      "question": "...",
      "subtype": "...",
      "evidence_pages": ["TICKER_10K_YEAR_PAGE.jpg"]
    }
  ]
}
```

Common fields:

- `id`: Sample ID within the current task file. IDs are not globally unique
  across files.
- `question`: The benchmark question or claim.
- `subtype`: Fine-grained task category.
- `evidence_pages`: Gold evidence page filenames. Filenames follow the pattern
  `<TICKER>_10K_<YEAR>_<PAGE>.jpg`.

Task-specific answer fields:

- Explainable Fact Verification: `answer` is one of `Supported`, `Refuted`, or
  `Insufficient`; `explanation` provides the evidence-grounded rationale.
- Numerical Reasoning: `answer` contains the reference numerical answer.
- Table Generation: `answer` contains the reference Markdown table.
- Chart Generation: `answer` contains the reference chart values and
  `answer_raw` contains the reference Python plotting code. Running this code
  requires common plotting packages such as pandas, Matplotlib, and Seaborn.
- Knowledge-Intensive Reasoning: `short_answer` and `long_answer` contain the
  concise and detailed reference responses.

The evidence images and the full retrieval corpus are not included in this
directory. `evidence_pages` records the page-level gold references needed to
evaluate retrieval when the corpus is available.

The annual-report corpus resources and a detailed source manifest will be
released soon. The manifest will list the industries, companies, and
annual-report years used in the corpus. Access instructions will be announced
in the repository's main README.

## Loading the Data

```python
import json
from pathlib import Path

path = Path("data_bench/numerical_reasoning.json")
with path.open("r", encoding="utf-8") as f:
    samples = json.load(f)["result"]

print(len(samples))
print(samples[0]["question"])
print(samples[0]["evidence_pages"])
```

When merging the task files, use `(task_name, id)` as the sample key.

## Citation

If you use these annotations, please cite the FinMRAGBench paper:

> Shouqing Yang, Qi Zhang, Yuhang Yang, et al. FinMRAGBench: A Realistic and
> Complex Benchmark for Multi-Modal RAG in Financial Document Analysis.
> Findings of ACL 2026, pages 3822-3867.

The paper page provides the official BibTeX entry:
<https://aclanthology.org/2026.findings-acl.187/>

## License

The benchmark annotations are released under the repository's
[Apache License 2.0](https://github.com/sqyangit/FinMRAGBench/blob/main/LICENSE). The annotation archive includes a copy of the
license. Original annual reports and model weights are distributed separately.
