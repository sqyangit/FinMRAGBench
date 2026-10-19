llm_judge_prompt_fact_verify = """
You are an expert evaluator of financial reasoning explanations.

You are given:
1. A financial claim
2. A human-written gold explanation that correctly verifies the claim
3. A model-generated explanation for the same claim

Your task is to evaluate whether the model explanation is logically consistent with the gold explanation and whether it captures the correct reasoning needed to justify the claim. Compare the reasoning structure, key analytical steps, and conclusion.

Evaluate the model explanation based on the following criteria:

1. **Correctness**: Does the explanation reach the correct conclusion regarding the claim?
2. **Logical Alignment**: Is its reasoning process consistent with that of the gold explanation?
3. **Completeness**: Does it include the essential reasoning steps presented in the gold explanation?
4. **Clarity**: Is the explanation easy to understand and well-organized?

---

### Scoring Guidelines (10-point scale):

- **1–2**: The explanation is irrelevant, incorrect, or fundamentally flawed.
- **3–4**: Marginal relevance; contains major reasoning gaps or misunderstandings.
- **5–6**: Partially correct; some alignment with the gold reasoning but missing key steps.
- **7–8**: Mostly correct and aligned; minor omissions but sound reasoning overall.
- **9–10**: Highly accurate, logically aligned, complete, and clearly articulated.

---

### Output Format

Return the evaluation in the following JSON format:

```json 
{{
    "rationale": "your rationale for the score, as a text",
    "score": "your score from 1 to 10"
}}
```

---

[Claim]:  
{claim}

[Gold Explanation]:  
{gold_explanation}

[Model Explanation]:  
{model_explanation}
"""

# knowledge reasoning
llm_judge_prompt_general = """
You are an expert evaluator of financial question-answering systems.

You are given:
1. A financial question
2. A human-annotated gold answer
3. A model-generated answer to the same question

Your task is to evaluate the model answer by comparing it with the gold answer, focusing on correctness, reasoning completeness, and clarity.

Evaluate the model answer on the following criteria:

1. **Correctness**: Does the model answer provide factually accurate, financially sound, and contextually appropriate information?
2. **Logical Consistency**: Is the reasoning coherent, logically valid, and aligned with the intent of the question?
3. **Coverage**: Does the answer fully and directly address all components of the question, matching the breadth and depth of the gold answer?
4. **Clarity**: Is the explanation clearly written, well-structured, and easy for a financially knowledgeable audience to understand?

---

### Scoring Guidelines (10-point scale):

- **1–2**: Irrelevant, incorrect, or fundamentally flawed; does not address the question meaningfully.
- **3–4**: Partially relevant but contains major factual errors, omissions, or logical issues.
- **5–6**: Moderately correct; captures some key ideas but lacks important details or clarity compared with the gold answer.
- **7–8**: Mostly correct and well-aligned with the gold answer; minor omissions or slight reasoning gaps.
- **9–10**: Highly accurate, comprehensive, logically consistent, and clearly presented; closely matches or fully meets the standard of the gold answer.

---

### Output Format

Return the evaluation in the following JSON format:

```json
{{
    "rationale": "your rationale for the score, as a text",
    "score": "your score from 1 to 10"
}}
```
---

[Question]:  
{question}

[Gold Answer]:  
{gold_answer}

[Model Answer]:  
{model_answer}
"""