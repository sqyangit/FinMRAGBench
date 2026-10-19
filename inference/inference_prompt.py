rationale_prompt = """
You are an expert in financial annual report analysis.
The inputs you will receive are annual report page images (tables, charts, or statement pages).
Images are already provided above.

Read the annual report pages above relevant to the following question:
{question}

Annual Report Page Source: {source_name}

Your task:
1. Identify which parts of the images are useful to answer the question.
2. Explain clearly how the extracted content supports the answer. (This is the RATIONALE.)
3. Provide the final concise answer based on the rationale. If none of the images support the answer, use your own knowledge and explain your reasoning.

Format your response EXACTLY as follows:

Rationale:
- (step-by-step reasoning explaining how you arrived at the answer, referencing Annual Report Page if needed)

Final Answer: AnswerName1, AnswerName2...

The last output line MUST be only the Final Answer line and must follow this exact format:
Final Answer: AnswerName1, AnswerName2...

Now start answering the question: {question}.
"""

numerical_reasoning_infer_prompt = """
# Role Play
Suppose you are an expert in financial annual report analysis and your task is to provide precise answers to questions based on the content of annual report tables.

# Chain-of-Thought
Let’s think step by step as follows and make the most of your strengths as a financial report analysis expert:
1) Fully understand the question and extract the necessary information from it, including the metric(s), company names, fiscal years, and calculation/comparison requirements.  
2) Clearly and comprehensively understand the content of the annual report tables, including the structure, meaning of each row and column header, and any summative or flag rows (such as Total, Consolidated, Average).  
3) Based on the question, select the relevant rows and columns and locate the required values across the appropriate companies or years.  
4) According to the requirements of the question, perform the needed operations such as calculation, ranking, comparison, or aggregation.If needed, pay special attention to unit conversions (e.g., thousands vs. millions vs. billions, local currency vs. USD). Unless otherwise specified in the question, the final results should be converted and expressed in millions.  
5) Output the reasoning steps and then the final answer in the specified format.  

# Output Control
1) First, you need to output your reasoning steps according to the question and table itself. The reasoning steps should follow the format below: [Reasoning steps for this question are as following: 1.First, we need to... 2.We need to...]. output steps until final answers get solved.
2) Then, you need to output the final answer. The final answer should follow the format below: [Answer Format] Final Answer: AnswerName1, AnswerName2... Ensure the final answer format is the last output line and can only be in the "Final Answer: AnswerName1, AnswerName2..." form, no other form.
3) Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. Note: If the final answer has multiple decimals, retain two decimals.

I will give you multiple annual report page images and a question; please use them to answer following the above reasoning + answer format strictly.

"""

numerical_direct = """
# Role Play
Suppose you are an expert in financial annual report analysis and your task is to provide precise answers to questions.

# Chain-of-Thought
Let’s think step by step as follows:
1) Fully understand the question and extract the necessary information from it, including the metric(s), company names, fiscal years, and calculation/comparison requirements.  
2) Based on the information available in the question and your financial reasoning ability, determine the required values or relationships.  
3) Perform the needed operations such as calculation, ranking, comparison, or aggregation. Pay attention to unit conversions when necessary. Unless otherwise specified, express final results in millions.  
4) Output the reasoning steps and then the final answer in the specified format.

# Output Control
1) First, output your reasoning steps in the format:  
   [Reasoning steps for this question are as following: 1.First, we need to... 2.We need to...].
2) Then output the final answer strictly in the format:  
   Final Answer: AnswerName1, AnswerName2...
3) Ensure the final answer is concise, without explanation, and retain two decimal places if applicable.
4) The final answer line must be the last output line.

I will give you a question. Please answer it following the above reasoning and output format strictly.

"""



TOOL_PROTOCOL = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{function_list}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

# 通用表格生成提示词
chart_generation_infer_prompt = """
# Role Play
Suppose you are an expert in table analysis and your task is to generate the code to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、The final answer should follow the format below and ensure the first three code lines is exactly the same with the following code block: [Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show(). 
2、You only need to output the final code without any interpretation, make sure that your code can be run directly without any syntax errors.
3、Please do not read external files, and the pandas and matplotlib libraries have been successfully introduced.
4、Ensure that the X-axis used for drawing in the code is arranged in ascending alphabetical or numerical order. Ensure the last line in python code can only be "plt.show()", no other from. Give the final answer to the question directly without any explanation.
I will give you multiple table images and a question, please use the images to answer the question in specified format without any explanation: [Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()
"""
# 金融文档表格生成提示词
chart_generation_infer_prompt2 = """
# Role Play
Suppose you are an expert in financial annual report analysis, and your task is to generate executable Python code that answers the question using the content of financial annual report.

# Chain-of-thought
Let’s think step by step as follows and make the most of your strengths as a financial report table analysis expert:
1) Understand the question and extract the required information: metrics, company names, fiscal years, and calculation/comparison requirements.  
2) Carefully read the annual report tables, including the structure, row and column headers, units, and any summative rows or special flags.  
3) Select the most relevant rows and columns according to the question and locate the corresponding cells across years or companies.  
4) Perform the required operations (calculation, ranking, comparison, or aggregation) and prepare the data for plotting with pandas and matplotlib.  

# Output Control
1、The final answer should follow the format below and ensure the first three code lines is exactly the same with the following code block: [Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show(). 
2、You only need to output the final code without any interpretation, make sure that your code can be run directly without any syntax errors.
3、Please do not read external files, and the pandas and matplotlib libraries have been successfully introduced.
4、Ensure that the X-axis used for drawing in the code is arranged in ascending alphabetical or numerical order. Ensure the last line in python code can only be "plt.show()", no other from. Give the final answer to the question directly without any explanation. 

I will give you multiple annual report page images and a question; please use them to answer the question in the specified format without any explanation:
[Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()
"""


chart_direct = """
# Role Play
Suppose you are an expert in financial annual report analysis, and your task is to generate executable Python code that answers the question.

# Chain-of-thought
Let’s think step by step as follows:
1) Understand the question and extract the required information: metrics, company names, fiscal years, and calculation/comparison requirements.  
2) Based on the information explicitly provided in the question, construct the needed dataset (values, units, time axis/categories) for plotting.  
3) Perform the required operations (calculation, ranking, comparison, or aggregation) and prepare the data for plotting with pandas and matplotlib.  

# Output Control
1、The final answer should follow the format below and ensure the first three code lines is exactly the same with the following code block: [Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show(). 
2、You only need to output the final code without any interpretation, make sure that your code can be run directly without any syntax errors.
3、Please do not read external files, and the pandas and matplotlib libraries have been successfully introduced.
4、Ensure that the X-axis used for drawing in the code is arranged in ascending alphabetical or numerical order. Ensure the last line in python code can only be "plt.show()", no other form. Give the final answer to the question directly without any explanation. 

I will give you a question. Please generate the executable Python code to answer it in the specified format without any explanation:
[Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()
"""

table_generation_infer_prompt = """
# Role Play
Suppose you are a senior financial annual report analysis expert and your task is to answer quantitative analysis questions by producing clean, markdown-formatted tables.

# Task Description
You are given a question along with one or more evidence tables or textual excerpts from annual reports.  
Your task is to calculate the required financial indicators and output the results strictly as a markdown table.  
The markdown table must be formatted with clear column headers and aligned rows, so that the output can be directly rendered without additional editing.

# Reasoning Process
1. Read the question carefully and identify the target metrics, entities, and fiscal year(s).  
2. Extract the necessary numerical values from the provided evidence, making sure to interpret units, scales, and negative formatting correctly. If the value represents a percentage, append the "%" sign.
3. Perform any required calculations (ratios, margins, growth rates, turnovers, etc.), paying special attention to unit conversions (e.g., thousands vs. millions), and round results to the number of decimal places specified in the question.  
4. Construct a markdown table with appropriate headers (e.g., Item, Year, Company) and rows corresponding to each metric.  

# Output Control
1. The final output must be a markdown table only, no extra text or explanation.  
2. Use vertical bars (`|`) and dashes (`-`) to format the table so that it renders correctly in markdown.  
3. Ensure the column headers exactly match what is requested in the question (e.g., "Item, 2023").  
4. Round all numeric results as instructed in the question (default: four decimal places if not otherwise specified).  
5. The final output values must be **pure numbers without any units** (e.g., output `1,120.4500`, not `$1,120.4500 millions`). Percentages, however, must explicitly include the "%" sign.
6. Do not include any text before or after the markdown table.  

"""

# 推理模型使用的表格生成提示词备选模板
# table_generation_infer_prompt = """
# # Role Play
# Suppose you are a senior financial annual report analysis expert and your task is to answer quantitative analysis questions by producing clean, markdown-formatted tables.

# # Task Description
# You are given a question along with one or more evidence tables or textual excerpts from annual reports.  
# Your task is to calculate the required financial indicators and output the results strictly as a markdown table.  
# The markdown table must be formatted with clear column headers and aligned rows, so that the output can be directly rendered without additional editing.

# # Reasoning Process
# 1. Read the question carefully and identify the target metrics, entities, and fiscal year(s).  
# 2. Extract the necessary numerical values from the provided evidence, making sure to interpret units, scales, and negative formatting correctly. If the value represents a percentage, append the "%" sign.
# 3. Perform any required calculations (ratios, margins, growth rates, turnovers, etc.), paying special attention to unit conversions (e.g., thousands vs. millions), and round results to the number of decimal places specified in the question.  
# 4. Construct a markdown table with appropriate headers (e.g., Item, Year, Company) and rows corresponding to each metric.  

# # Output Control
# 1. The final output must be a markdown table only, no extra text or explanation.  
# 2. Use vertical bars (`|`) and dashes (`-`) to format the table so that it renders correctly in markdown.  
# 3. Ensure the column headers exactly match what is requested in the question (e.g., "Item, 2023").  
# 4. Round all numeric results as instructed in the question (default: four decimal places if not otherwise specified).  
# 5. The final output values must be **pure numbers without any units** (e.g., output `1,120.4500`, not `$1,120.4500 millions`). Percentages, however, must explicitly include the "%" sign.
# 6. Do not include any text before or after the markdown table.  

# # Output format (markdown table only, no extra text):

# ```markdown
# | Item | 2020 | 2021 |
# |------|------|------|
# | ...  | ...  | ...  |
# ```
# The output MUST be enclosed in a fenced code block using ```markdown and ```.
# Do NOT include any text before or after the code block.
# """

# 标注时用
table_generation_infer_prompt2 = """
# Role Play
Suppose you are a senior financial annual report analysis expert and your task is to answer quantitative analysis questions by producing clean, markdown-formatted tables.

# Task Description
You are given a question along with one or more evidence tables or textual excerpts from annual reports.  
Your task is to calculate the required financial indicators and output the results strictly as a markdown table.  
The markdown table must be formatted with clear column headers and aligned rows, so that the output can be directly rendered without additional editing.

# Reasoning Process
1. Read the question carefully and identify the target metrics, entities, and fiscal year(s).  
2. Extract the necessary numerical values from the provided evidence, making sure to interpret units, scales, and negative formatting correctly.  
3. Perform any required calculations (ratios, margins, growth rates, turnovers, etc.), paying special attention to unit conversions (e.g., thousands vs. millions), and round results to the number of decimal places specified in the question.  
4. Construct a markdown table with appropriate headers (e.g., Item, Year, Company) and rows corresponding to each metric.  

# Output Control
1. Output TWO parts in order:
   (a) **Reasoning steps** in the exact bracketed format:
       [Reasoning steps for this question are as following: 1.First, we need to... 2.Then, we need to...]
       Keep steps concise and numbered until the answer is derived.
   (b) **Markdown table** on the next lines.
2. The final output must be a markdown table only, no extra text or explanation.  
3. Use vertical bars (`|`) and dashes (`-`) to format the table so that it renders correctly in markdown.  
4. Ensure the column headers exactly match what is requested in the question (e.g., "Item, 2023").  
5. Round all numeric results as instructed in the question (default: four decimal places if not otherwise specified).  
6. The final output values must be **pure numbers without any units** (e.g., output `1,120.4500`, not `$1,120.4500 millions`).  
7. Do not include any text before or after the markdown table.  


# Question
The question is:
"""

table_direct = """
# Role Play
Suppose you are a senior financial analysis expert and your task is to answer quantitative analysis questions by producing clean, markdown-formatted tables.

# Task Description
You are given a question.  
Your task is to calculate the required financial indicators and output the results strictly as a markdown table.  
The markdown table must be formatted with clear column headers and aligned rows, so that the output can be directly rendered without additional editing.

# Reasoning Process
1. Read the question carefully and identify the target metrics, entities, and fiscal year(s).  
2. Use only the numerical values explicitly provided in the question (if any) and standard financial reasoning to obtain the needed quantities. Interpret units, scales, and negative formatting correctly. If the value represents a percentage, append the "%" sign.
3. Perform any required calculations (ratios, margins, growth rates, turnovers, etc.), paying special attention to unit conversions (e.g., thousands vs. millions), and round results to the number of decimal places specified in the question.  
4. Construct a markdown table with appropriate headers (e.g., Item, Year, Company) and rows corresponding to each metric.  

# Output Control
1. The final output must be a markdown table only, no extra text or explanation.  
2. Use vertical bars (`|`) and dashes (`-`) to format the table so that it renders correctly in markdown.  
3. Ensure the column headers exactly match what is requested in the question (e.g., "Item, 2023").  
4. Round all numeric results as instructed in the question (default: four decimal places if not otherwise specified).  
5. The final output values must be pure numbers without any units (e.g., output `1,120.4500`, not `$1,120.4500 millions`). Percentages, however, must explicitly include the "%" sign.
6. Do not include any text before or after the markdown table.  

"""


multimodal_fact_verify_infer_prompt = """
You are an AI assistant for Fact Verification.

Task:
- Your role is to verify whether the claim expressed in the FIRST image (the claim image) is factually correct.
- The first image in the input list is ALWAYS the claim image.
- The following images are retrieved evidence pages (e.g., annual reports, financial statements) that you must examine and use as supporting or refuting evidence.

Instructions:
1. Compare the content of the claim image against the evidence images.
2. Pay special attention to chart types (PieChart, LineChart, BarChart, ScatterChart). Verify that the chart correctly represents the claimed indicators (e.g., values, proportions, trends).
3. Decide the fact verification label. The answer must be one of the three categories:
   - Supported: if the claim is fully consistent with the evidence.
   - Refuted: if the claim is contradicted by the evidence.
   - Insufficient: if the evidence is not enough to verify or refute the claim.
4. Provide a clear explanation of your reasoning.

Output format (strict JSON-like structure):
{
  "answer": "<Supported / Refuted / Insufficient>",
  "explanation": "<short reasoning>",
}
"""

fact_verify_infer_prompt = """
# Role Play
Suppose you are a senior financial annual report analysis expert and your task is to answer fact verification questions based on evidence from annual reports.

# Task Description
You are given a statement and one or more evidence excerpts from financial annual reports.  
Your task is to decide whether the statement is:
- "Supported": fully confirmed by the provided evidence,  
- "Refuted": contradicted by the provided evidence,  
- "Insufficient": the provided evidence is not enough to confirm or refute the statement.  

You must also provide a short explanation that clearly summarizes the reasoning behind your decision.  
The explanation should cite the relevant metrics, years, companies, or textual information from the evidence.

# Reasoning Process
1. Read the statement carefully and identify the key metric(s), company name(s), year(s), and claim being made.  
2. Examine the provided evidence excerpts (tables or text) to see if the required information is present.  
3. Compare the statement against the evidence:  
   - If the evidence confirms all aspects, label as "Supported".  
   - If the evidence contradicts, label as "Refuted".  
   - If the evidence is incomplete or does not cover all aspects, label as "Insufficient".  
4. Summarize the reasoning in a concise explanation.  

# Output Control
Please generate the output in strict JSON format:
{
  "answer": "Supported or Refuted or Insufficient",
  "explanation": "Your concise reasoning here, referring to the evidence"
}
"""

fact_direct = """
# Role Play
Suppose you are a senior financial analysis expert and your task is to answer fact verification questions.

# Task Description
You are given a statement.  
Your task is to decide whether the statement is:
- "Supported": the statement is consistent with well-known financial facts or can be reasonably verified based on general financial knowledge,  
- "Refuted": the statement contradicts well-known financial facts,  
- "Insufficient": the statement cannot be confirmed or refuted without additional specific evidence.  

You must also provide a short explanation that clearly summarizes the reasoning behind your decision.

# Reasoning Process
1. Read the statement carefully and identify the key metric(s), company name(s), year(s), and claim being made.  
2. Determine whether the claim can be verified using general financial knowledge or logical reasoning alone.  
3. Classify the statement as:
   - "Supported" if it aligns with known facts or can be reasonably verified,  
   - "Refuted" if it clearly contradicts known facts,  
   - "Insufficient" if the claim depends on specific undisclosed data or documents.  
4. Summarize the reasoning in a concise explanation.

# Output Control
Please generate the output in strict JSON format:
{
  "answer": "Supported or Refuted or Insufficient",
  "explanation": "Your concise reasoning here"
}

"""


# 用于测试时回答
text_reasoning_infer_prompt1 = """
# Role
You are a professional financial analysis assistant. Your task is to answer textual reasoning questions strictly based on retrieved evidence images from company annual reports.

# Input
The images are retrieved evidence pages from annual reports. They may contain management discussions, strategic outlooks, ESG disclosures, risk factor narratives, or other narrative sections. Treat these images as the only valid sources of information for answering the question.

# Task
1. Carefully read the question and identify the key entities, years, and themes.
2. Extract and interpret the relevant passages directly from the retrieved evidence images.
3. Compare, synthesize, or trace changes across years or companies if multiple images are provided.
4. Provide a reasoned, well-structured answer in natural language.

# Answer Requirements
- Base your answer only on the content visible in the retrieved evidence images; do not use outside knowledge or assumptions.
- Provide ONE combined long-form answer that naturally weaves together the reasoning process and the final conclusion.
- If the retrieved evidence is insufficient, explicitly write: “Insufficient evidence.”

# Answer Format
Answer:
[A coherent paragraph or a few short paragraphs combining reasoning and conclusion into one long-form response.]

"""

# 用于标注时回答
text_reasoning_infer_prompt2 = """
# Role
You are a professional financial analysis assistant. Your task is to answer textual reasoning questions strictly based on retrieved evidence images from company annual reports.

# Input
The images are retrieved evidence pages from annual reports. They may contain management discussions, strategic outlooks, ESG disclosures, risk factor narratives, or other narrative sections. Treat these images as the only valid sources of information for answering the question.

# Task
1. Carefully read the question and identify the key entities, years, and themes.
2. Extract and interpret the relevant passages directly from the retrieved evidence images.
3. Compare, synthesize, or trace changes across years or companies if multiple images are provided.
4. Provide a reasoned, well-structured answer in natural language.

# Answer Requirements
- Base both answers only on the content visible in the retrieved evidence images; do not use outside knowledge or assumptions.
- Long Answer: Provide ONE combined long-form answer that naturally weaves together the reasoning process and the final conclusion.
- Short Answer: should be a brief, clear, and direct summary (2–4 sentences).
- If the retrieved evidence is insufficient, explicitly write “Insufficient evidence” for both.

# Answer Format
Long Answer:
[Detailed reasoning and conclusion in one long-form response]

Short Answer:
[Concise summary of the conclusion in a few sentences]

# Question
The question is:
"""

text_direct = """
# Role
You are a professional financial analysis assistant. Your task is to answer textual reasoning questions.

# Input
You are given a question. No external documents, images, or retrieved evidence are provided.

# Task
1. Carefully read the question and identify the key entities, years, and themes.
2. Use general financial knowledge and logical reasoning to address the question.
3. Compare, synthesize, or describe trends across years or companies if required by the question.
4. Provide a reasoned, well-structured answer in natural language.

# Answer Requirements
- Do not assume access to any annual report text, images, or retrieved evidence.
- Do not cite or reference specific documents.
- Provide ONE combined long-form answer that naturally weaves together the reasoning process and the final conclusion.
- If the question cannot be reasonably answered without specific report-level evidence, explicitly write: “Insufficient information.”

# Answer Format
Answer:
[A coherent paragraph or a few short paragraphs combining reasoning and conclusion into one long-form response.]
"""


type_judge_prompt_old = """
Given a question and corresponding evidence page, classify the type of evidence and return the result in a JSON format.

Classification rules:
- If the evidence contains only text, return "evidence_type": "text".
- If the evidence contains only a table, return "evidence_type": "table".
- If the evidence contains only a chart, return "evidence_type": "chart".
- If the evidence contains a combination of any two modalities (text + table, text + chart, or table + chart), return "evidence_type": "hybrid".

Please provide a reasoning for your classification, explaining why the evidence falls into the selected category.

JSON output format:
{
  "evidence_type": "text|table|chart|hybrid",
  "reason": "Explanation of the classification"
}
"""

type_judge_prompt = """
Given a question and corresponding evidence pages, classify the type of each evidence page and return the result in a JSON format.

Classification rules:
- If the evidence contains a chart, classify it as "chart".
- If the evidence contains a table, classify it as "table".
- If the evidence contains only text, classify it as "text".

The output should list the types of each evidence page. Each page should be classified independently, and if a page contains more than one modality (e.g., text + chart, text + table), it should still only be classified based on the primary modality (chart or table if present).

JSON output format:
{
  "evidence_types": ["text", "table", "chart"],  // List of evidence types for each evidence page
  "reasons": ["Explanation for each classification"]  // Corresponding reasons for each evidence type
}
"""
