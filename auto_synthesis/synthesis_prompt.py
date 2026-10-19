# four types of numerical_reasoning + tool call
numerical_reasoning_syn_prompt = """
# Role play
Suppose you are a senior annual report analysis expert and your task is to generate high-quality and diverse numerical reasoning questions that can only be answered through retrieval and synthesis across multiple pages or documents of company financial reports.

# Task Descriptions
Numerical reasoning tasks in financial report analysis involve interpreting and calculating with numerical data found in structured report excerpts, such as income statements, balance sheets, and KPIs across different companies and fiscal years. These tasks typically require retrieving relevant values from multiple report sections, recognizing trends, comparing performance, or evaluating quantitative shifts across time or entities. Each question must require multi-source retrieval to answer, simulating a real-world retrieval-augmented analysis workflow.

The input consists of one or more images from official annual reports. These images may represent different pages from the same or different companies, and may span multiple fiscal years. Based on the type of input images, design questions that match the appropriate comparison mode — use cross-company horizontal comparisons when images are from different companies, and cross-year vertical analyses when images span multiple fiscal years of the same company.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial textual or tabular data, give due consideration to the following aspects:
1.	Complexity: Include a range of problem complexities. Simpler questions may involve a direct comparison or percentage difference. More complex questions should require multi-hop reasoning that spans across multiple documents, companies, years, and logically connected metrics — for example, identifying the company with the highest value in one metric, then retrieving another metric for that specific entity (but still be expressed as a single complete sentence).
2.	Length: Questions should be between 20 and 50 words. Keep language compact but fully informative.
3.	Diversity: The subtypes covered by the questions should include: Multi-hop Numerical Reasoning, Ranking, Comparison, and Calculation(Numerical Calculation and Time-based Calculation).
4.	Real-World Relevance: Questions must reflect the type of financial inquiry a professional analyst would perform when reviewing annual reports — such as identifying revenue leaders, cost-saving trends, or changes in capital structure.
5.	Writing-Style: Use professional, clear, and consistent financial terminology. Avoid vague wording. Specify the target metric, entities, or years clearly.
6.	Question Form: The question must be phrased as a single, grammatically complete sentence, even if it involves multi-hop reasoning. Do not split the question into multiple sub-questions.
7.	Context-Independent Question: Each question must be fully self-contained and context-independent. The question must explicitly mention all necessary information, including the specific metric name, company name or entity names, and fiscal year(s), so it can be understood and answered in isolation within a retrieval-augmented setting.(e.g., a question that specifies the metric, such as revenue; the companies involved, such as Company A, B, and C; and the time frame, such as fiscal years 2021 to 2023)
8.	Answer Control: Answers must follow the format “AnswerName” or “AnswerName1, AnswerName2…”. “AnswerName” should be a numeric value or company/entity name — short and unambiguous.
9.	Retrieval Dependency: Ensure that each question requires combining information from multiple companies, multiple years, and multiple report pages. The answer must depend on retrieval; it must not be answerable using only a single data point or page.

# Output Control
Please generate each output as a strict JSON object in the following format:
{
    "result":[
        {
          "question": "Your question here (cross-year, cross-company, reasoning-heavy), and you must specify entity names such as company name, year, and metric",
          "answer": "The answer of the question must follow the format AnswerName or AnswerName1, AnswerName2…",
          "subtype": "The subtype of the question should be one of Multi-hop Numerical Reasoning, Ranking, Comparison, Calculation.",       
          "evidence_pages": [" ", " ", " "]
        },
        {
          "question": "question 2",
          "answer": " ",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        },
        {
          "question": "question 3",
          "answer": " ",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        },
    ]
}
"""

numerical_reasoning_syn_prompt_text = """
# Role play
Suppose you are a senior annual report analysis expert and your task is to generate high-quality and diverse numerical reasoning questions that can only be answered through retrieval and synthesis across multiple pages or documents of company financial reports.

# Task Descriptions
Numerical reasoning tasks in financial report analysis involve interpreting and calculating with numerical data found in structured report excerpts, such as income statements, balance sheets, and KPIs across different companies and fiscal years. These tasks typically require retrieving relevant values from multiple report sections, recognizing trends, comparing performance, or evaluating quantitative shifts across time or entities. Each question must require multi-source retrieval to answer, simulating a real-world retrieval-augmented analysis workflow.

The input consists of OCR-extracted texts from one or more pages of official annual reports. These texts may represent different pages from the same or different companies, and may span multiple fiscal years. Based on the type of input texts, design questions that match the appropriate comparison mode — use cross-company horizontal comparisons when the texts are from different companies, and cross-year vertical analyses when they span multiple fiscal years of the same company.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial textual or tabular data, give due consideration to the following aspects:
1. Complexity: Include a range of problem complexities. Simpler questions may involve a direct comparison or percentage difference. More complex questions should require multi-hop reasoning that spans across multiple documents, companies, years, and logically connected metrics — for example, identifying the company with the highest value in one metric, then retrieving another metric for that specific entity (but still be expressed as a single complete sentence).
2. Length: Questions should be between 20 and 50 words. Keep language compact but fully informative.
3. Diversity: The subtypes covered by the questions should include: Multi-hop Numerical Reasoning, Ranking, Comparison, and Calculation (Numerical Calculation and Time-based Calculation).
4. Real-World Relevance: Questions must reflect the type of financial inquiry a professional analyst would perform when reviewing annual reports — such as identifying revenue leaders, cost-saving trends, or changes in capital structure.
5. Writing-Style: Use professional, clear, and consistent financial terminology. Avoid vague wording. Specify the target metric, entities, or years clearly.
6. Question Form: The question must be phrased as a single, grammatically complete sentence, even if it involves multi-hop reasoning. Do not split the question into multiple sub-questions.
7. Context-Independent Question: Each question must be fully self-contained and context-independent. The question must explicitly mention all necessary information, including the specific metric name, company name or entity names, and fiscal year(s), so it can be understood and answered in isolation within a retrieval-augmented setting.
8. Answer Control: Answers must follow the format “AnswerName” or “AnswerName1, AnswerName2…”. “AnswerName” should be a numeric value or company/entity name — short and unambiguous.
9. Retrieval Dependency: Ensure that each question requires combining information from multiple companies, multiple years, and multiple report pages. The answer must depend on retrieval; it must not be answerable using only a single data point or page.

# Input Texts
Below are the OCR-extracted texts from the selected report pages. Use these as the factual basis for question generation:
{text_block}

# Output Control
Please generate each output as a strict JSON object in the following format:
{{
    "result":[
        {{
          "question": "Your question here (cross-year, cross-company, reasoning-heavy), and you must specify entity names such as company name, year, and metric",
          "answer": "The answer of the question must follow the format AnswerName or AnswerName1, AnswerName2…",
          "subtype": "The subtype of the question should be one of Multi-hop Numerical Reasoning, Ranking, Comparison, Calculation.",
          "evidence_pages": [" ", " ", " "]
        }},
        {{
          "question": "question 2",
          "answer": " ",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        }},
        {{
          "question": "question 3",
          "answer": " ",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        }}
    ]
}}
"""

numerical_reasoning_syn_prompt_easy = """
# Role play
Suppose you are a senior annual report analysis expert and your task is to generate high-quality and diverse numerical reasoning questions that can only be answered through retrieval and synthesis across multiple pages or documents of company financial reports.

# Task Descriptions
Numerical reasoning tasks in financial report analysis involve interpreting and calculating with numerical data found in structured report excerpts, such as income statements, balance sheets, and KPIs across different companies and fiscal years. These tasks typically require retrieving relevant values from multiple report sections, recognizing trends, comparing performance, or evaluating quantitative shifts across time or entities. Each question must require multi-source retrieval to answer, simulating a real-world retrieval-augmented analysis workflow.

The input consists of one or more images from official annual reports. These images may represent different pages from the same or different companies, and may span multiple fiscal years. Based on the type of input images, design questions that match the appropriate comparison mode — use cross-company horizontal comparisons when images are from different companies, and cross-year vertical analyses when images span multiple fiscal years of the same company.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial textual or tabular data, give due consideration to the following aspects:
1.	Complexity: Include a range of problem complexities. Simpler questions may involve a direct comparison or percentage difference. More complex questions should require multi-hop reasoning that spans across multiple documents, companies, years, and logically connected metrics — for example, identifying the company with the highest value in one metric, then retrieving another metric for that specific entity (but still be expressed as a single complete sentence).
2.	Length: Questions should be between 20 and 50 words. Keep language compact but fully informative.
3.	Diversity: The subtypes covered by the questions should include: Multi-hop Numerical Reasoning, Ranking, Comparison, and Calculation(Numerical Calculation and Time-based Calculation).
4.	Real-World Relevance: Questions must reflect the type of financial inquiry a professional analyst would perform when reviewing annual reports — such as identifying revenue leaders, cost-saving trends, or changes in capital structure.
5.	Writing-Style: Use professional, clear, and consistent financial terminology. Avoid vague wording. Specify the target metric, entities, or years clearly.
6.	Question Form: The question must be phrased as a single, grammatically complete sentence, even if it involves multi-hop reasoning. Do not split the question into multiple sub-questions.
7.	Context-Independent Question: Each question must be fully self-contained and context-independent. The question must explicitly mention all necessary information, including the specific metric name, company name or entity names, and fiscal year(s), so it can be understood and answered in isolation within a retrieval-augmented setting.(e.g., a question that specifies the metric, such as revenue; the companies involved, such as Company A, B, and C; and the time frame, such as fiscal years 2021 to 2023)
8.	Answer Control: Answers must follow the format “AnswerName”. “AnswerName” should be a single numeric value OR a single company/entity name — short and unambiguous. Do not include multiple fields in the same answer.Only output multiple AnswerNames when the task type is Ranking.
9.	Retrieval Dependency: Ensure that each question requires combining information from multiple companies, multiple years, and multiple report pages. The answer must depend on retrieval; it must not be answerable using only a single data point or page.

# Output Control
Please generate each output as a strict JSON object in the following format:
{
    "result":[
        {
          "question": "Your question here (cross-year, cross-company, reasoning-heavy)",
          "answer": "The answer must follow the format AnswerName (a single numeric value or entity name).",
          "subtype": "The subtype of the question should be one of Multi-hop Numerical Reasoning, Ranking, Comparison, Calculation.",       
          "evidence_pages": [" ", " ", " "]
        },
        {
          "question": "question 2",
          "answer": " ",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        },
        {
          "question": "question 3",
          "answer": " ",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        },
    ]
}
"""

tool_call_syn_prompt = """
"""

# four types of chart generation
chart_generation_syn_prompt = """
# Role play
Suppose you are a senior financial visualization expert and your task is to generate high-quality and diverse chart generation questions based on financial report excerpts (tabular or textual data) that require transforming structured tables into meaningful visualizations.

# Task Descriptions
Chart generation in financial report analysis involves selecting the most appropriate visualization to clearly present relationships, trends, comparisons, and distributions of financial metrics. These tasks typically require identifying the relevant metrics from multiple company reports or fiscal years, determining the analysis objective (such as revenue growth, cost breakdown, or market share comparison), and specifying the correct chart type (e.g., line chart, bar chart, pie chart, scatter plot). The purpose is to simplify complex tabular data, enhance interpretability, and provide clear insights for decision-making. Each generated question must explicitly specify what type of chart should be drawn, the metric(s) to be visualized, and the entities/years to be compared.

The input consists of one or more tables or text blocks extracted from annual reports, potentially spanning multiple companies and multiple fiscal years. Based on the type of input, design visualization questions that require meaningful financial analysis — use cross-company comparisons when the input involves multiple companies, and cross-year analyses when it spans multiple fiscal years of the same company.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial tabular data, give due consideration to the following aspects:
1. Complexity: Include a range of complexities. Simple tasks may require plotting a single metric as a line or bar chart. More complex tasks should require combining multiple metrics or entities, such as plotting operating expenses and net income over time, or visualizing market share composition.
2. Length: Each question should be between 20 and 50 words, phrased concisely but fully informative.
3. Diversity: The subtypes covered by the questions should include LineChart Generation, BarChart Generation, ScatterChart Generation, and PieChart Generation.
4. Real-World Relevance: Questions must reflect actual financial analysis needs, such as revenue trends across years, cost structure composition, profitability comparisons, or cross-company performance benchmarking.
5. Writing-Style: Use professional, clear, and consistent financial terminology. Avoid vague expressions. Specify chart type, metric(s), entity names, and timeframes explicitly in each question.
6. Question Form: Each question must be phrased as a single, grammatically complete sentence. Do not split into multiple sub-questions.
7. Context-Independent Question: Each question must be self-contained, explicitly mentioning all necessary details (metric, company, fiscal years, chart type) so it can be understood without external context.
8. Retrieval Dependency: Ensure that each question requires referencing multiple companies, multiple years, or multiple metrics — no single-cell or trivial table lookups.

# Output Control
Please generate each output as a strict JSON object in the following format:
{
    "result":[
        {
        "question": "Your chart generation question here (cross-year or cross-company, visualization-heavy)",
        "subtype": "The subtype of the question should be one of LineChart Generation, BarChart Generation, ScatterChart Generation, PieChart Generation",
        "evidence_pages": [" ", " ", " "]
        },
        {
        "question": "question 2",
        "subtype": " ",
        "evidence_pages": [" ", " ", " "]
        },
        {
        "question": "question 3",
        "subtype": " ",
        "evidence_pages": [" ", " ", " "]
        }
    ]
}
"""

chart_generation_syn_prompt_text = """
# Role play
Suppose you are a senior financial visualization expert and your task is to generate high-quality and diverse chart generation questions based on financial report excerpts (tabular or textual data) that require transforming structured tables into meaningful visualizations.

# Task Descriptions
Chart generation in financial report analysis involves selecting the most appropriate visualization to clearly present relationships, trends, comparisons, and distributions of financial metrics. These tasks typically require identifying the relevant metrics from multiple company reports or fiscal years, determining the analysis objective (such as revenue growth, cost breakdown, or market share comparison), and specifying the correct chart type (e.g., line chart, bar chart, pie chart, scatter plot). The purpose is to simplify complex tabular data, enhance interpretability, and provide clear insights for decision-making. Each generated question must explicitly specify what type of chart should be drawn, the metric(s) to be visualized, and the entities/years to be compared.

The input consists of OCR-extracted tables or text blocks from one or more pages of official annual reports, potentially spanning multiple companies and multiple fiscal years. Based on the type of input texts, design visualization questions that require meaningful financial analysis — use cross-company comparisons when the input involves multiple companies, and cross-year analyses when it spans multiple fiscal years of the same company.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial tabular data, give due consideration to the following aspects:
1. Complexity: Include a range of complexities. Simple tasks may require plotting a single metric as a line or bar chart. More complex tasks should require combining multiple metrics or entities, such as plotting operating expenses and net income over time, or visualizing market share composition.
2. Length: Each question should be between 20 and 50 words, phrased concisely but fully informative.
3. Diversity: The subtypes covered by the questions should include LineChart Generation, BarChart Generation, ScatterChart Generation, and PieChart Generation.
4. Real-World Relevance: Questions must reflect actual financial analysis needs, such as revenue trends across years, cost structure composition, profitability comparisons, or cross-company performance benchmarking.
5. Writing-Style: Use professional, clear, and consistent financial terminology. Avoid vague expressions. Specify chart type, metric(s), entity names, and timeframes explicitly in each question.
6. Question Form: Each question must be phrased as a single, grammatically complete sentence. Do not split into multiple sub-questions.
7. Context-Independent Question: Each question must be self-contained, explicitly mentioning all necessary details (metric, company, fiscal years, chart type) so it can be understood without external context.
8. Retrieval Dependency: Ensure that each question requires referencing multiple companies, multiple years, or multiple metrics — no single-cell or trivial table lookups.

# Input Texts
Below are the OCR-extracted tables or text blocks from the selected report pages. Use these as the factual basis for chart generation question construction:
{text_block}

# Output Control
Please generate each output as a strict JSON object in the following format:
{{
    "result":[
        {{
          "question": "Your chart generation question here (cross-year or cross-company, visualization-heavy)",
          "subtype": "The subtype of the question should be one of LineChart Generation, BarChart Generation, ScatterChart Generation, PieChart Generation",
          "evidence_pages": [" ", " ", " "]
        }},
        {{
          "question": "question 2",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        }},
        {{
          "question": "question 3",
          "subtype": " ",
          "evidence_pages": [" ", " ", " "]
        }}
    ]
}}
"""


# another four types of chart generation
chart_generation_syn_prompt2 = """
# Role play
Suppose you are a senior financial visualization expert and your task is to generate high-quality and diverse chart generation questions based on financial report excerpts (tabular or textual data) that require transforming structured tables into meaningful visualizations.

# Task Descriptions
Chart generation in financial report analysis involves selecting the most appropriate visualization to clearly present relationships, trends, comparisons, and distributions of financial metrics. These tasks typically require identifying the relevant metrics from multiple company reports or fiscal years, determining the analysis objective (such as revenue growth, cost breakdown, or market share comparison), and specifying the correct chart type (e.g., line chart, bar chart, pie chart, scatter plot). The purpose is to simplify complex tabular data, enhance interpretability, and provide clear insights for decision-making. Each generated question must explicitly specify what type of chart should be drawn, the metric(s) to be visualized, and the entities/years to be compared.

The input consists of one or more tables or text blocks extracted from annual reports, potentially spanning multiple companies and multiple fiscal years. Based on the type of input, design visualization questions that require meaningful financial analysis — use cross-company comparisons when the input involves multiple companies, and cross-year analyses when it spans multiple fiscal years of the same company.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial tabular data, give due consideration to the following aspects:
1. Complexity: Include a range of complexities. Simple tasks may require plotting a single metric as a line or bar chart. More complex tasks should require combining multiple metrics or entities, such as plotting operating expenses and net income over time, or visualizing market share composition.
2. Length: Each question should be between 20 and 50 words, phrased concisely but fully informative.
3. Diversity: The subtypes covered by the questions should include BubbleChart Generation, RadarChart Generation, AreaChart Generation, and StackedBarChart Generation.
4. Real-World Relevance: Questions must reflect actual financial analysis needs, such as revenue trends across years, cost structure composition, profitability comparisons, or cross-company performance benchmarking.
5. Writing-Style: Use professional, clear, and consistent financial terminology. Avoid vague expressions. Specify chart type, metric(s), entity names, and timeframes explicitly in each question.
6. Question Form: Each question must be phrased as a single, grammatically complete sentence. Do not split into multiple sub-questions.
7. Context-Independent Question: Each question must be self-contained, explicitly mentioning all necessary details (metric, company, fiscal years, chart type) so it can be understood without external context.
8. Retrieval Dependency: Ensure that each question requires referencing multiple companies, multiple years, or multiple metrics — no single-cell or trivial table lookups.
9. **Axis/Encoding Specification**: If applicable, the question should clearly indicate how the visualization is structured, such as specifying the X-axis, Y-axis, bubble size, or stacked dimensions when these details are necessary for clarity.



# Output Control
Please generate each output as a strict JSON object in the following format:
{
    "result":[
        {
        "question": "Your chart generation question here (cross-year or cross-company, visualization-heavy)",
        "subtype": "The subtype of the question should be one of  BubbleChart Generation, RadarChart Generation, AreaChart Generation, StackedBarChart Generation",
        "evidence_pages": [" ", " ", " "]
        },
        {
        "question": "question 2",
        "subtype": " ",
        "evidence_pages": [" ", " ", " "]
        },
        {
        "question": "question 3",
        "subtype": " ",
        "evidence_pages": [" ", " ", " "]
        }
    ]
}
"""

# 提取类型
# Extract the company’s [revenue, cost of revenue, net income, cash and cash equivalents, accounts receivable, accounts payable, total assets, total liabilities, and net cash flow from operating activities, ...] in 2022 and 2023 from the attached data. Output the results in a markdown-formatted table. Use ’Item’, ’2022’, and ’2023’ as the column headers.
table_generation_syn_prompt_text1 = """
# Role
You are a senior financial analysis expert. Your task is to generate simple table-generation questions that require extracting multiple financial statement items (2–4 items) for the years {year} using the provided annual report data.

# Task Description
You will receive structured financial data extracted from annual report pages (OCR text or tables). Based on this data, generate a question that asks the model to extract several financial statement line items. These items may include but are not limited to revenue, cost of revenue, gross profit, operating income, net income, cash and cash equivalents, accounts receivable, inventories, accounts payable, total assets, total liabilities, total equity, and net cash flow from operating activities.

# Question Construction Requirements
1. Each question must require extracting **2 to 4 financial items** for the years **{year}**.
2. The question must strictly follow this natural-language template structure:

  "Extract the {company_name}’s [financial_item1, financial_item2, ...] in {year} from the attached data. Output the results in a markdown-formatted table. Use 'Item' and {year} as the column headers."

3. Items can be chosen from the provided list or any valid financial statement line item derivable from typical annual report data.
  The provided list: [revenue, cost of revenue, net income, cash and cash equivalents, accounts receivable, accounts payable, total assets, total liabilities, net cash flow from operating activities, ...]

4. The question must be self-contained and fully understandable without external context.

5. The question must require extraction of multiple values from the attached OCR-extracted financial data.

# Input Texts
Below are the OCR-extracted financial tables or text blocks from selected report pages. Use these as the factual basis for generating the question:
{text_block}

# Output Format
Return the final result as a strict JSON object:
{{
    "result": [
        {{
            "question": "Your generated question1 here",
            "items": ["item1", "item2", "..."],
            "evidence_pages": [" ", " ", " "]
        }},
        {{
            "question": "Your generated question2 here",
            "items": ["item1", "item2", "..."],
            "evidence_pages": [" ", " ", " "]
        }},
        {{
            "question": "Your generated question3 here",
            "items": ["item1", "item2", "..."],
            "evidence_pages": [" ", " ", " "]
        }},
    ]
}}
"""

# 计算类问题的合成提示词
# The provided list:[return on equity, return on assets, gross margin, net profit margin, revenue growth rate, net profit growth rate, debt to assets, debt to equity, equity to assets, current ratio, quick ratio, inventory turnover, receivables turnover, ...]
table_generation_syn_prompt_text2 = """
# Role
You are a senior financial analysis expert. Your task is to generate simple table-generation questions that require calculating multiple financial indicators (2–4 metrics) for the year {year} using the provided annual report data.

# Task Description
You will receive structured financial data extracted from annual report pages (OCR text or tables). Based on this data, generate a question that asks the model to compute several financial ratios or financial performance metrics. These metrics may include but are not limited to standard indicators such as return on equity, return on assets, gross margin, net profit margin, revenue growth rate, net profit growth rate, debt to assets, debt to equity, equity to assets, current ratio, quick ratio, inventory turnover, receivables turnover, and any other financially meaningful ratio that can be computed from the provided data.

# Question Construction Requirements
1. Each question must require calculating **2 to 4 financial metrics** for the year **{year}**.
2. The question must strictly follow this natural-language template structure:

  "Calculate the {company_name}’s [financial_metric1, financial_metric2, ...] in {year} given the attached data. Output the results in a markdown-formatted table. Use 'Item' and {year} as the column headers. Express the result as a decimal, rounded to four decimal places."

3. Metrics can be chosen from the provided list or any valid financial ratio derivable from typical annual report data.
  The provided list:[return on equity, return on assets, gross margin, net profit margin, revenue growth rate, net profit growth rate, debt to assets, debt to equity, equity to assets, current ratio, quick ratio, inventory turnover, receivables turnover, ...]
4. The question must be self-contained and fully understandable without external context.
5. The question must require computation using multiple values from the attached OCR-extracted financial data.

# Input Texts
Below are the OCR-extracted financial tables or text blocks from selected report pages. Use these as the factual basis for generating the question:
{text_block}

# Output Format
Return the final result as a strict JSON object:
{{
    "result": [
        {{
            "question": "Your generated question1 here",
            "metrics": ["metric1", "metric2", "..."],
            "evidence_pages": [" ", " ", " "]
        }},
        {{
            "question": "Your generated question2 here",
            "metrics": ["metric1", "metric2", "..."],
            "evidence_pages": [" ", " ", " "]
        }},
        {{
            "question": "Your generated question3 here",
            "metrics": ["metric1", "metric2", "..."],
            "evidence_pages": [" ", " ", " "]
        }},        
    ]
}}
"""

# three types of fact verify
multimodal_fact_verify_syn_prompt = """
"""

fact_verify_syn_prompt_easy = """
# Role Play
Suppose you are a senior financial annual report analysis expert and your task is to generate high-quality fact verification question–answer pairs based on annual report tables and textual excerpts.

# Task Description
Fact verification tasks in financial report analysis involve checking whether a given natural language statement is consistent with the information contained in annual reports. Each statement should be specific to certain metrics, company names, fiscal years, or textual disclosures. The answer must be one of three categories:
- "Supported": the statement is fully confirmed by the evidence in the annual report.
- "Refuted": the statement contradicts the evidence in the annual report.
- "Insufficient": the evidence in the annual report is not enough to judge the statement as true or false.

The input consists of one or more tables or text excerpts from annual reports. The generated statements must require retrieval of relevant information across these inputs, and the correctness must be determinable only with reference to the provided evidence.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial tabular data, give due consideration to the following aspects:
1. Complexity: Include both simple checks (direct extraction of a metric or textual fact) and more complex reasoning (calculations, cross-year comparisons, or multi-step textual reasoning).  
2. Length: Each statement should be between 15 and 40 words.  
3. Diversity: Cover two distinct types of fact verification tasks, (depending on whether the provided annual report pages are text, tables, or a mix of both):
   - **Numerical Verification**: tatements that involve verifying financial indicators, which may come from tables or from textual descriptions within the annual report, either by direct extraction or through more complex calculations.
   - **Textual Verification**: statements that involve verifying qualitative or narrative information from any part of the annual report text (including but not limited to strategic goals, risk disclosures, management discussion, or general narrative sections), requiring simple fact checks or deeper reasoning.
4. Real-World Relevance: Statements must reflect realistic checks that analysts or auditors would perform.  
5. Writing Style: Use professional and precise financial terminology. Explicitly mention metric, company, fiscal year, or textual disclosure subject.  
6. Balance: Generate statements that distribute across Supported, Refuted, and Insufficient. Ensure Refuted and Insufficient are realistic and plausible.  
7. Context-Independent: Each statement must be understandable on its own without external context.  

# Output Control
Please generate the output as a strict JSON object in the following format:
{
  "result":[
    {
      "question": "Your fact verification statement here",
      "answer": "Supported or Refuted or Insufficient",
      "explanation": "A brief explanation of why the answer is Supported, Refuted, or Insufficient",
      "subtype": "The subtype of the question should be one of Numerical Fact Verify or Textual Fact Verify",
      "evidence_pages": [" ", " "]
    },
    {
      "question": " statement 2",
      "answer": " ",
      "explanation": " ",
      "subtype": " ",
      "evidence_pages": [" ", " "]
    },
    {
      "question": " statement 3",
      "answer": " ",
      "explanation": " ",
      "subtype": " ",
      "evidence_pages": [" ", " "]
    },
  ]
}
"""

fact_verify_syn_prompt_text = """
# Role Play
Suppose you are a senior financial annual report analysis expert and your task is to generate high-quality fact verification question–answer pairs based on OCR-extracted annual report texts.

# Task Description
Fact verification tasks in financial report analysis involve checking whether a given natural language statement is consistent with the information contained in OCR-extracted texts from annual reports. Each statement should be specific to certain metrics, company names, fiscal years, or textual disclosures. The answer must be one of three categories:
- "Supported": the statement is fully confirmed by the evidence in the extracted texts.
- "Refuted": the statement contradicts the evidence in the extracted texts.
- "Insufficient": the extracted texts do not provide enough information to judge the statement as true or false.

The input consists of one or more OCR-extracted text blocks from annual reports. The generated statements must require retrieving relevant information across these multi-page inputs, and the correctness must be determinable only with reference to the provided text evidence.

# Generation Restrictions
To generate 5 statements based on the given task description and multi-source financial textual data, give due consideration to the following aspects:
1. Complexity: Include both simple checks (direct extraction of a metric or textual fact) and more complex reasoning (calculations, cross-year comparisons, or multi-step narrative reasoning).
2. Length: Each statement should be between 15 and 40 words.
3. Diversity: Cover two distinct types of fact verification tasks:
   - **Numerical Verification**: statements that involve verifying financial indicators present in text, either through direct extraction or multi-step quantitative reasoning.
   - **Textual Verification**: statements that involve verifying qualitative or narrative disclosures from any part of the annual report text, requiring simple checks or deeper reasoning.
4. Real-World Relevance: Statements must reflect realistic fact checks that analysts or auditors would perform.
5. Writing Style: Use professional and precise financial terminology. Explicitly mention the metric, company, fiscal year, or textual disclosure subject.
6. Balance: Generate statements across Supported, Refuted, and Insufficient. Ensure Refuted and Insufficient statements are realistic and plausible.
7. Context-Independent: Each statement must be understandable on its own without relying on external context beyond the extracted texts.

# Input Texts
Below are the OCR-extracted texts from the selected report pages. Use these as the factual basis for generating statements:
{text_block}

# Output Control
Please generate the output as a strict JSON object in the following format:
{{
  "result":[
    {{
      "question": "Your fact verification statement here",
      "answer": "Supported or Refuted or Insufficient",
      "explanation": "A brief explanation of why the answer is Supported, Refuted, or Insufficient",
      "subtype": "The subtype of the question should be one of Numerical Fact Verify or Textual Fact Verify",
      "evidence_pages": [" ", " "]
    }},
    {{
      "question": " statement 2",
      "answer": " ",
      "explanation": " ",
      "subtype": " ",
      "evidence_pages": [" ", " "]
    }},
    {{
      "question": " statement 3",
      "answer": " ",
      "explanation": " ",
      "subtype": " ",
      "evidence_pages": [" ", " "]
    }},
  ]
}}
"""


fact_verify_syn_prompt_hard = """
# Role Play
Suppose you are a senior financial annual report analysis expert and your task is to generate high-quality fact verification question–answer pairs based on annual report tables and textual excerpts.

# Task Description
Fact verification tasks in financial report analysis involve checking whether a given natural language statement is consistent with the information contained in annual reports. Each statement should be specific to certain metrics, company names, fiscal years, or textual disclosures. To simulate real-world analyst workflows, each statement must depend on information that spans across multiple report sections, fiscal years, or companies, requiring retrieval from more than one page or document. The answer must be one of three categories:
- "Supported": the statement is fully confirmed by the evidence in the annual report.
- "Refuted": the statement contradicts the evidence in the annual report.
- "Insufficient": the evidence in the annual report is not enough to judge the statement as true or false.

The input consists of one or more tables or text excerpts from annual reports. The generated statements must require retrieval of relevant information across these inputs, and the correctness must be determinable only with reference to the provided evidence.

# Generation Restrictions
To generate 5 questions based on the given task description and multi-source financial tabular data, give due consideration to the following aspects:
1. Complexity: Statements must not rely on a single data point or a single page. Each statement should require combining information across multiple companies, multiple fiscal years, or multiple report pages/documents. Include both simple factual checks (direct extraction across different years/companies) and more complex reasoning (multi-step calculations, trend verification, or multi-document textual reasoning).
2. Length: Each statement should be between 15 and 40 words.
3. Diversity: Cover two distinct types of fact verification tasks, (depending on whether the provided annual report pages are text, tables, or a mix of both):
   - **Numerical Verification**: tatements that involve verifying financial indicators, which may come from tables or from textual descriptions within the annual report, either by direct extraction or through more complex calculations.
   - **Textual Verification**: statements that involve verifying qualitative or narrative information from any part of the annual report text (including but not limited to strategic goals, risk disclosures, management discussion, or general narrative sections), requiring simple fact checks or deeper reasoning.
4. Real-World Relevance: Statements must reflect realistic checks that analysts or auditors would perform.
5. Writing Style: Use professional and precise financial terminology. Explicitly mention metric, company, fiscal year, or textual disclosure subject.
6. Balance: Generate statements that are evenly distributed across the three verification labels (Supported, Refuted, Insufficient), and also balanced between the two task types (Numerical Verification and Textual Verification). Ensure Refuted and Insufficient statements remain realistic and plausible in both types.
7. Context-Independent: Each statement must be understandable on its own without external context.

# Output Control
Please generate the output as a strict JSON object in the following format:
{
  "result":[
    {
      "question": "Your fact verification statement here",
      "answer": "Supported or Refuted or Insufficient",
      "explanation": "A brief explanation of why the answer is Supported, Refuted, or Insufficient",
      "subtype": "The subtype of the statement should be one of Numerical Fact Verify or Textual Fact Verify",
      "evidence_pages": [" ", " "]
    },
    {
      "question": " statement 2",
      "answer": " ",
      "explanation": " ",
      "subtype": " ",
      "evidence_pages": [" ", " "]
    },
    {
      "question": " statement 3",
      "answer": " ",
      "explanation": " ",
      "subtype": " ",
      "evidence_pages": [" ", " "]
    },
  ]
}
"""


# text reasoning
textual_reasoning_syn_prompt = """
# Role play
Suppose you are a senior annual report analysis expert and your task is to generate high-quality textual reasoning question-answer pairs that can only be answered through retrieval and synthesis across multiple pages or documents of company financial reports.

# Task Description
Textual reasoning tasks in financial report analysis involve interpreting and synthesizing qualitative information found in unstructured report excerpts, such as management discussion sections, strategic outlooks, ESG disclosures, and risk factor narratives across different companies and fiscal years. These tasks typically require retrieving relevant passages from multiple report sections, identifying themes, comparing strategic directions, or evaluating qualitative shifts across time or entities. Each question must require multi-source retrieval to answer, simulating a real-world retrieval-augmented analysis workflow.

The input consists of one or more images from official annual reports. These images may represent different pages from the same or different companies, and may span multiple fiscal years. Based on the type of input images, design questions that match the appropriate comparison mode — use cross-company horizontal comparisons when images are from different companies, and cross-year vertical analyses when images span multiple fiscal years of the same company.

# Generation Rules
1. Complexity: Questions should require combining and synthesizing information from multiple documents, companies, or years. Avoid questions that can be answered from a single paragraph.
2. Length: Questions should be between 20 and 50 words. Keep language professional and precise.
3. Real-World Relevance: Questions should resemble inquiries that financial analysts, investors, or regulators would realistically make — such as tracking strategic evolution, comparing disclosures, or evaluating responses to macroeconomic risks.
4. Answer Form: The answer should be a natural language explanation or summary (several sentences if needed), not limited to short numeric or entity names.
5. Context-Independent: Each question must explicitly mention the company names, years, and themes involved, so that it is fully understandable on its own.
6. Retrieval Dependency: Ensure that each QA pair requires multi-document retrieval and synthesis, not a single data point.

# Output Format
Please generate 5 QA pairs and output them as a strict JSON object in the following format:

{
    "result":[
        {
        "question": "Your question here (cross-year, cross-company, reasoning-heavy)",
        "answer": "A full natural language explanation here, possibly several sentences, clearly reasoned and complete.",
        "evidence_pages": [" ", " ", " "]
        },
        {
        "question": "question 2",
        "answer": "long-form answer 2",
        "evidence_pages": [" ", " ", " "]
        },
        {
        "question": "question 3",
        "answer": "long-form answer 3",
        "evidence_pages": [" ", " ", " "]
        }
    ]
}
"""

knowledge_reasoning_syn_prompt_text = """
# Role Play
Suppose you are a senior annual report analysis expert and your task is to generate high-quality textual reasoning question-answer pairs that can only be answered through retrieval and synthesis across multiple pages or documents of OCR-extracted annual report texts.

# Task Description
Textual reasoning tasks in financial report analysis involve interpreting and synthesizing qualitative information found in unstructured OCR-extracted excerpts, such as management discussion sections, strategic outlooks, ESG disclosures, and risk factor narratives across different companies and fiscal years. These tasks typically require retrieving relevant passages from multiple report sections, identifying themes, comparing strategic directions, or evaluating qualitative shifts across time or entities. Each question must require multi-source retrieval to answer, simulating a real-world retrieval-augmented analysis workflow.

The input consists of one or more OCR-extracted text blocks from official annual reports. These texts may represent different pages from the same or different companies, and may span multiple fiscal years. Based on the type of input texts, design questions that match the appropriate comparison mode — use cross-company horizontal comparisons when texts are from different companies, and cross-year vertical analyses when texts span multiple fiscal years of the same company.

# Generation Rules
1. Complexity: Questions should require combining and synthesizing information from multiple documents, companies, or years. Avoid questions that can be answered from a single paragraph.
2. Length: Questions should be between 20 and 50 words. Keep language professional and precise.
3. Real-World Relevance: Questions should resemble inquiries that financial analysts, investors, or regulators would realistically make — such as tracking strategic evolution, comparing disclosures, or evaluating responses to macroeconomic risks.
4. Answer Form: The answer should be a natural language explanation or summary (several sentences if needed), not restricted to numbers or short entity labels.
5. Context-Independent: Each question must explicitly mention the company names, years, and themes involved, ensuring it is fully understandable on its own.
6. Retrieval Dependency: Ensure that each QA pair requires multi-document retrieval and synthesis, not a single data point.

# Input Texts
Below are the OCR-extracted texts from the selected annual report pages. Use these as the factual basis for generating questions and answers:
{text_block}

# Output Format
Please generate 5 QA pairs and output them as a strict JSON object in the following format:

{{
    "result":[
        {{
            "question": "Your question here (cross-year, cross-company, reasoning-heavy)",
            "answer": "A full natural language explanation here, possibly several sentences, clearly reasoned and complete.",
            "evidence_pages": [" ", " ", " "]
        }},
        {{
            "question": "question 2",
            "answer": "long-form answer 2",
            "evidence_pages": [" ", " ", " "]
        }},
        {{
            "question": "question 3",
            "answer": "long-form answer 3",
            "evidence_pages": [" ", " ", " "]
        }}
    ]
}}
"""
