# 金融指标、任务提示词及行业与公司映射


# 金融指标
finance_metrics = [
    "total assets",
    "revenue",
    "revenue growth rate",
    "operating income",
    "operating income ratio",
    "gross profit/income",
    "net income/profit",
    "Gross Profit Margin",
    "Net Profit Margin",
    "ROE (return on equity)",
    "ROA (return on assets)",
    "Inventory turnover rate",
    "receivables turnover",
    "Equity Multiplier",
    "common dividends",
    "R&D expenses",
    "research and development expenses",
    "SG&A expense",
    "basic earnings per share (EPS)",
    "Current Ratio",
    "Quick Ratio",
    "debt to assets ratio",
    "debt to equity ratio",
    "equity to assets ratio",
    "interest coverage ratio",
    # "ROIC",
]

# 问题改写prompt
rewrite_prompt = """
You are a financial AI Assistant. 
Given a financial question, company ticker, and document period, determine whether the question already includes the company name and report year. 
If the question does not include both, rewrite it to include the missing information naturally.

Rules:
- If the question already contains both the company name and report year, do not modify it.
- If either the company name or the report year is missing, rewrite the question to include both.
- Maintain the original intent and factual accuracy of the question.
- Integrate the company name and report year smoothly into the sentence (avoid direct connectors like 'in', 'for', 'according to').
- The ticker is only informational and should not appear in the question.

Input:
- company ticker: {ticker}
- document_period: {period}
- question: {question}

Output format (in JSON):
```json 
{{
    "rewritten_flag": "0 or 1",
    "new_question": "None or your rewritten question"
}}
```

Where:
- "rewritten_flag": "0" → The question already includes company name and report year, no rewrite needed.
- "rewritten_flag": "1" → The question was missing company name and/or report year, rewritten accordingly.
- "new_question": "None" if no rewrite needed, otherwise contains the rewritten question.
"""

# 问题分解prompt
decomposition_prompt = """
You are a query decomposition assistant.  
Decompose a natural-language query into a list of **retrieval subqueries** that are as atomic and explicit as possible.  
Each subquery should correspond to one concrete fact to be retrieved (e.g., one company × one year × one metric).  
If decomposition is not possible, just return the original query.

Each subquery must represent the most atomic unit of retrieval:
- Exactly one company
- Exactly one year (or fiscal year)
- Exactly one simple metric  
If the query involves multiple companies or years, expand them into multiple subqueries.  
If the query contains a complex metric, decompose it into its basic components.
Common examples of complex metric decomposition:
- ROA (Return on Assets) = Net Income / Total Assets  
- ROE (Return on Equity) = Net Income / Shareholders' Equity  
- Operating Margin = Operating Income / Revenue  
- Debt-to-Equity = Total Debt / Shareholders' Equity 

Output strictly in JSON format:
```json 
{{
  "subqueries": [ ... ]
}}
```

Here are some examples:

{examples_block}

Now decompose the following query:

Original query: {query}
"""


# 页面检查prompt
recheck_prompt = """
You are reviewing one page from a company's financial report.

Your task:
Evaluate how informative this page is for answering the following question.

Question: {subquery}

Output strictly in JSON format (no markdown, no explanation outside JSON):
```json 
{{
  "score": int,                 // 0–10 integer; higher = more useful
  "rationale": "Brief reason (≤20 words)"
}}
```

Scoring guidelines:
- 0–2: Irrelevant (e.g., title page, table of contents, disclaimers)
- 3–4: Marginal (mentions section names or headers only, no real data)
- 5–6: Somewhat relevant (mentions related concepts but lacks specific values)
- 7–8: Useful (contains partial data, figures, or textual discussion relevant to the question)
- 9–10: Highly useful (contains detailed tables, numerical values, or clear evidence directly answering the question)

Rules:
- Focus on whether the page provides *substantive financial information* (numbers, ratios, results, tables).
- Do NOT give a high score if it only lists report sections or references other pages.
- Output only valid JSON, nothing else.
"""

extra_metadata_prompt = """
You are given a subquery related to companies in the "{industry}" industry. 
Your task is to extract two key pieces of metadata from the text:

1. "ticker" — the stock ticker symbol of the mentioned company, based on the mapping below.
2. "year" — the specific year mentioned in the subquery, if any.

Here is the mapping between company names and their tickers (in JSON format):
{ticker_map}

Rules:
- Match the company name (or abbreviation) in the subquery to the correct ticker from the mapping.
- If no company name is found, set "ticker" to an empty string.
- If no year is found, set "year" to an empty string.
- Always return the result as a valid JSON object with exactly two keys: "ticker" and "year".
- Do not include any explanation, reasoning, or extra text outside of the JSON.

Example:
Input: "Texas Instruments 2022 R&D expenses"
Output:
```json 
{{
  "ticker": "TXN",
  "year": "2022"
}}
```

Now extract the metadata for the following subquery:
{subquery}
"""

extra_metadata_prompt_all = """
You are given a subquery related to public companies.
Your task is to extract two key pieces of metadata from the text:

1. "ticker" — the stock ticker symbol of the mentioned company, based on the mapping below.
2. "year" — the specific year mentioned in the subquery, if any.

Here is the mapping between company names and their tickers (in JSON format):
{ticker_map}

Rules:
- Match the company name (or abbreviation) in the subquery to the correct ticker from the mapping.
- If no company name is found, set "ticker" to an empty string.
- If no year is found, set "year" to an empty string.
- Always return the result as a valid JSON object with exactly two keys: "ticker" and "year".
- Do not include any explanation, reasoning, or extra text outside of the JSON.

Example:
Input: "Texas Instruments 2022 R&D expenses"
Output:
```json 
{{
  "ticker": "TXN",
  "year": "2022"
}}
```

Now extract the metadata for the following subquery:
{subquery}
"""


subquery_judge_prompt = """
You are given a single financial sub-question (subquery) and several OCR-extracted text excerpts from report pages that were retrieved specifically for this subquery.
Your task is to judge whether these texts are SUFFICIENT to answer the subquery, including cases where the subquery is composite and requires combining multiple metrics.

Important judging criteria:
1) Target coverage: Check whether the necessary information to answer THIS subquery appears in the texts. If the subquery requires a numeric value, confirm the presence of that value (and any operands needed).
2) Composite needs: If the subquery involves a sum/ratio/difference or multi-hop lookup, the texts are sufficient if ALL operands/inputs can be obtained from the provided texts (they may be on different pages).
3) Consistency: Time period (e.g., fiscal year), units, and entity scope must be present or unambiguously inferable. Conflicts/ambiguities mean NOT sufficient.
4) Semantic equivalence: Treat common financial terminology variations and synonyms as equivalent (e.g., "Shareholders' equity" ≈ "Stockholders' equity"; "Operating income" ≈ "Operating profit"; "Revenue" ≈ "Sales"; "Net income" ≈ "Net earnings"/"Profit after tax"). Ignore plural/singular and minor wording differences.
5) Do NOT compute the answer; only judge sufficiency.
6) If ANY required operand/context is missing or inconsistent, judge "NO".

Return a STRICT JSON object with exactly TWO fields:
- "judge": "YES" if the texts are sufficient to answer this subquery, otherwise "NO".
- "rationale": ONE sentence explaining the key reason for your decision (e.g., which operand/year/entity is missing).

Subquery (evaluate sufficiency for this only):
{subquery}

Retrieved texts for this subquery (paired with file indices and filenames):
{text_block}

Output ONLY the JSON object, with no extra commentary.
The JSON must strictly follow this schema:
```json
{{
  "judge": "YES" or "NO",
  "rationale": "Brief explanation in one sentence"
}}
```

Now make your decision based on the given inputs.
"""

subquery_judge_prompt_easy = """
You are given a financial question and several text excerpts (from OCR of report pages). 
Your task is to determine whether the provided texts are sufficient to answer the question.

Question:
{query}

Extracted texts from the selected pages:
{text_contents}

Please analyze the relevance and completeness of the provided texts relative to the question. 
Then decide whether these pages are sufficient to answer the question.

Output a valid JSON object with exactly the following two fields:
- "judge": "yes" if the texts are sufficient to answer the question, otherwise "no".
- "rationale": a one-sentence explanation of your reasoning.

Example:
Input question: "What was NVIDIA's total R&D expense in 2022?"
If the texts contain a clear statement of NVIDIA's 2022 R&D expense, output:
{{
  "judge": "yes",
  "rationale": "The provided text explicitly lists NVIDIA's R&D expense for 2022."
}}
Otherwise, if the information is missing or incomplete, output:
{{
  "judge": "no",
  "rationale": "The provided text does not include the requested year's R&D expense."
}}

Now make your decision based on the given inputs.
"""

final_judge_prompt = """
You are given a financial question, a list of sub-questions (subqueries), and several text excerpts (OCR from report pages).
Your task is to judge whether the provided texts are SUFFICIENT to answer the overall question, including composite metrics that require combining multiple subqueries.

Important judging criteria:
1) Coverage of subqueries: For EACH subquery, check whether the necessary information to answer it appears in the texts.
2) Composite metrics: If the main question requires combining multiple subqueries (e.g., sum/ratio/difference), the texts are sufficient if ALL operands for those subqueries can be obtained from the provided texts, even if not on the same page.
3) Consistency: Required context such as time period, units, and company scope must be present or unambiguously inferable from the provided texts. Conflicts or ambiguities mean NOT sufficient.
4) Semantic equivalence: Treat common financial terminology variations and synonyms as equivalent. 
   For example:
   - "Shareholders' equity" ≈ "Stockholders' equity"
   - "Operating income" ≈ "Operating profit"
   - "Revenue" ≈ "Sales"
   - "Net income" ≈ "Net earnings" or "Profit after tax"
   - Ignore plural/singular differences, capitalization, and minor wording variations.
5) No need to actually compute the final answer—only judge sufficiency.
6) If ANY required subquery is missing, or the units/periods are inconsistent or ambiguous, judge "no".

Return a STRICT JSON object with exactly TWO fields:
- "judge": "yes" if the texts are sufficient to answer the overall question (including all subqueries), otherwise "no".
- "rationale": ONE sentence explaining the key reason for your decision.

Main question:
{query}

Subqueries (check coverage for each):
{subqueries_block}

Extracted texts from selected pages (may be long):
{text_block}

Output ONLY the JSON object, with no extra commentary:
```json 
{{
  "judge": "YES" or "NO",
  "rationale": "Brief explanation in one sentence"
}}
```

Now make your decision based on the given inputs.
"""

# 10大行业，公司名与股票代码的对应关系
ticker_to_name_semiconductors = {
    "NVDA": "NVIDIA Corporation",
    "AVGO": "Broadcom Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "TXN": "Texas Instruments Incorporated",
    "QCOM": "QUALCOMM Incorporated",
    "MU": "Micron Technology, Inc.",
    "KLAC": "KLA Corporation",
    "ADI": "Analog Devices, Inc.",
    "INTC": "Intel Corporation",
    "MRVL": "Marvell Technology, Inc.",
    "NXPI": "NXP Semiconductors N.V.",
    "MPWR": "Monolithic Power Systems, Inc.",
    "ALAB": "Astera Labs, Inc.",
    "MCHP": "Microchip Technology Inc.",
    "FSLR": "First Solar, Inc.",
}

ticker_to_name_restaurants = {
    "MCD": "McDonald's Corporation",
    "SBUX": "Starbucks Corporation",
    "CMG": "Chipotle Mexican Grill, Inc.",
    "YUM": "Yum! Brands, Inc.",
    "QSR": "Restaurant Brands International Inc.",
    "DRI": "Darden Restaurants, Inc.",
    "YUMC": "Yum China Holdings, Inc.",
    "DPZ": "Domino's Pizza Inc",
    "TXRH": "Texas Roadhouse, Inc.",
    "ARMK": "Aramark",
    "WING": "Wingstop Inc.",
    "CAVA": "CAVA Group, Inc.",
    "EAT": "Brinker International, Inc.",
    "SHAK": "Shake Shack, Inc.",
    "CAKE": "The Cheesecake Factory Inc.",
}

ticker_to_name_pharmaceuticals = {
    "LLY": "Eli Lilly and Company",
    "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie Inc.",
    # "NVO": "Novo Nordisk A/S",
    # "NVS": "Novartis AG",
    # "AZN": "AstraZeneca PLC",
    "MRK": "Merck & Company, Inc.",
    "PFE": "Pfizer, Inc.",
    "VRTX": "Vertex Pharmaceuticals Inc.",
    # "SNY": "Sanofi",
    "BMY": "Bristol-Myers Squibb Company",
    "ZTS": "Zoetis Inc.",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "ALNY": "Alnylam Pharmaceuticals, Inc.",
}

ticker_to_name_aerospace_defense = {
    "GE": "GE Aerospace",
    "RTX": "RTX Corporation",
    "BA": "Boeing Company (The)",
    "HON": "Honeywell International Inc.",
    "LMT": "Lockheed Martin Corporation",
    "TDG": "Transdigm Group Inc.",
    "GD": "General Dynamics Corporation",
    "NOC": "Northrop Grumman Corporation",
    "HWM": "Howmet Aerospace Inc.",
    "AXON": "Axon Enterprise, Inc.",
    "LHX": "L3Harris Technologies, Inc.",
    "HEI": "Heico Corporation",
    "TDY": "Teledyne Technologies Inc.",
    "RKLB": "Rocket Lab Corporation",
    "ESLT": "Elbit Systems Ltd.",
}

ticker_to_name_automobile = {
    "TSLA": "Tesla, Inc.",
    "GM": "General Motors Company",
    "F": "Ford Motor Company",
    "RIVN": "Rivian Automotive, Inc.",
    "OSK": "Oshkosh Corporation",
    "FSS": "Federal Signal Corporation",
    "LCID": "Lucid Group, Inc.",
    "HOG": "Harley-Davidson, Inc.",
    "REVG": "REV Group, Inc.",
    "BLBD": "Blue Bird Corporation",
}

ticker_to_name_apparel = {
    "NKE": "Nike, Inc.",
    "LULU": "lululemon athletica inc.",
    "RL": "Ralph Lauren Corporation",
    "DECK": "Deckers Outdoor Corporation",
    "SKX": "Skechers U.S.A., Inc.",
    "LEVI": "Levi Strauss & Co",
    "VFC": "V.F. Corporation",
    "CROX": "Crocs, Inc.",
    "KTB": "Kontoor Brands, Inc.",
    "PVH": "PVH Corp.",
    "COLM": "Columbia Sportswear Company",
}

ticker_to_name_electric_utilities = {
    "NEE": "NextEra Energy, Inc.",
    "SO": "Southern Company (The)",
    "CEG": "Constellation Energy Corporation",
    "DUK": "Duke Energy Corporation",
    "VST": "Vistra Corp.",
    "AEP": "American Electric Power Company",
    "D": "Dominion Energy, Inc.",
    "EXC": "Exelon Corporation",
    "XEL": "Xcel Energy Inc.",
    "PEG": "Public Service Enterprise Group",
    "ETR": "Entergy Corporation",
    "ED": "Consolidated Edison, Inc.",
    "WEC": "WEC Energy Group, Inc.",
    "PCG": "Pacific Gas & Electric Co.",
    "DTE": "DTE Energy Company",
}

ticker_to_name_retail = {
    "AMZN": "Amazon.com, Inc.",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corporation",
    "HD": "Home Depot, Inc. (The)",
    "TJX": "TJX Companies, Inc. (The)",
    "LOW": "Lowe's Companies, Inc.",
    "MELI": "MercadoLibre, Inc.",
    "CVS": "CVS Health Corporation",
    "ORLY": "O'Reilly Automotive, Inc.",
    "CVNA": "Carvana Co.",
    "AZO": "AutoZone, Inc.",
    "CPNG": "Coupang, Inc.",
    "ROST": "Ross Stores, Inc.",
    "CPRT": "Copart, Inc.",
    "KR": "Kroger Company (The)",
    "TGT": "Target Corporation",
    "EBAY": "eBay Inc.",
    "TSCO": "Tractor Supply Company",
    "WSM": "Williams-Sonoma, Inc.",
    "DG": "Dollar General Corporation",
}

ticker_to_name_energy = {
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources, Inc.",
    "MPC": "Marathon Petroleum Corporation",
    "PSX": "Phillips 66",
    "VLO": "Valero Energy Corporation",
    "OXY": "Occidental Petroleum Corporation",
    "FANG": "Diamondback Energy, Inc.",
    "EQT": "EQT Corporation",
}

ticker_to_name_bank = {
    "JPM": "JP Morgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "WFC": "Wells Fargo & Company",
    "C": "Citigroup, Inc.",
    "COF": "Capital One Financial Corporation",
    "PNC": "PNC Financial Services Group, Inc. (The)",
    "USB": "U.S. Bancorp",
    "TFC": "Truist Financial Corporation",
    "MTB": "M&T Bank",
    "FITB": "Fifth Third Bancorp",
}

industry_to_dict = {
    "Semiconductors": ticker_to_name_semiconductors,
    "Restaurants": ticker_to_name_restaurants,
    "Pharmaceuticals": ticker_to_name_pharmaceuticals,
    "AerospaceDefense": ticker_to_name_aerospace_defense,
    "Automobile": ticker_to_name_automobile,
    "Apparel": ticker_to_name_apparel,
    "ElectricUtilities": ticker_to_name_electric_utilities,
    "Retail": ticker_to_name_retail,
    "Energy": ticker_to_name_energy,
    "Bank": ticker_to_name_bank,
}

if __name__ == "__main__":
    import random
    
    # 随机选择一个行业中的三家公司
    industry = random.choice(list(industry_to_dict.keys()))
    company_dict = industry_to_dict[industry]
    sample_companies = random.sample(list(company_dict.items()), 3)

    print(f"随机选中的行业: {industry}")
    print("随机选中的公司:")
    for ticker, name in sample_companies:
        print(f"{ticker}: {name}")