import sys
import os
import json
import logging
from pathlib import Path
import re
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import traceback
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deplot import metrics


# def extract_markdown_from_response(response):
#     """Extract markdown data from API response"""

#     markdown_pattern = r"```(?:markdown)?\s*([\s\S]*?)```"
#     match = re.search(markdown_pattern, response, re.IGNORECASE)
    
#     if match:
#         content = match.group(1).strip()
#         split_content = content.split('\n\n', 1)
#         return split_content[0].strip() if len(split_content) > 1 else content
    
#     return response.strip()



def extract_markdown_from_response(response):
    """Extract markdown data from API response and normalize table separators"""

    markdown_pattern = r"```(?:markdown)?\s*([\s\S]*?)```"
    match = re.search(markdown_pattern, response, re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        split_content = content.split('\n\n', 1)
        extracted = split_content[0].strip() if len(split_content) > 1 else content
    else:
        extracted = response.strip()

    # 将 Markdown 表格的对齐分隔符 ":-" 统一为 "-"
    extracted = extracted.replace(":-", "-")
    
    return extracted


def normalize_markdown(markdown_text):
    """Convert Markdown text to DePlot evaluation supported format
    
    Args:
        markdown_text (str): Markdown text output by model
        
    Returns:
        str: Normalized markdown text
    """
    if not markdown_text:
        return ""
        
    # Remove extra empty lines
    lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
    
    # Process header separators
    normalized_lines = []
    for i, line in enumerate(lines):
        # Skip header separator lines
        if i >  0 and all(c in ['-', '|'] for c in line.replace(' ', '')):
            continue
        # Process regular lines
        if '|' in line:
            # Clean extra vertical bars
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if parts:
                # Process numbers in each cell (remove commas)
                processed_parts = []
                for j, part in enumerate(parts):
                    # If it's a header row (first row) or header column (first column), keep as is
                    if i == 0 or j == 0:
                        processed_parts.append(part)
                    else:
                        # Try to convert cell content to number (remove commas)
                        try:
                            # If it's a number (may contain commas), remove commas
                            if any(c.isdigit() for c in part):
                                # Remove all commas
                                processed_part = part.replace(',', '')
                                # If it's in scientific notation, keep the format
                                if 'e' in processed_part.lower():
                                    processed_parts.append(processed_part)
                                else:
                                    # Try to convert to number and back to string to ensure correct format
                                    try:
                                        num = float(processed_part)
                                        processed_parts.append(str(num))
                                    except ValueError:
                                        processed_parts.append(processed_part)
                            else:
                                processed_parts.append(part)
                        except:
                            processed_parts.append(part)
                # Remove vertical bars at beginning and end
                normalized_lines.append(' | '.join(processed_parts))
        else:
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)


def evaluate_prediction(prediction, ground_truth):
    """Evaluate differences between prediction and ground truth"""
    try:
        # Clean and normalize prediction and ground truth
        gt_markdown = normalize_markdown(ground_truth)
        pred_markdown = normalize_markdown(prediction)

        # Print normalized data
        logging.info(f"Normalized ground truth: {gt_markdown}")
        logging.info(f"Normalized prediction: {pred_markdown}")

        # Use normalized data to construct DePlot evaluation format
        targets = [[gt_markdown]]       # list[list[str]] each outer layer represents a sample, inner layer is multiple candidate answers (usually 1)
        predictions = [pred_markdown]   # list[str] corresponding one-to-one with targets, each is model output

        # Calculate various evaluation metrics
        result = {}
        result = metrics.table_datapoints_precision_recall(targets, predictions, number_theta=0.01)
        result.update(metrics.table_number_accuracy_extext(targets, predictions, number_theta=0.01))
        
        # Log evaluation results
        logging.info(f"Evaluation results: "
                    f"Precision={result['table_datapoints_precision_onlyvalue']:.2f}, "
                    f"Recall={result['table_datapoints_recall_onlyvalue']:.2f}, "
                    f"F1_score={result['table_datapoints_f1_onlyvalue']:.2f}, "
                    f"Numbers accuracy excluding headers={result['numbers_match_extext']:.2f}")
        
        return result
    
    except Exception as e:
        logging.error(f"Error evaluating prediction result: {str(e)}")
        logging.error(f"Prediction: {prediction}")
        logging.error(f"Ground truth: {ground_truth}")
        logging.error(traceback.format_exc())
        return {
            "table_datapoints_precision": 0,
            "table_datapoints_recall": 0,
            "table_datapoints_f1": 0,
            "numbers_match_extext": 0
        }