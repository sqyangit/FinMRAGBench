import numpy as np
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

def compare(list1, list2):
    # sort the list
    list1.sort()
    list2.sort()
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if np.isnan(list1[i]):
            if not np.isnan(list2[i]):
                return False
        elif list1[i] != list2[i]:
            return False
    return True

def std_digit(list_nums):
    new_list = []
    for i in range(len(list_nums)):
        new_list.append(round(list_nums[i], 2))
    return new_list

def is_close(a, b, tol=0.01):
    """比较两个数是否在允许的误差范围内 (默认 1%)"""
    if a == 0:  # 避免除以 0 的情况
        return abs(b) <= tol
    return abs(a - b) <= abs(a) * tol

def compare_with_tolerance(refs, preds, tol=0.01):
    if len(refs) != len(preds):
        return False
    for r, p in zip(refs, preds):
        if not is_close(r, p, tol):
            return False
    return True

def compare_with_tolerance_sorted(refs, preds, tol=0.01):
    
    # 排序降低要求
    refs = sorted(refs)
    preds = sorted(preds)

    if len(refs) != len(preds):
        return False
    for r, p in zip(refs, preds):
        if not is_close(r, p, tol):
            return False
    return True

def compare_with_tolerance_sorted_score(list1, list2, tol=0.01):
    """先排序，再逐一比较，返回匹配比例 (0~1)"""
    list1 = sorted(list1)
    list2 = sorted(list2)

    if len(list1) != len(list2):
        return 0.0  # 长度不同直接判 0 分

    correct = 0
    for a, b in zip(list1, list2):
        if is_close(a, b, tol):
            correct += 1

    return correct / len(list1) if list1 else 1.0

def compute_general_chart_metric(references, predictions):
    processed_references = []
    processed_predictions = []
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    return compare(processed_references, processed_predictions)

def compute_general_chart_metric_loose(references, predictions):
    processed_references = []
    processed_predictions = []

    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)

    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)

    # 允许 1% 误差
    return compare_with_tolerance_sorted(processed_references, processed_predictions, tol=0.01)
    return compare_with_tolerance_sorted_score(processed_references, processed_predictions, tol=0.01)

def compute_pie_chart_metric(references, predictions):
    processed_references = []
    processed_predictions = []
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)
    references = processed_references
    processed_references = []
    total = 0
    for reference in references:
        total += reference
    for reference in references:
        processed_references.append(round(reference / total, 2))

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    return compare(processed_references, processed_predictions)

def compute_pie_chart_metric_loose(references, predictions, tol=0.01):
    processed_references = []
    processed_predictions = []

    # 展平 references
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)

    # 归一化成比例
    total = sum(processed_references)
    normalized_references = [round(ref / total, 2) for ref in processed_references]

    # 展平 predictions
    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)

    # 保留两位小数
    normalized_references = std_digit(normalized_references)
    processed_predictions = std_digit(processed_predictions)

    # 比较，允许 tol（默认 1%）误差
    return compare_with_tolerance_sorted(normalized_references, processed_predictions, tol=tol)
    return compare_with_tolerance_sorted_score(normalized_references, processed_predictions, tol=tol)


def get_line_y_predictions(plt):
    line_y_predctions = []
    lines = plt.gca().get_lines()
    line_y_predctions = [list(line.get_ydata()) for line in lines]
    return line_y_predctions


def get_bar_y_predictions(plt):
    bar_y_predctions = []
    patches = plt.gca().patches
    bar_y_predctions = [patch.get_height() for patch in patches]
    return bar_y_predctions


def get_hbar_y_predictions(plt):
    hbar_y_predctions = []
    patches = plt.gca().patches
    hbar_y_predctions = [patch.get_width() for patch in patches]
    return hbar_y_predctions


def get_pie_y_predictions(plt):
    pie_y_predctions = []
    patches = plt.gca().patches
    for patch in patches:
        theta1, theta2 = patch.theta1, patch.theta2
        value = round((theta2 - theta1) / 360.0, 2)
        pie_y_predctions.append(value)
    return pie_y_predctions


def get_area_y_predictions(plt):
    area_y_predctions = []
    area_collections = plt.gca().collections
    for area_collection in area_collections:
        area_items = []
        for item in area_collection.get_paths()[0].vertices[:, 1]:
            if item != 0:
                area_items.append(item)
        area_y_predctions.append(area_items)
    return list(area_y_predctions)


def get_radar_y_predictions(plt):
    radar_y_predctions = []
    radar_lines = plt.gca().get_lines()
    radar_y_predctions = [list(line.get_ydata()) for line in radar_lines]
    for i in range(len(radar_y_predctions)):
        radar_y_predctions[i] = radar_y_predctions[i][:-1]
    return radar_y_predctions


def get_scatter_y_predictions(plt):
    scatter_y_predctions = []
    scatter_collections = plt.gca().collections
    for scatter_collection in scatter_collections:
        scatter_items = []
        for item in scatter_collection.get_offsets():
            scatter_items.append(item[1])
        scatter_y_predctions.append(scatter_items)
    return scatter_y_predctions


def get_waterfall_y_predictions(plt):
    waterfall_y_predctions = []
    patches = plt.gca().patches
    waterfall_y_predctions = [patch.get_height() for patch in patches]
    return waterfall_y_predctions

def get_bubble_y_predictions(plt):
    scatter_collection = plt.gca().collections[0]  # 只取第一个
    offsets = scatter_collection.get_offsets()
    sizes = scatter_collection.get_sizes()
    xs = offsets[:, 0].tolist()
    ys = offsets[:, 1].tolist()
    ss = sizes.tolist()
    return [xs, ys, ss]

def get_stackedbar_y_predictions(plt):
    stackedbar_y_predictions = []
    patches = plt.gca().patches
    for patch in patches:
        stackedbar_y_predictions.append(patch.get_height())
    return stackedbar_y_predictions