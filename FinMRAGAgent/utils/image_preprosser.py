
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
from PIL import Image
from collections import Counter



def concat_images_with_bbox(images, bboxs=None, arrangement=(3,3), scale=1.0, line_width=40,max_pixel=None):
    """
    将一组图片按照指定排布拼接，并对最终图片进行缩放，同时在每张图片之间添加分隔线。
    现在我有bbox列表，List[List[[y_min,x_min,y_max,x_max]]]，0<=y_min<y_max<=1,0<=x_min<x_max<=1，表示在图片的相对位置，请在拼接后的图片上画出bbox，并且同时return bbox在拼接后的图片的相对位置，List[List[y_min,x_min,y_max,x_max]]，0<=y_min<y_max<=1,0<=x_min<x_max<=1

    参数:
    - images: list of PIL.Image，待拼接的图片列表。
    - bboxs: list of list of list，每个元素是一个包含四个元素的列表，表示一个 bbox 的坐标。
    - arrangement: tuple 或 str，图片的排布方式。可以是（行，列）的元组，例如 (3, 3)，
                   或字符串 'horizontal'（水平拼接）或 'vertical'（垂直拼接）。
    - scale: float，最终图片的缩放系数。
    - line_width: int，分隔线的宽度。

    返回:
    - PIL.Image，拼接并缩放后的图片。
    - List[List[float]]，bbox 在拼接后的图片的相对位置。
    """
    # 获取所有图片的尺寸
    images = [Image.open(img).convert('RGB') for img in images]
    sizes = [image.size for image in images]

    # 找到最常见的尺寸
    size_counts = Counter(sizes)
    most_common_size = size_counts.most_common(1)[0][0]
    width, height = most_common_size

    # 将所有图片调整为最常见的尺寸
    images_resized = [img.resize(most_common_size) for img in images]

    # 确定排布方式
    if isinstance(arrangement, tuple) and len(arrangement) == 2:
        rows, columns = arrangement
    elif arrangement == 'horizontal':
        rows = 1
        columns = len(images_resized)
    elif arrangement == 'vertical':
        rows = len(images_resized)
        columns = 1
    else:
        # 默认水平拼接
        rows = 1
        columns = len(images_resized)

    # 计算需要的总单元数
    total_cells = rows * columns

    # 调整图片列表，填充空白或截断多余的图片
    if len(images_resized) < total_cells:
        # 用空白图片填充
        num_padding = total_cells - len(images_resized)
        blank_img = Image.new('RGB', most_common_size, color=(255,255,255))  # 白色背景
        images_resized.extend([blank_img]*num_padding)
    elif len(images_resized) > total_cells:
        # 只使用前 total_cells 张图片
        images_resized = images_resized[:total_cells]

    # 创建新画布
    total_width = columns * width + (columns - 1) * line_width  # 加上间隔线的宽度
    total_height = rows * height + (rows - 1) * line_width  # 加上间隔线的高度
    new_img = Image.new('RGB', (total_width, total_height), color=(0,0,0))  # 白色背景

    # 将图片逐一粘贴到新画布
    for idx, img in enumerate(images_resized):
        row = idx // columns
        col = idx % columns
        x = col * (width + line_width)  # 加上分隔线的宽度
        y = row * (height + line_width)  # 加上分隔线的高度
        new_img.paste(img, (x, y))

    # 对最终图片进行缩放
    if max_pixel is not None:
        total_pixels = total_width * total_height
        scale = (max_pixel / total_pixels) ** 0.5
        scaled_width = int(total_width * scale)
        scaled_height = int(total_height * scale)
        final_img = new_img.resize((scaled_width, scaled_height))
    else:
        if scale > 1:
            max_side = max(total_width, total_height)
            scale = scale / max_side
        scaled_width = int(total_width * scale)
        scaled_height = int(total_height * scale)
        final_img = new_img.resize((scaled_width, scaled_height))

    return final_img

if __name__ == "__main__":
    # 假设当前目录下有三张图片 test1.jpg, test2.jpg, test3.jpg
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
    evidence_pages = [
                "AMZN_10K_2023_38.jpg",
                "AMZN_10K_2023_40.jpg",
                "AMZN_10K_2023_41.jpg",
                "AMZN_10K_2024_37.jpg",
                "AMZN_10K_2024_39.jpg",
                "AMZN_10K_2024_40.jpg"
    ]
    image_files = [base_path + filename for filename in evidence_pages]

    # 调用拼接函数
    result_img = concat_images_with_bbox(
        images=image_files,
        arrangement=(2, 3),   # 一行三列
        scale=1.0,
        line_width=40
    )

    # 保存到当前目录
    result_img.save("concat_result.jpg")
    print("拼接结果已保存为 concat_result.jpg")