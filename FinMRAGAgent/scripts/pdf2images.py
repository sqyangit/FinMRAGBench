import os
from pdf2image import pdfinfo_from_path
from pdf2image import convert_from_path
from tqdm import tqdm
# yum install poppler
# yum install poppler-utils

# datasets = ["ViDoSeek", "SlideVQA"]
datasets = ["FinanceReport"]
for dataset in datasets:
    root_path = f"./data/{dataset}"
    pdf_path = os.path.join(root_path, "pdf")
    pdf_files = [file for file in os.listdir(pdf_path) if file.endswith("pdf")]

    # 按字母分组
    # szm = "Y"
    # print(szm)
    # pdf_files = [file for file in os.listdir(pdf_path) if file.endswith(".pdf") and file.startswith(szm)]

    # 按公司分组
    # pdf_files = (
    #     [f"NKE_10K_{year}.pdf" for year in range(2020, 2026)]
    #     + [f"LULU_10K_{year}.pdf" for year in range(2020, 2026)]
    #     + [f"RL_10K_{year}.pdf" for year in range(2020, 2026)]
    # )
    # print(pdf_files)
    # import pdb;pdb.set_trace()
    
    # pdf转换为图片
    # img_path = os.path.join(root_path, "img")
    # os.makedirs(img_path, exist_ok=True)
    # for filename in tqdm(pdf_files):
    #     filepath = os.path.join(pdf_path, filename)
    #     imgname = filename.split(".pdf")[0]
    #     images = convert_from_path(filepath, thread_count=8)
    #     for i, image in enumerate(images):
    #         idx = i + 1
    #         image.save(
    #             os.path.join(root_path, "img", f"{imgname}_{idx}.jpg"), "JPEG"
    #         )

    # 计数总页数
    total_pages = 0
    for filename in tqdm(pdf_files, desc="Counting pages"):
        filepath = os.path.join(pdf_path, filename)
        try:
            info = pdfinfo_from_path(filepath)
            total_pages += info["Pages"]
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    print(f"Dataset {dataset} → Total PDFs: {len(pdf_files)}, Total Pages: {total_pages}")