
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
import torch
from PIL import Image
import sys
import base64
from io import BytesIO
from openai import OpenAI
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

def _encode_image(image_path):
    if isinstance(image_path, Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        base64_encoded = base64.b64encode(img_data).decode("utf-8")
        return base64_encoded
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


# 远程模型客户端
class Qwen_VL_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-32B-Instruct", base_url=None, api_key="dummy"):
        # vLLM 提供 OpenAI API 兼容接口
        if base_url is not None:
            pass
        # 未指定服务地址时按模型规模选择默认端口
        elif "7B" in model_name or "8B" in model_name:
            base_url="http://127.0.0.1:8001/v1"
        elif "32B" in model_name:
            base_url="http://127.0.0.1:8011/v1"
        else: 
            pass
        print("base_url: ", base_url)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_pixels = 512 * 28 * 28 * 2
        self.min_pixels = 256 * 28 * 28

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image/jpeg;base64,{base64_string}"

        return image, base64_qwen
    def generate(self, query, image):
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # 组装 messages
        content = []
        content.append({"type": "text", "text": query})

        if image:
            # 如果是单张图片，包成 list
            # import pdb;pdb.set_trace()
            if not isinstance(image, (list, tuple)):
                image = [image]

            for img in image:
                if isinstance(img, (str, Path)):  # 路径
                    # base64_image = _encode_image(img)
                    image_raw = Image.open(img)
                    image_input, base64_image = self.process_image(image_raw)
                elif isinstance(img, Image.Image):  # PIL.Image
                    # base64_image = _encode_image(img)
                    image_input, base64_image = self.process_image(img)
                else:
                    raise ValueError(f"不支持的图片类型: {type(img)}")


                # 预处理过
                content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image}
                })

        messages = [{"role": "user", "content": content}]

        # 调用 vLLM OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,   # 可调
            temperature=0,

            # ViDoRAG 实验的可选采样参数
            # temperature=0.8,
            # top_p=0.9,
            # frequency_penalty=0.6,
            # presence_penalty=0.4,
        )
        # token 统计
        print("Prompt tokens:", response.usage.prompt_tokens)
        print("Completion tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)

        return response.choices[0].message.content


class Qwen_LM_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

    def generate(self, query, max_new_tokens=512):
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # 构造 prompt
        messages = [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 模型生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 也可以设置为 True 以获得多样性更强的回答
        )

        # 去掉 prompt 部分，只返回新生成内容
        generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response


class GPT_LOCAL:
    def __init__(self, model_name="gpt-4o"):
        from openai import OpenAI
        self.model_name = model_name
        if self.model_name.startswith("claude") or self.model_name.startswith("gemini"):
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""), # claude、gemini
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
        else:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""), # gpt
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
        self.max_pixels = 512 * 28 * 28 * 2
        self.min_pixels = 256 * 28 * 28

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        # image.save("<PROJECT_PATH>")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        # base64_qwen = f"data:image/jpeg;base64,{base64_string}"
        base64_qwen = f"data:image/jpeg;base64,{base64_string}" # 远程 API 使用标准图像数据格式

        return image, base64_qwen
    def extract_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join([c["text"] for c in content if c.get("type") == "text"])
        return ""
    def generate(self, query, image):
        content = [{
            "type": "text",
            "text": query
        }]
        if image != '' and image is not None:
            # 如果是单张图片，包成 list
            if not isinstance(image, (list, tuple)):
                image = [image]

            # filepaths = [Path(img).resolve().as_posix() for img in image]
            # for filepath in filepaths:
            #     base64_image = _encode_image(filepath)
            #     content.append({
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/jpeg;base64,{base64_image}"}}
            #         )
            if image:
                for img in image:
                    if isinstance(img, (str, Path)):  # 路径
                        # base64_image = _encode_image(img)
                        image_raw = Image.open(img)
                        image_input, base64_image = self.process_image(image_raw)
                    elif isinstance(img, Image.Image):  # PIL.Image
                        image_input, base64_image = self.process_image(img)
                    else:
                        raise ValueError(f"不支持的图片类型: {type(img)}")

                    # 预处理过
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_image}
                    })
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": ""},
                {"role": "user", "content": content}
            ],
            # repetition_penalty=1.1, #ViDoRAG 实验的可选重复惩罚参数
            # reasoning={"effort": "none"} # 仅在服务支持 reasoning 参数时启用
            # temperature=0, # 使用确定性解码以保持实验设置一致
            # top_p=0.95
        )
        # import pdb;pdb.set_trace()
        # token 统计
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        print(self.model_name + "/Prompt tokens:", input_tokens)
        print(self.model_name + "/Completion tokens:", output_tokens)
        print(self.model_name + "/Total tokens:", total_tokens)

        log_input = self.extract_text(content)
        log_output = completion.choices[0].message.content

        record = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        filename = f"{PROJECT_ROOT}/results/api_use/{self.model_name}_used.jsonl"
        if os.environ.get("FINMRAG_LOG_USAGE") == "1":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return completion.choices[0].message.content

class LLM:
    def __init__(self, model_name, base_url=None):
        self.model_name = model_name
        if any(name in model_name for name in ('Qwen2.5-VL', 'Qwen3-VL', 'InternVL', 'Llama', 'MiniCPM')):
            self.model = Qwen_VL_2_5(model_name, base_url)
        elif (model_name.startswith(('gpt', 'claude', 'gemini'))
              or (model_name.startswith('o') and model_name[1:2].isdigit())):
            self.model = GPT_LOCAL(model_name)
        elif base_url:
            # OpenAI-compatible services may expose a custom served-model name.
            self.model = Qwen_VL_2_5(model_name, base_url)
        else:
            raise ValueError(f"Unsupported model {model_name!r}; specify a compatible base_url")

    def generate(self, **kwargs):
        return self.model.generate(kwargs.get('query', ''), kwargs.get('image', ''))

class LLM2:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = Qwen_LM_2_5(model_name)

    def generate(self, **kwargs):
        query = kwargs.get("query", "")
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        return self.model.generate(query)


def load_and_prepare_query():
    base_path = os.path.join(PROJECT_ROOT, 'examples/ppocr')
    filenames = [
        "873001-纬达光电2022年年度报告_2023-03-24_74.txt",
        "873001-纬达光电2022年年度报告_2023-03-24_75.txt",
        "873001-纬达光电2022年年度报告_2023-03-24_11.txt",
        "873001-纬达光电2022年年度报告_2023-03-24_18.txt",
        "873001-纬达光电2022年年度报告_2023-03-24_10.txt",
    ]

    contents = []
    for fname in filenames:
        file_path = base_path + fname
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                contents.append(f.read())
        except Exception as e:
            raise FileNotFoundError(f"无法读取文件 {file_path}：{e}")

    full_text = "\n".join(contents)
    question = "纬达光电2022年末的流动比率是多少？通过计算获得，不要直接提取"

    return full_text + "\n\n根据以上检索到的信息，回答问题：" + question


if __name__ == "__main__":
    # llm = LLM("Qwen/Qwen3-VL-8B-Instruct", base_url="http://127.0.0.1:8001/v1")
    llm = LLM('claude-sonnet-4-5-20250929')
    base_path = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport/img/')
    evidence_pages = [
        "AMZN_10K_2023_38.jpg",
        "AMZN_10K_2023_40.jpg",
        "AMZN_10K_2023_41.jpg",
        "AMZN_10K_2024_37.jpg",
        "AMZN_10K_2024_39.jpg",
        "AMZN_10K_2024_40.jpg",
    ]
    imagelist = [base_path + filename for filename in evidence_pages]

    response = llm.generate(
        query="Calculate Amazon.com, Inc.’s [revenue growth rate, net income growth rate, EPS growth rate, book value per share] in 2022, 2023, and 2024 given the attached data. Output the results in a markdown-formatted table. Use Item, 2022, 2023, and 2024 as the column headers. Express growth rates as decimals, and valuation ratios rounded to four decimal places.",
        image=imagelist,
    )

    # llm = LLM2("<PROJECT_PATH>")
    # llm = LLM2("<PROJECT_PATH>")
    # response = llm.generate(query="请解释一下大语言模型是如何训练的？")
    # query = load_and_prepare_query()
    # response = llm.generate(query=query)

    print(response)
