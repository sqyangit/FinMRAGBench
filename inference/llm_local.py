
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
import torch
from PIL import Image
import sys
import base64
from io import BytesIO
from openai import OpenAI
from transformers import Qwen2_5_VLProcessor
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _encode_image(image_path):
    if isinstance(image_path,Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        base64_encoded = base64.b64encode(img_data).decode("utf-8")
        return base64_encoded
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

# vLLM 远程模型客户端
class Qwen_VLM_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-32B-Instruct", base_url=None, api_key="dummy"):
        # vLLM 提供 OpenAI API 兼容接口
        if base_url is not None:
            pass
        # 未指定服务地址时按模型规模选择默认端口
        elif "Qwen" in model_name:
            base_url="http://127.0.0.1:8001/v1"
        else: 
            pass
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
    def generate(self, query, image):
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # 组装 messages
        content = []
        content.append({"type": "text", "text": query})

        if image:
            # 如果是单张图片，包成 list
            if not isinstance(image, (list, tuple)):
                image = [image]

            for img in image:
                if isinstance(img, (str, Path)):  # 路径
                    base64_image = _encode_image(img)
                elif isinstance(img, Image.Image):  # PIL.Image
                    base64_image = _encode_image(img)
                else:
                    raise ValueError(f"不支持的图片类型: {type(img)}")

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

        messages = [{"role": "user", "content": content}]

        # 调用 vLLM OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,   # 可调
            # temperature=0
        )
        
        # token 统计
        is_calculate_token = False
        if is_calculate_token:
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Completion tokens:", response.usage.completion_tokens)
            print("Total tokens:", response.usage.total_tokens)

        return response.choices[0].message.content

class Qwen_LLM_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

    def generate(self, query, max_new_tokens=4096):
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # 构造 prompt
        messages = [
            {"role": "user", "content": query}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 模型生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # 也可以设置为 True 以获得多样性更强的回答
        )

        # 去掉 prompt 部分，只返回新生成内容
        generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response

class GPT_LOCAL:
    def __init__(self, model_name="gpt-4o"):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""), # gpt
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    def extract_text(self, content): # 只用于统计token
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
            filepaths = [Path(img).resolve().as_posix() for img in image]

            for filepath in filepaths:
                base64_image = _encode_image(filepath)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}}
                    )
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": content}
            ],
            # temperature=0,
        )

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
    
class VLM:
    def __init__(self, model_name, base_url=None):
        self.model_name = model_name
        if any(name in model_name for name in ('Qwen2.5-VL', 'Qwen3-VL', 'InternVL', 'Llama', 'MiniCPM')):
            self.model = Qwen_VLM_2_5(model_name, base_url)
        elif (model_name.startswith(('gpt', 'claude', 'gemini'))
              or (model_name.startswith('o') and model_name[1:2].isdigit())):
            self.model = GPT_LOCAL(model_name)
        elif base_url:
            # OpenAI-compatible services may expose a custom served-model name.
            self.model = Qwen_VLM_2_5(model_name, base_url)
        else:
            raise ValueError(f"Unsupported model {model_name!r}; specify a compatible base_url")

    def generate(self, **kwargs):
        return self.model.generate(kwargs.get('query', ''), kwargs.get('image', ''))

# 纯文本生成客户端
class LLM:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = Qwen_LLM_2_5(model_name)

    def generate(self, **kwargs):
        query = kwargs.get("query", "")
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        return self.model.generate(query)



if __name__ == '__main__':
    # llm = VLM('Qwen/Qwen2.5-VL-7B-Instruct')
    llm = VLM('gpt-4o')
    response = llm.generate(query='请描述这张图片，用json格式返回一段中文描述信息即可',
    image=[
        # '<PROJECT_PATH>',# 16299 930
        # '<PROJECT_PATH>',# 4991 930
        os.path.join(PROJECT_ROOT, 'examples/page_1_828*1171.jpg'),#  1295 930
    ])
    print(response)