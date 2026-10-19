
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
import asyncio
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

import sys
# <PROJECT_PATH>
# sys.path.append("<PROJECT_PATH>")
# from qwen3_vl_embedding import Qwen3VLEmbedder

def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


class VL_Embedding(MultiModalEmbedding):
    model: str = Field(description="The Multi-model to use.")

    api_key: Optional[str] = Field(
        default=None,
        description="The API key.",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in embedding-3 and later models. embedding-2 is fixed at 1024."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout.",
    )

    mode: str = Field(
        default="text",
        description="The mode of the model, either 'text' or 'image'.",
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show progress bars.",
    )

    embed_model: Union[ColQwen2, AutoModel, None] = Field(default=None)
    # embed_model: Union[ColQwen2, AutoModel, Qwen3VLEmbedder, None] = Field(default=None)
    processor: Optional[ColQwen2Processor] = Field(default=None)
    # processor: Union[ColQwen2Processor, AutoProcessor, None] = Field(default=None)
    tokenizer: Optional[AutoTokenizer] = Field(default=None)

    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )

        self.mode = mode

        if "openbmb" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
            self.embed_model = (
                AutoModel.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="cuda:1",
                )
                .cuda()
                .eval()
            )
            # self.embed_model.eval()
        elif "vidore" in model and "qwen" in model:
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # or "mps" if on Apple Silicon
                # device_map="auto",
            ).eval()
            # self.processor = ColQwen2Processor.from_pretrained(model, max_num_visual_tokens=5000)
            self.processor = ColQwen2Processor.from_pretrained(model)
            # 可选的 fast processor 配置，使用前需验证模型兼容性
            # self.processor = ColQwen2Processor.from_pretrained(model, use_fast=True)
        elif "vidore" in model and "pali" in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)
        # elif "Qwen3-VL-Embedding" in model:
        #     # Qwen3-VL-Embedding 模型支持
        #     self.embed_model = Qwen3VLEmbedder(model_name_or_path=model)

    @classmethod
    def class_name(cls) -> str:
        return "VL_Embedding"

    def embed_img(self, img_path):
        if isinstance(img_path, str):
            img_path = [img_path]
        if "vidore" in self.model:
            images = [Image.open(img) for img in img_path]
            batch_images = self.processor.process_images(images).to(
                self.embed_model.device
            )
            with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
        elif "openbmb" in self.model:
            images = [Image.open(img).convert("RGB") for img in img_path]
            inputs = {
                "text": [""] * len(images),
                "image": images,
                "tokenizer": self.tokenizer,
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)
                image_embeddings = (
                    F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                )
                # image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()[0]
            # image_embeddings = embeddings.tolist()[0]
        # elif "Qwen3-VL-Embedding" in self.model:
        #     # Qwen3-VL-Embedding 图像编码
        #     inputs = [{"image": img} for img in img_path]
        #     image_embeddings = self.embed_model.process(inputs)

        return image_embeddings

    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        if "colqwen" in self.model:
            batch_queries = self.processor.process_queries(text).to(
                self.embed_model.device
            )
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif "colpali" in self.model:
            batch_queries = self.processor.process_queries(text).to(
                self.embed_model.device
            )
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif "openbmb" in self.model:
            INSTRUCTION = (
                "Represent this query for retrieving relevant documents: "
            )
            queries = [INSTRUCTION + query for query in text]
            inputs = {
                "text": queries,
                "image": [None] * len(queries),
                "tokenizer": self.tokenizer,
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)
                # query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                query_embeddings = (
                    F.normalize(reps, p=2, dim=1).detach().cpu().tolist()
                )
                # query_embeddings = embeddings.tolist()[0]
        # elif "Qwen3-VL-Embedding" in self.model:
        #     # Qwen3-VL-Embedding 文本编码
        #     inputs = [{"text": t} for t in text]
        #     query_embeddings = self.embed_model.process(inputs)

        return query_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def __call__(self, nodes, **kwargs):
        if "vidore" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img(
                    [node.metadata["file_path"] for node in nodes]
                )
                embeddings = embeddings.view(embeddings.size(0), -1).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                embeddings = embeddings.view(embeddings.size(0), -1).tolist()

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        elif "openbmb" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img(
                    [node.metadata["file_path"] for node in nodes]
                )
                embeddings = embeddings.tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                # embeddings = embeddings.tolist()
                # embeddings = [embeddings]

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
        # elif "Qwen3-VL-Embedding" in self.model:
        #     if self.mode == "image":
        #         embeddings = self.embed_img(
        #             [node.metadata["file_path"] for node in nodes]
        #         )
        #         # Qwen3VLEmbedder.process 返回 numpy array，转换为 list of lists
        #         import numpy as np
        #         if isinstance(embeddings, np.ndarray):
        #             embeddings = embeddings.tolist()
        #         elif isinstance(embeddings, list):
        #             # 确保是 list of lists
        #             if len(embeddings) > 0 and not isinstance(embeddings[0], list):
        #                 embeddings = [embeddings]
        #     else:
        #         embeddings = self.embed_text([node.text for node in nodes])
        #         # Qwen3VLEmbedder.process 返回 numpy array，转换为 list of lists
        #         import numpy as np
        #         if isinstance(embeddings, np.ndarray):
        #             embeddings = embeddings.tolist()
        #         elif isinstance(embeddings, list):
        #             # 确保是 list of lists
        #             if len(embeddings) > 0 and not isinstance(embeddings[0], list):
        #                 embeddings = [embeddings]

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        return nodes

    def score(self, image_embeddings, text_embeddings):
        if "vidore" in self.model:
            score = self.processor.score_multi_vector(
                image_embeddings, text_embeddings
            )
        elif "openbmb" in self.model:
            score = text_embeddings @ image_embeddings.T
        # elif "Qwen3-VL-Embedding" in self.model:
        #     # Qwen3-VL-Embedding 使用点积计算相似度（与官方示例一致）
        #     import numpy as np
        #     if isinstance(text_embeddings, list):
        #         text_embeddings = np.array(text_embeddings)
        #     if isinstance(image_embeddings, list):
        #         image_embeddings = np.array(image_embeddings)
        #     score = text_embeddings @ image_embeddings.T
        #     # 转换为 torch tensor 以保持一致性
        #     if isinstance(score, np.ndarray):
        #         score = torch.from_numpy(score)
        return score


if __name__ == "__main__":
    # colpali = VL_Embedding("<PROJECT_PATH>")
    colpali = VL_Embedding('Qwen/Qwen3-VL-Embedding-2B')

    image_embeddings = colpali.embed_img(
        # "./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg"
        os.path.join(PROJECT_ROOT, 'examples/page_1_3311*4683.jpg')
    )
    text_embeddings = colpali.embed_text("Hello, world!")
    # score = colpali.processor.score_multi_vector(
    #     image_embeddings, text_embeddings
    # )
    score = colpali.score(image_embeddings, text_embeddings)
    # check shapes
    print(image_embeddings.shape)  # torch.Size([1, 779, 128])
    print(text_embeddings.shape)  # torch.Size([1, 26, 128])
    print(score)  # tensor([[115.5000]])
    # print(colpali.processor.image_processor.size)
