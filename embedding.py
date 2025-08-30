import os
import torch

from typing import List, Any, Union
from pydantic import PrivateAttr

from transformers import AutoModel, AutoTokenizer
from langchain.embeddings.base import Embeddings
from llama_index.core.embeddings import BaseEmbedding

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class Embedding(Embeddings):
    def __init__(self, model_name="vinai/phobert-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
        self.normalize = True

    @torch.inference_mode()
    def encode(self, text: str):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy() 

    def embed_query(self, text: str):
        return self.encode(text)[0].tolist()
    
    def embed_documents(self, docs: List[str]):
        return self.encode(docs).tolist()


class LlamaIndexPhobertEmbedding(BaseEmbedding):
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 128
    normalize: bool = True

    _device: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False).to(self._device)
        self._model.eval()

    @torch.inference_mode()
    def _encode(self, text: Union[List[str], str]):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        inputs = self._tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().tolist()

    def _get_query_embedding(self, query: str):
        return self._encode([query])[0]

    def _get_text_embedding(self, text: str):
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: List[str]):
        if not texts:
            return []
        return self._encode(texts)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]):
        return self._get_text_embeddings(texts)
