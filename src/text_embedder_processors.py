import openai
import torch
from sentence_transformers import SentenceTransformer

from typing import List

from enum import Enum as BaseEnum

from psqlextra.types import StrEnum as BaseStrEnum
from vertexai.language_models import TextEmbeddingModel
import vertexai

vertexai.init(project="intense-wares-407020", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.01,
    "top_p": 0.8,
    "top_k": 40
}


class Enum(BaseEnum):
    """Extends the base enum class with some useful methods."""

    @classmethod
    def all(cls) -> List["Enum"]:
        return [choice for choice in cls]  # pylint: disable=unnecessary-comprehension

    @classmethod
    def values(cls) -> List[int]:
        return [choice.value for choice in cls]


class StrEnum(BaseStrEnum):
    """String Enum class"""


class TextEmbeddingMethods(StrEnum):
    ST = "sentence_transformers"
    ADA = "ada-002"
    VERTEX = "vertex"


import logging
from os import environ

OPENAI_API_BASE = environ.get(
    "OPENAI_ENDPOINT", "https://apadua-openai.openai.azure.com/"
)
OPENAI_API_TYPE = environ.get("OPENAI_API_TYPE", "azure")
OPENAI_API_KEY = environ.get("OPENAI_API_KEY", "383d7b5cc97745a3bfafbcb6cc4da9a8")
OPEN_AI_VERSION = environ.get("OPENAI_API_VERSION", "2023-05-15")

ENVIRONMENT = environ.get("ENVIRONMENT", "development")


logging.info("Starting Apadata in [%s] mode!", ENVIRONMENT)

PINECONE_API_KEY = environ.get(
    "PINECONE_API_KEY", "70c96cec-121a-46ce-9912-83db7b4457b1"
)
PINECONE_ENVIRONMENT = environ.get("PINECONE_ENVIRONMENT", "gcp-starter")


from typing import Any

import abc


model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")


class TextProcessor(abc.ABC):
    def __init__(self, text: str):
        self.text = text

    @abc.abstractmethod
    def process(self) -> Any:
        pass


def load_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        model.cuda()
    return model


class TextEmbedderProcessor(TextProcessor):
    """
    This class will embed a text and return its embedding
    """

    MODEL = load_model()

    def __init__(self, text: str, method: str = TextEmbeddingMethods.VERTEX):
        super().__init__(text)
        self.method = method
        if self.method == TextEmbeddingMethods.ADA:
            self.set_openai_api()

    @staticmethod
    def set_openai_api():
        openai.api_base = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY
        openai.api_type = OPENAI_API_TYPE
        openai.api_version = OPEN_AI_VERSION

    def process(self) -> Any:
        if self.method == TextEmbeddingMethods.ST:
            return TextEmbedderProcessor.MODEL.encode(
                self.text, convert_to_tensor=False
            ).astype("float")
        elif self.method == TextEmbeddingMethods.ADA:
            response = openai.Embedding.create(
                input=self.text, model="text-embedding-ada-002",
                engine="text-embedding-ada-002"
            )
            embedding = response["data"][0]["embedding"]
            return embedding
        elif self.method == TextEmbeddingMethods.VERTEX:
            embeddings = model.get_embeddings([self.text])
            for embedding in embeddings:
                vector = embedding.values
            return vector
        else:
            raise ValueError(
                f"Wrong text embedding method given, it must be one of "
                f"the following: {TextEmbeddingMethods.values()}"
            )
