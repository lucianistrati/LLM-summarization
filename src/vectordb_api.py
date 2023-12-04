from typing import Any, Optional

import pinecone

from text_embedder_processors import TextEmbedderProcessor

# utils
from typing import Any, List


def flatten(nested_list: List[Any]) -> List[Any]:
    flat_list = []
    stack = [nested_list]

    while stack:
        current = stack.pop()
        for item in current:
            if isinstance(item, list):
                stack.append(item)
            else:
                flat_list.append(item)

    return flat_list


class Singleton(type):
    """
    Singleton metaclass.
    This can be added to any class to make it a singleton.

    e.g.
    class GlobalSettings(metaclass=Singleton):
        pass

    usage:
    settings1 = GlobalSettings()
    settings2 = GlobalSettings()
    settings1 is settings2  # True
    """

    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


import logging
from os import environ


ENVIRONMENT = environ.get("ENVIRONMENT", "development")
logging.info("Starting Apadata in [%s] mode!", ENVIRONMENT)


PINECONE_API_KEY = environ.get(
    "PINECONE_API_KEY", "70c96cec-121a-46ce-9912-83db7b4457b1"
)
PINECONE_ENVIRONMENT = environ.get("PINECONE_ENVIRONMENT", "gcp-starter")


class VectorDBAPI(metaclass=Singleton):
    """
    Class that gets requests results from Pinecone's Vector DB
    """

    def __init__(
        self,
        index_name: str = "main",
        dimension: int = 768,
        metric: str = "cosine",
        renew_index: bool = False,
    ):
        self.set_pinecone_api()
        self.index_name = index_name
        self.renew_index = renew_index
        self.metric = metric
        self.dimension = dimension
        if self.index_name not in pinecone.list_indexes():
            self.create_index()
        elif renew_index:
            self.delete_index()
            self.create_index()
        self.index = pinecone.Index(index_name=index_name)
        self.text_embedder = TextEmbedderProcessor("")

    @staticmethod
    def set_pinecone_api():
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    def create_index(self) -> Any:
        return pinecone.create_index(
            self.index_name, dimension=self.dimension, metric=self.metric
        )

    def insert(self, text: str, embedding: Optional[Any] = None) -> Any:
        if embedding is None:
            self.text_embedder.text = text
            embedding = list(flatten(self.text_embedder.process()))
        text = text[:min(511, len(text))]
        self.index.upsert([(text, embedding)])
        return embedding

    def query(self, embedding: Optional[Any] = None, top_k: int = 3) -> Any:
        return self.index.query(vector=embedding, top_k=top_k)

    def delete_index(self) -> Any:
        return pinecone.delete_index(self.index_name)
