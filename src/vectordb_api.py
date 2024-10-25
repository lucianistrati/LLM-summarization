import logging
import pinecone
from os import environ
from typing import Any, List, Optional
from text_embedder_processors import TextEmbedderProcessor

# Setup environment configurations and logging
ENVIRONMENT = environ.get("ENVIRONMENT", "development")
logging.info("Starting Apadata in [%s] mode!", ENVIRONMENT)

PINECONE_API_KEY = environ.get("PINECONE_API_KEY", "70c96cec-121a-46ce-9912-83db7b1")
PINECONE_ENVIRONMENT = environ.get("PINECONE_ENVIRONMENT", "gcp-starter")


def flatten(nested_list: List[Any]) -> List[Any]:
    """Flattens a nested list into a single list of elements."""
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
    """Singleton metaclass to ensure only one instance exists per class."""
    
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class VectorDBAPI(metaclass=Singleton):
    """
    Interface for Pinecone Vector Database API.
    Manages embedding storage and retrieval using Pinecone's vector DB.
    """

    def __init__(
        self,
        index_name: str = "main",
        dimension: int = 768,
        metric: str = "cosine",
        renew_index: bool = False,
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.renew_index = renew_index
        self.text_embedder = TextEmbedderProcessor("")

        self.set_pinecone_api()
        self._initialize_index()

    @staticmethod
    def set_pinecone_api():
        """Initialize Pinecone API with specified environment settings."""
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    def _initialize_index(self):
        """
        Initialize or renew the Pinecone index. Creates or reinitializes
        the index based on `renew_index` flag.
        """
        if self.index_name not in pinecone.list_indexes():
            self.create_index()
        elif self.renew_index:
            self.delete_index()
            self.create_index()
        self.index = pinecone.Index(self.index_name)

    def create_index(self) -> Any:
        """Create a new index in Pinecone with specified dimensions and metric."""
        return pinecone.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric
        )

    def insert(self, text: str, embedding: Optional[Any] = None) -> Any:
        """
        Insert an embedding into the index. Generates an embedding if none is provided.

        Args:
            text (str): The text to embed.
            embedding (Optional[Any]): Precomputed embedding vector. If None, generates embedding.

        Returns:
            Any: The embedding vector used for insertion.
        """
        if embedding is None:
            self.text_embedder.text = text
            embedding = list(flatten(self.text_embedder.process()))
        truncated_text = text[:511]  # Truncate text for Pinecone compatibility
        self.index.upsert([(truncated_text, embedding)])
        return embedding

    def query(self, embedding: Optional[Any], top_k: int = 3) -> Any:
        """
        Query the vector database for the closest embeddings to the provided vector.

        Args:
            embedding (Any): The embedding vector to query with.
            top_k (int): The number of closest matches to retrieve.

        Returns:
            Any: The query result from Pinecone.
        """
        return self.index.query(vector=embedding, top_k=top_k)

    def delete_index(self) -> Any:
        """Delete the current index from Pinecone."""
        return pinecone.delete_index(self.index_name)
