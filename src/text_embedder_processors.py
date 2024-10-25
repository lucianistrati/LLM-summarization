import os
import openai
import torch
import vertexai
import logging
from enum import Enum as BaseEnum
from psqlextra.types import StrEnum as BaseStrEnum
from sentence_transformers import SentenceTransformer
from vertexai.language_models import TextEmbeddingModel
from typing import List, Any
import abc

# Vertex AI initialization
vertexai.init(project="intense-wares-407020", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.01,
    "top_p": 0.8,
    "top_k": 40
}

# Logging setup
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging.info("Starting Apadata in [%s] mode!", ENVIRONMENT)

# API keys and environment variables
OPENAI_API_BASE = os.getenv("OPENAI_ENDPOINT", "https://apadua-openai.openai.azure.com/")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "azure")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "383d7b5cc97745a3bfafbcb6cc4da9a8")
OPEN_AI_VERSION = os.getenv("OPENAI_API_VERSION", "2023-05-15")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "70c96cec-121a-46ce-9912-83db7b4457b1")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

class Enum(BaseEnum):
    """Extends the base enum class with methods to retrieve all choices or their values."""
    
    @classmethod
    def all(cls) -> List["Enum"]:
        return list(cls)

    @classmethod
    def values(cls) -> List[int]:
        return [choice.value for choice in cls]


class StrEnum(BaseStrEnum):
    """Extended string enum for easier type handling."""


class TextEmbeddingMethods(StrEnum):
    ST = "sentence_transformers"
    ADA = "ada-002"
    VERTEX = "vertex"


class TextProcessor(abc.ABC):
    """Abstract text processor base class."""
    
    def __init__(self, text: str):
        self.text = text

    @abc.abstractmethod
    def process(self) -> Any:
        """Abstract method for processing text to be implemented by subclasses."""
        pass


def load_model() -> SentenceTransformer:
    """Loads and returns a SentenceTransformer model, using GPU if available."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        model.cuda()
    return model


# Load model for sentence transformers
ST_MODEL = load_model()
# Load Vertex AI model
VERTEX_MODEL = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")


class TextEmbedderProcessor(TextProcessor):
    """Embeds a text using a specified embedding method."""
    
    def __init__(self, text: str, method: TextEmbeddingMethods = TextEmbeddingMethods.VERTEX):
        super().__init__(text)
        self.method = method
        if self.method == TextEmbeddingMethods.ADA:
            self.set_openai_api()

    @staticmethod
    def set_openai_api():
        """Configures OpenAI API settings based on environment variables."""
        openai.api_base = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY
        openai.api_type = OPENAI_API_TYPE
        openai.api_version = OPEN_AI_VERSION

    def process(self) -> Any:
        """
        Processes text based on the selected embedding method.
        
        Returns:
            Any: Embedding vector depending on the method.
            
        Raises:
            ValueError: If an invalid embedding method is specified.
        """
        if self.method == TextEmbeddingMethods.ST:
            return ST_MODEL.encode(self.text, convert_to_tensor=False).astype("float")
        
        elif self.method == TextEmbeddingMethods.ADA:
            response = openai.Embedding.create(input=self.text, model="text-embedding-ada-002")
            return response["data"][0]["embedding"]
        
        elif self.method == TextEmbeddingMethods.VERTEX:
            embeddings = VERTEX_MODEL.get_embeddings([self.text])
            return embeddings[0].values  # Assuming single embedding output for the text
        
        else:
            raise ValueError(f"Invalid embedding method '{self.method}', must be one of {TextEmbeddingMethods.values()}")
