from typing import Any, Dict, List

# Note: The openai-python library support for Azure OpenAI is in preview.
import openai
from openai.error import ServiceUnavailableError


from os import environ
OPENAI_API_BASE = environ.get(
    "OPENAI_ENDPOINT", "https://apadua-openai.openai.azure.com/"
)
CHATGPT_SUBSCRIPTION_ID = environ.get("CHATGPT_SUBSCRIPTION_ID")
OPENAI_API_TYPE = environ.get("OPENAI_API_TYPE", "azure")
OPENAI_API_KEY = environ.get("OPENAI_API_KEY", "383d7b5cc97745a3bfafbcb6cc4da9a8")
OPEN_AI_VERSION = environ.get("OPENAI_API_VERSION", "2023-05-15")


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


class ChatGpt(metaclass=Singleton):
    """ChatGpt communication class"""

    def __init__(self):
        openai.api_base = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY
        openai.api_type = OPENAI_API_TYPE
        openai.api_version = OPEN_AI_VERSION

    @staticmethod
    def get_output_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        output_messages = []
        choices = response["choices"]
        for choice in choices:
            message = choice["message"]
            content = message["content"]
            role = message["role"]
            output_message = {"content": content, "role": role}
            output_messages.append(output_message)
        return output_messages

    @staticmethod
    def default_configuration():
        return {
            "engine": "gpt35",
            "temperature": 0.01,
            "max_tokens": 200,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def request(self, messages: List[Dict[Any, Any]], **kwargs: Any) -> Any:
        try:
            return openai.ChatCompletion.create(
                messages=messages, **{**self.default_configuration(), **kwargs}
            )
        except ServiceUnavailableError:
            return None


gpt = ChatGpt()


def openai_summary(text):
    messages = [
        {"role": "system", "content": f"Summarize this text: "
                                      f"{text}"},
    ]
    return gpt.request(messages, temperature=0.01)
