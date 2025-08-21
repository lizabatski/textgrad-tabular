import os
import textgrad as tg
from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient

def get_engines():
    provider = os.getenv("TEXTGRAD_PROVIDER", "openai").lower()

    if provider == "deepseek":
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        client = OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_key)
        return (
            ChatExternalClient(client=client, model_string='deepseek-chat'),
            ChatExternalClient(client=client, model_string='deepseek-chat'),
            ChatExternalClient(client=client, model_string='deepseek-chat'),
            provider
        )
    elif provider == "openai":
        return (
            tg.get_engine("gpt-3.5-turbo"),
            tg.get_engine("gpt-4o"),
            tg.get_engine("gpt-4o"),
            provider
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
