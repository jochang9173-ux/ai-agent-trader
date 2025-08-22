"""
Unified LLM client for all modules
"""

import os
from typing import Optional

from langchain_openai import AzureChatOpenAI

from app.config import settings


def get_llm_client(
    temperature: float = 0.1, max_tokens: int = 4000, **kwargs
) -> AzureChatOpenAI:
    """
    Get unified Azure OpenAI client

    Args:
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens
        **kwargs: Other parameters

    Returns:
        AzureChatOpenAI client instance
    """
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


class LLMClientConfig:
    """LLM Client Configuration Class"""

    def __init__(
        self,
        deployment_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ):
        self.deployment_name = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.endpoint = endpoint or settings.AZURE_OPENAI_ENDPOINT
        self.api_version = api_version or settings.AZURE_OPENAI_API_VERSION
        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_client(self) -> AzureChatOpenAI:
        """Create client instance"""
        return AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


# Default configuration instance
default_config = LLMClientConfig()


def get_configured_client(config: Optional[LLMClientConfig] = None) -> AzureChatOpenAI:
    """
    Get client using configuration

    Args:
        config: LLM 客戶端配置

    Returns:
        配置好的 AzureChatOpenAI 客戶端
    """
    if config is None:
        config = default_config
    return config.create_client()
