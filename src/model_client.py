# -*- coding: utf-8 -*-
"""Adapter for Azure OpenAI services."""

from loguru import logger
from openai import AzureOpenAI, OpenAI

from src.config import settings


class ModelClient:
    """
    A client to interact with OpenAI models, supporting both Azure and standard OpenAI APIs.
    """

    def __init__(self):
        """
        Initializes the appropriate OpenAI client based on the API provider setting.
        """
        try:
            if settings.api_provider == "azure":
                logger.info("Initializing Azure OpenAI client.")
                self._client = AzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.openai_api_version,
                    azure_endpoint=settings.azure_openai_endpoint,
                )
                logger.info(f"Using Azure endpoint: {settings.azure_openai_endpoint}")
            elif settings.api_provider == "openai":
                logger.info("Initializing standard OpenAI client.")
                self._client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url, # Will be None if not set, which is fine
                )
                logger.info(f"Using OpenAI API. Base URL: {settings.openai_base_url or 'Default'}")
            else:
                raise ValueError(f"Unsupported API provider: {settings.api_provider}")
            
            logger.info("OpenAI client initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def get_embedding_client(self) -> "AzureOpenAI | OpenAI":
        """
        Provides access to the underlying client for embedding tasks.

        Returns:
            The configured AzureOpenAI or OpenAI client instance.
        """
        return self._client

    def get_llm_client(self) -> "AzureOpenAI | OpenAI":
        """
        Provides access to the underlying client for language model tasks.

        Returns:
            The configured AzureOpenAI or OpenAI client instance.
        """
        return self._client

    def health_check(self) -> bool:
        """
        Performs a simple health check by trying to list available models.

        Returns:
            True if the connection is successful, False otherwise.
        """
        try:
            logger.debug("Performing health check...")
            self._client.models.list()
            logger.success(
                "Health check passed. Connection to Azure OpenAI is successful."
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed. Could not connect to Azure OpenAI: {e}")
            return False


# A global instance for easy access across the application
model_client = ModelClient()
