# -*- coding: utf-8 -*-
"""Configuration management for the CLU data cleaning project."""

import os
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Manages application settings using environment variables."""

    # API Provider Selection
    api_provider: Literal["azure", "openai"] = Field(
        "azure", description="API provider to use ('azure' or 'openai')."
    )

    # Azure OpenAI Settings (required if api_provider is 'azure')
    azure_openai_endpoint: Optional[str] = Field(None, description="Azure OpenAI endpoint.")
    azure_openai_api_key: Optional[str] = Field(None, description="Azure OpenAI API key.")
    
    # Standard OpenAI Settings (required if api_provider is 'openai')
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key.")
    openai_base_url: Optional[str] = Field(None, description="Optional OpenAI base URL for custom endpoints.")

    # Provider-specific model names from .env.example
    azure_embedding_model: Optional[str] = Field(None, description="Deployment name for Azure embedding model.")
    azure_llm_model: Optional[str] = Field(None, description="Deployment name for Azure LLM.")
    openai_embedding_model: str = Field("text-embedding-3-large", description="Model name for OpenAI embedding.")
    openai_llm_model: str = Field("gpt-4o", description="Model name for OpenAI LLM.")

    # Generic model names used by the application, populated dynamically
    embedding_model: str = ""
    llm_model: str = ""
    
    # Common Model Settings
    openai_api_version: str = Field(
        "2023-12-01-preview", description="OpenAI API version (primarily for Azure)."
    )

    # Logging Settings from fds.md
    log_level: str = Field("INFO", description="Logging level.")
    log_dir: str = Field("logs", description="Directory to store log files.")
    log_rotate: str = Field("00:00", description="Log rotation policy (time or size).")
    log_retention: str = Field("14 days", description="Log retention period.")
    log_compress: str = Field("zip", description="Log compression format.")

    # Application Settings
    run_id: Optional[str] = Field(
        None, description="A unique ID for the current run, injected for logging."
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @model_validator(mode='after')
    def setup_models_and_check_keys(self) -> 'Settings':
        if self.api_provider == 'azure':
            if not self.azure_openai_endpoint or not self.azure_openai_api_key:
                raise ValueError("For 'azure' provider, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")
            if not self.azure_embedding_model or not self.azure_llm_model:
                raise ValueError("For 'azure' provider, AZURE_EMBEDDING_MODEL and AZURE_LLM_MODEL must be set.")
            # Dynamically set the generic model names
            self.embedding_model = self.azure_embedding_model
            self.llm_model = self.azure_llm_model

        elif self.api_provider == 'openai':
            if not self.openai_api_key:
                raise ValueError("For 'openai' provider, OPENAI_API_KEY must be set.")
            # Dynamically set the generic model names
            self.embedding_model = self.openai_embedding_model
            self.llm_model = self.openai_llm_model
        
        else:
            raise ValueError(f"Invalid API_PROVIDER: '{self.api_provider}'. Must be 'azure' or 'openai'.")
            
        return self

# Instantiate the settings object to be used throughout the application.
settings = Settings()
