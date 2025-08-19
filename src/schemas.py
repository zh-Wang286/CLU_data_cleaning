# -*- coding: utf-8 -*-
"""Pydantic models for CLU project data structure."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """Represents a single utterance in the dataset."""

    text: str = Field(..., min_length=1, description="The text of the utterance.")
    language: Optional[str] = Field(None, description="The language of the utterance.")
    intent: str = Field(..., description="The intent assigned to the utterance.")
    entities: List[Dict[str, Any]] = Field(
        [], description="Entities associated with the utterance."
    )
    dataset: Optional[str] = Field(
        "Train", description="Dataset split: Train, Validate, Test, or Augment."
    )


class Intent(BaseModel):
    """Represents an intent definition."""

    category: str = Field(..., description="The name of the intent.")
    description: Optional[str] = Field(
        None, description="Optional description for the intent."
    )


class Entity(BaseModel):
    """Represents an entity definition. Can be complex."""

    category: str = Field(..., description="The name of the entity.")
    # The rest of the entity structure is complex and not essential for the current scope.
    # We keep it flexible using a dictionary.
    additional_properties: Dict[str, Any] = Field({}, alias="additionalProperties")

    class Config:
        """Pydantic config."""

        extra = "allow"


class Assets(BaseModel):
    """Represents the assets of a CLU project."""

    projectKind: str = Field(..., description="The kind of the project asset.")
    intents: List[Intent] = Field(..., description="List of intents.")
    entities: List[Dict[str, Any]] = Field(
        [], description="List of entities, kept as dict for flexibility."
    )
    utterances: List[Utterance] = Field(..., description="List of utterances.")


class Settings(BaseModel):
    """Represents the project settings."""

    confidenceThreshold: float
    normalizeCasing: bool
    augmentDiacritics: bool


class ProjectMetadata(BaseModel):
    """Represents the metadata of the CLU project."""

    projectKind: str
    settings: Settings
    projectName: str
    multilingual: bool
    description: Optional[str] = None
    language: str


class CLUProject(BaseModel):
    """Represents the entire CLU project structure."""

    projectFileVersion: str
    stringIndexType: str
    metadata: ProjectMetadata
    assets: Assets
