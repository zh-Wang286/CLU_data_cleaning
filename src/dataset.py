# -*- coding: utf-8 -*-
"""Dataset loading, validation, and access layer."""

from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger
from pydantic import ValidationError

from src.schemas import CLUProject, Intent, Utterance


class CLUDataset:
    """
    Handles loading, validation, and providing access to the CLU dataset.
    """

    def __init__(self, project_data: CLUProject):
        """
        Initializes the dataset with validated project data.

        Args:
            project_data: A CLUProject object populated from the source JSON.
        """
        self._project = project_data
        self._intents_map: Dict[str, Intent] = {
            intent.category: intent for intent in self._project.assets.intents
        }
        self._utterances_by_intent: Dict[str, List[Utterance]] = (
            self._group_utterances_by_intent()
        )
        logger.info(
            f"Dataset initialized for project: '{self._project.metadata.projectName}'"
        )
        self._validate_utterance_intents()

    @classmethod
    def from_json(cls, file_path: Path) -> "CLUDataset":
        """
        Loads and validates a CLU project from a JSON file.

        Args:
            file_path: The path to the JSON file.

        Returns:
            An instance of CLUDataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is malformed or fails validation.
        """
        logger.info(f"Loading CLU dataset from: {file_path}")
        if not file_path.exists():
            logger.error(f"Dataset file not found at: {file_path}")
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            project_data = CLUProject.model_validate(data)
            logger.success("Successfully validated JSON against CLUProject schema.")
            return cls(project_data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in {file_path}") from e
        except ValidationError as e:
            logger.error(f"Schema validation failed for {file_path}:\n{e}")
            raise ValueError(f"Schema validation failed for {file_path}") from e

    def _group_utterances_by_intent(self) -> Dict[str, List[Utterance]]:
        """Groups utterances by their assigned intent."""
        grouped: Dict[str, List[Utterance]] = {
            intent: [] for intent in self._intents_map
        }
        for utt in self._project.assets.utterances:
            if utt.intent in grouped:
                grouped[utt.intent].append(utt)
        return grouped

    def _validate_utterance_intents(self):
        """Logs a warning for any utterance pointing to a non-defined intent."""
        defined_intents = set(self._intents_map.keys())
        referenced_intents = {utt.intent for utt in self._project.assets.utterances}

        undefined = referenced_intents - defined_intents
        if undefined:
            logger.warning(
                f"Found utterances referencing non-existent intents: {undefined}"
            )

    def get_intents(self) -> List[Intent]:
        """Returns all intent definitions."""
        return self._project.assets.intents

    def get_utterances(self, intent: Optional[str] = None) -> List[Utterance]:
        """
        Returns utterances, optionally filtered by intent.

        Args:
            intent: If provided, returns utterances only for this intent.

        Returns:
            A list of Utterance objects.
        """
        if intent:
            return self._utterances_by_intent.get(intent, [])
        return self._project.assets.utterances

    def count_intents(self) -> int:
        """Returns the total number of defined intents."""
        return len(self._intents_map)

    def count_utterances(self, intent: Optional[str] = None) -> int:
        """
        Counts utterances, optionally filtered by intent.

        Args:
            intent: If provided, counts utterances only for this intent.

        Returns:
            The count of utterances.
        """
        if intent:
            return len(self._utterances_by_intent.get(intent, 0))
        return len(self._project.assets.utterances)

    def get_intent_counts(self) -> Dict[str, int]:
        """Returns a dictionary of utterance counts per intent."""
        return {
            intent: len(utterances)
            for intent, utterances in self._utterances_by_intent.items()
        }

    def warn_low_utterance_intents(self, threshold: int = 25) -> Dict[str, int]:
        """
        Finds and logs intents with fewer utterances than the threshold.

        Args:
            threshold: The minimum number of utterances required.

        Returns:
            A dictionary of intents and their counts that are below the threshold.
        """
        intent_counts = self.get_intent_counts()
        low_sample_intents = {
            intent: count
            for intent, count in intent_counts.items()
            if count < threshold
        }

        if low_sample_intents:
            logger.warning(
                f"Found {len(low_sample_intents)} intents with fewer than {threshold} utterances:"
            )
            for intent, count in low_sample_intents.items():
                logger.warning(f"  - Intent '{intent}': {count} utterances")
        else:
            logger.info(f"All intents have at least {threshold} utterances.")

        return low_sample_intents

    @property
    def project(self) -> CLUProject:
        """Returns the raw CLUProject object."""
        return self._project
