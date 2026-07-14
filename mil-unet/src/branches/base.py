from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple


class BaseBranch(ABC):
    """Common interface for all 4 classification branches."""

    @abstractmethod
    def train(self, train_tool_ids: List[str], tool_index: Dict, config: Dict) -> None:
        """Train on images from train_tool_ids."""

    @abstractmethod
    def predict(self, tool_id: str, tool_index: Dict, config: Dict) -> Tuple[float, List[float]]:
        """Returns (tool_score, image_scores). tool_score >= 0.5 means worn."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model from disk."""
