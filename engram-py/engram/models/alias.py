from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Alias:
    id: str
    entity_id: str
    value: str
    normalized_value: str
    embedding_id: Optional[str]
    source_episode_id: str
    confidence: float
    created_at: datetime
