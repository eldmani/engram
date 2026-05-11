from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Fact:
    id: str
    scope_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: Optional[str]
    object_value_json: Optional[Dict[str, Any]]
    fact_type: str
    valid_from: datetime
    valid_to: Optional[datetime]
    # truth_state: current | superseded | disputed | hidden
    truth_state: str
    source_episode_id: str
    confidence: float
    salience: float
    strength: float
    access_count: int
    last_accessed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
