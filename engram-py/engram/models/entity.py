from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Entity:
    id: str
    scope_id: str
    type: str
    canonical_name: str
    summary: Optional[str]
    created_at: datetime
    updated_at: datetime
    first_seen_at: datetime
    last_seen_at: datetime
    salience: float
    strength: float
    confidence: float
