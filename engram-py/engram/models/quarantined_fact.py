from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class QuarantinedFact:
    id: str
    scope_id: str
    source_episode_id: str
    extracted_subject: str
    extracted_predicate: str
    extracted_object: str
    candidate_supersedes_fact_id: Optional[str]
    extractor_confidence: float
    resolution_confidence: float
    reason: str
    # status: pending | approved | rejected
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime]
