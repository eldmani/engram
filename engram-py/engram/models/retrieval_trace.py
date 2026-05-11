from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RetrievalTrace:
    id: str
    query: str
    scope_id: str
    candidate_fact_id: Optional[str]
    semantic_score: float
    keyword_score: float
    graph_score: float
    temporal_score: float
    final_score: float
    matched_entities: List[str]
    source_episode_ids: List[str]
    created_at: datetime
