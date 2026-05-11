from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .models import Entity, Fact, Episode, QuarantinedFact, RetrievalTrace


@dataclass
class StoreResult:
    episode_id: str
    entity_ids: List[str] = field(default_factory=list)
    fact_ids: List[str] = field(default_factory=list)
    quarantined_count: int = 0


@dataclass
class FactView:
    fact_id: str
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: Optional[datetime]
    truth_state: str
    confidence: float
    salience: float
    score: float = 0.0


@dataclass
class EntityView:
    entity_id: str
    canonical_name: str
    type: str
    summary: Optional[str]
    salience: float


@dataclass
class TimelineEvent:
    timestamp: datetime
    description: str
    fact_id: Optional[str] = None
    episode_id: Optional[str] = None


@dataclass
class EpisodeView:
    episode_id: str
    raw_text: str
    created_at: datetime
    source: str
    score: float = 0.0


@dataclass
class EvidenceView:
    episode_ids: List[str]
    description: str
    fact_id: Optional[str] = None


@dataclass
class RetrievalResult:
    facts: List[Any] = field(default_factory=list)          # List[Fact]
    entities: List[Any] = field(default_factory=list)       # List[Entity]
    episodes: List[Any] = field(default_factory=list)       # List[Episode]
    scores: Dict[str, float] = field(default_factory=dict)
    trace_ids: List[str] = field(default_factory=list)


@dataclass
class ContextResult:
    facts_block: List[FactView] = field(default_factory=list)
    entities_block: List[EntityView] = field(default_factory=list)
    timeline_block: List[TimelineEvent] = field(default_factory=list)
    episodes_block: List[EpisodeView] = field(default_factory=list)
    evidence_block: List[EvidenceView] = field(default_factory=list)
    confidence: float = 0.0
    formatted: str = ""


@dataclass
class ForgetResult:
    affected_facts: int = 0
    affected_entities: int = 0
    affected_aliases: int = 0


@dataclass
class ExplainResult:
    traces: List[Any] = field(default_factory=list)              # List[RetrievalTrace]
    quarantined_facts: List[Any] = field(default_factory=list)   # List[QuarantinedFact]
    summary: str = ""
