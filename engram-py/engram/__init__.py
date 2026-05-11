"""
engram — model-agnostic temporal memory store for AI agents.

Quick start::

    from engram import Engram

    mem = Engram()
    mem.store("Alice prefers dark mode.")
    result = mem.get_context("what does Alice prefer?")
    print(result.formatted)
"""

__version__ = "0.1.0"
from .config import EngramConfig
from .engram import Engram
from .extraction.base import BaseExtractor, ExtractionResult, ExtractedEntity, ExtractedFact, NullExtractor
from .models import Alias, Entity, Episode, Fact, QuarantinedFact, RetrievalTrace
from .results import (
    ContextResult,
    EntityView,
    EpisodeView,
    EvidenceView,
    ExplainResult,
    FactView,
    ForgetResult,
    RetrievalResult,
    StoreResult,
    TimelineEvent,
)

__all__ = [
    # Main class
    "Engram",
    "EngramConfig",
    # Extraction
    "BaseExtractor",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedFact",
    "NullExtractor",
    # Models
    "Episode",
    "Entity",
    "Fact",
    "Alias",
    "QuarantinedFact",
    "RetrievalTrace",
    # Results
    "StoreResult",
    "RetrievalResult",
    "ContextResult",
    "ForgetResult",
    "ExplainResult",
    "FactView",
    "EntityView",
    "EpisodeView",
    "EvidenceView",
    "TimelineEvent",
]
