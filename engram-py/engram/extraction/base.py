from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    name: str
    type: str = "unknown"
    confidence: float = 1.0


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    fact_type: str = "assertion"


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    confidence: float = 1.0


class BaseExtractor(ABC):
    """
    Plug-in contract for entity and fact extraction.

    Implement this class with any LLM or rule-based backend. The extractor
    receives normalized text and returns entities and candidate facts. It does
    NOT write to the store — the ingestion pipeline does that.
    """

    @abstractmethod
    def extract(self, data: str | dict) -> ExtractionResult:
        """Extract entities and candidate facts from input data."""
        ...


class NullExtractor(BaseExtractor):
    """
    Default extractor that performs no extraction.

    With NullExtractor, store() still creates immutable episodes that are
    indexed by FTS and retrievable by keyword search. Swap in a real extractor
    to activate entity resolution and temporal facts.
    """

    def extract(self, data: str | dict) -> ExtractionResult:
        return ExtractionResult()
