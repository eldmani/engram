from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngramConfig:
    # Storage
    db_path: str = "engram.db"
    default_scope: str = "default"

    # Confidence thresholds
    extractor_confidence_min: float = 0.7
    resolution_confidence_min: float = 0.75
    supersession_confidence_min: float = 0.85

    # Retrieval defaults
    default_top_k: int = 10

    # Hybrid scoring weights (must sum to 1.0 for full calibration)
    # NOTE: weight_semantic is reserved for a future vector embedder.
    # Until set_embedder() is called, semantic_score is always 0.0 and
    # keyword scoring carries the retrieval signal for that weight's share.
    weight_semantic: float = 0.40
    weight_keyword: float = 0.30
    weight_graph: float = 0.10
    weight_temporal: float = 0.10
    weight_salience: float = 0.05
    weight_strength: float = 0.05

    # Memory dynamics
    reinforcement_boost: float = 0.05
    stable_access_threshold: int = 5
    decay_interval_hours: float = 24.0
    heat_promotion_threshold: float = 10.0
