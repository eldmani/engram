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
    # weight_semantic is used only when set_embedder() is called with a
    # real embedder. With the default NullEmbedder, semantic_score is 0.0.
    # weight_graph reflects N-hop entity graph traversal scores (see graph_max_hops).
    weight_semantic: float = 0.40
    weight_keyword: float = 0.30
    weight_graph: float = 0.10
    weight_temporal: float = 0.10
    weight_salience: float = 0.05
    weight_strength: float = 0.05

    # Graph traversal depth for hybrid retrieval (2 = seed + 2-hop neighbours)
    graph_max_hops: int = 2

    # Memory dynamics
    reinforcement_boost: float = 0.05
    stable_access_threshold: int = 5
    decay_interval_hours: float = 24.0
    heat_promotion_threshold: float = 10.0
