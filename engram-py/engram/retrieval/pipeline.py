from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from ..config import EngramConfig
from ..models import Entity, Episode, Fact, RetrievalTrace
from ..results import (
    ContextResult,
    EntityView,
    EpisodeView,
    EvidenceView,
    FactView,
    RetrievalResult,
    TimelineEvent,
)
from ..storage.sqlite_store import SQLiteStore

_log = logging.getLogger(__name__)


def _temporal_score(fact: Fact, now: datetime) -> float:
    """Recency bonus: facts valid more recently score higher."""
    if fact.valid_from is None:
        return 0.0
    age_days = max(0.0, (now - fact.valid_from).total_seconds() / 86400)
    return 1.0 / (1.0 + age_days)


def _fuse(
    keyword: float,
    semantic: float,
    temporal: float,
    salience: float,
    strength: float,
    cfg: EngramConfig,
) -> float:
    return (
        cfg.weight_keyword * keyword
        + cfg.weight_semantic * semantic
        + cfg.weight_temporal * temporal
        + cfg.weight_salience * salience
        + cfg.weight_strength * strength
    )


class RetrievalPipeline:
    def __init__(self, store: SQLiteStore, config: EngramConfig) -> None:
        self._store = store
        self._cfg = config

    def retrieve(
        self,
        query: str,
        top_k: int,
        scope_id: str,
        as_of: datetime | None,
    ) -> RetrievalResult:
        now = as_of or datetime.now(timezone.utc)

        # --- keyword search over episodes ---
        ep_hits = self._store.fts_search_episodes(query, scope_id, top_k * 3)

        # --- keyword search over facts ---
        fact_hits = self._store.fts_search_facts(query, scope_id, top_k * 3, as_of)

        # --- top current facts in scope (for context even without keyword match) ---
        current_facts = self._store.get_current_facts_for_scope(scope_id, top_k)

        # Build scored candidates
        scored_facts: dict[str, tuple[Fact, float]] = {}

        for fact, kw in fact_hits:
            t = _temporal_score(fact, now)
            score = _fuse(kw, 0.0, t, fact.salience, fact.strength, self._cfg)
            scored_facts[fact.id] = (fact, score)

        for fact in current_facts:
            if fact.id not in scored_facts:
                t = _temporal_score(fact, now)
                score = _fuse(0.0, 0.0, t, fact.salience, fact.strength, self._cfg)
                scored_facts[fact.id] = (fact, score)

        ranked_facts = sorted(
            scored_facts.values(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Score episodes (FTS hits first, then fallback to recent)
        scored_eps: dict[str, tuple[Episode, float]] = {}
        for ep, kw in ep_hits:
            scored_eps[ep.id] = (ep, kw)
        if not scored_eps:
            for ep in self._store.get_recent_episodes(scope_id, top_k):
                scored_eps.setdefault(ep.id, (ep, 0.0))

        # Collect unique entities from ranked facts
        entity_ids: set[str] = set()
        for f, _ in ranked_facts:
            entity_ids.add(f.subject_entity_id)
            if f.object_entity_id:
                entity_ids.add(f.object_entity_id)
        entities = [e for eid in entity_ids if (e := self._store.get_entity(eid))]

        # Build and persist retrieval traces
        trace_ids: list[str] = []
        scores: dict[str, float] = {}
        for fact, score in ranked_facts:
            t = _temporal_score(fact, now)
            trace = RetrievalTrace(
                id=str(uuid4()),
                query=query,
                scope_id=scope_id,
                candidate_fact_id=fact.id,
                semantic_score=0.0,  # reserved — plug in an embedder to populate
                keyword_score=scored_facts.get(fact.id, (None, 0.0))[1],
                graph_score=0.0,
                temporal_score=t,
                final_score=score,
                matched_entities=[fact.subject_entity_id],
                source_episode_ids=[fact.source_episode_id],
                created_at=datetime.now(timezone.utc),
            )
            self._store.insert_retrieval_trace(trace)
            self._store.reinforce_fact(fact.id, self._cfg.reinforcement_boost)
            trace_ids.append(trace.id)
            scores[fact.id] = score

        for ep, kw in scored_eps.values():
            scores[ep.id] = kw

        return RetrievalResult(
            facts=[f for f, _ in ranked_facts],
            entities=entities,
            episodes=[e for e, _ in scored_eps.values()][:top_k],
            scores=scores,
            trace_ids=trace_ids,
        )

    def get_context(
        self,
        query: str,
        top_k: int,
        scope_id: str,
        as_of: datetime | None,
        template: str,
    ) -> ContextResult:
        result = self.retrieve(query, top_k, scope_id, as_of)
        now = as_of or datetime.now(timezone.utc)

        # Build entity lookup for display names
        entity_names: dict[str, str] = {
            e.id: e.canonical_name for e in result.entities
        }

        # Facts block
        facts_block: list[FactView] = []
        for fact in result.facts:
            subj = entity_names.get(fact.subject_entity_id, fact.subject_entity_id)
            obj = (
                entity_names.get(fact.object_entity_id, fact.object_entity_id)
                if fact.object_entity_id
                else str(fact.object_value_json.get("value", ""))
                if fact.object_value_json
                else ""
            )
            facts_block.append(FactView(
                fact_id=fact.id,
                subject=subj,
                predicate=fact.predicate,
                object=obj,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                truth_state=fact.truth_state,
                confidence=fact.confidence,
                salience=fact.salience,
                score=result.scores.get(fact.id, 0.0),
            ))

        # Entities block
        entities_block: list[EntityView] = [
            EntityView(
                entity_id=e.id,
                canonical_name=e.canonical_name,
                type=e.type,
                summary=e.summary,
                salience=e.salience,
            )
            for e in result.entities
        ]

        # Timeline — facts ordered by valid_from
        timeline_block: list[TimelineEvent] = sorted(
            [
                TimelineEvent(
                    timestamp=f.valid_from,
                    description=f"{entity_names.get(f.subject_entity_id, '?')} {f.predicate} "
                    + (
                        entity_names.get(f.object_entity_id, "")
                        if f.object_entity_id
                        else str(f.object_value_json.get("value", ""))
                        if f.object_value_json
                        else ""
                    ),
                    fact_id=f.id,
                )
                for f in result.facts
                if f.valid_from
            ],
            key=lambda e: e.timestamp,
        )

        # Episodes block
        episodes_block: list[EpisodeView] = [
            EpisodeView(
                episode_id=ep.id,
                raw_text=ep.raw_text,
                created_at=ep.created_at,
                source=ep.source,
                score=result.scores.get(ep.id, 0.0),
            )
            for ep in result.episodes
        ]

        # Evidence block
        evidence_block: list[EvidenceView] = []
        for fact in result.facts:
            evidence_block.append(EvidenceView(
                fact_id=fact.id,
                episode_ids=[fact.source_episode_id],
                description=f"Derived from episode {fact.source_episode_id[:8]}...",
            ))
        if result.episodes and not result.facts:
            evidence_block.append(EvidenceView(
                episode_ids=[ep.id for ep in result.episodes],
                description="Derived from episodes " + ", ".join(
                    ep.id[:8] + "..." for ep in result.episodes[:5]
                ),
            ))

        # Overall confidence
        if result.facts:
            confidence = sum(f.confidence for f in result.facts) / len(result.facts)
        elif result.episodes:
            confidence = 0.5
        else:
            confidence = 0.0

        formatted = _format(facts_block, entities_block, timeline_block, episodes_block, evidence_block)

        return ContextResult(
            facts_block=facts_block,
            entities_block=entities_block,
            timeline_block=timeline_block,
            episodes_block=episodes_block,
            evidence_block=evidence_block,
            confidence=confidence,
            formatted=formatted,
        )


def _format(
    facts: list[FactView],
    entities: list[EntityView],
    timeline: list[TimelineEvent],
    episodes: list[EpisodeView],
    evidence: list[EvidenceView],
) -> str:
    lines: list[str] = []

    if facts:
        lines.append("[FACTS]")
        for f in facts:
            lines.append(f"- {f.subject} {f.predicate} {f.object}.")
        lines.append("")

    if entities and any(e.summary for e in entities):
        lines.append("[ENTITIES]")
        for e in entities:
            if e.summary:
                lines.append(f"- {e.canonical_name}: {e.summary}")
        lines.append("")

    if timeline and len(timeline) > 1:
        lines.append("[TIMELINE]")
        for event in timeline:
            lines.append(f"- [{event.timestamp.strftime('%Y-%m-%d')}] {event.description}.")
        lines.append("")

    if episodes:
        lines.append("[EPISODES]")
        for ep in episodes:
            lines.append(f"- {ep.raw_text[:200].rstrip()}")
        lines.append("")

    if evidence:
        lines.append("[EVIDENCE]")
        for ev in evidence:
            lines.append(f"- {ev.description}")

    return "\n".join(lines).strip()
