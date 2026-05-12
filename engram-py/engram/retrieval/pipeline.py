from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from ..config import EngramConfig
from ..embedding.base import BaseEmbedder, NullEmbedder, cosine_similarity
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
    graph: float,
    temporal: float,
    salience: float,
    strength: float,
    cfg: EngramConfig,
) -> float:
    return (
        cfg.weight_keyword * keyword
        + cfg.weight_semantic * semantic
        + cfg.weight_graph * graph
        + cfg.weight_temporal * temporal
        + cfg.weight_salience * salience
        + cfg.weight_strength * strength
    )


class RetrievalPipeline:
    def __init__(
        self,
        store: SQLiteStore,
        config: EngramConfig,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self._store = store
        self._cfg = config
        self._embedder: BaseEmbedder = embedder or NullEmbedder()

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
        # Track per-fact score components for trace persistence
        kw_scores: dict[str, float] = {}
        sem_scores: dict[str, float] = {}
        graph_scores: dict[str, float] = {}

        for fact, kw in fact_hits:
            t = _temporal_score(fact, now)
            score = _fuse(kw, 0.0, 0.0, t, fact.salience, fact.strength, self._cfg)
            scored_facts[fact.id] = (fact, score)
            kw_scores[fact.id] = kw

        for fact in current_facts:
            if fact.id not in scored_facts:
                t = _temporal_score(fact, now)
                score = _fuse(0.0, 0.0, 0.0, t, fact.salience, fact.strength, self._cfg)
                scored_facts[fact.id] = (fact, score)

        # --- semantic search (active only when a real embedder is set) ---
        if not isinstance(self._embedder, NullEmbedder) and query:
            try:
                q_vecs = self._embedder.embed([query])
                q_vec = q_vecs[0] if q_vecs else []
            except Exception:
                _log.warning("Embedder raised during query embedding; skipping.", exc_info=True)
                q_vec = []

            if q_vec:
                emb_fact_ids = self._store.get_embedded_fact_ids_for_scope(scope_id)
                emb_pairs = self._store.get_embeddings_for_refs("fact", emb_fact_ids)
                for ref_id, vec in emb_pairs:
                    sem = cosine_similarity(q_vec, vec)
                    sem_scores[ref_id] = sem
                    if ref_id in scored_facts:
                        fact, old_score = scored_facts[ref_id]
                        kw = kw_scores.get(ref_id, 0.0)
                        t = _temporal_score(fact, now)
                        new_score = _fuse(kw, sem, 0.0, t, fact.salience, fact.strength, self._cfg)
                        scored_facts[ref_id] = (fact, new_score)
                    else:
                        fact = self._store.get_fact(ref_id)
                        if fact:
                            t = _temporal_score(fact, now)
                            score = _fuse(0.0, sem, 0.0, t, fact.salience, fact.strength, self._cfg)
                            scored_facts[ref_id] = (fact, score)

        # --- graph traversal: N-hop BFS from primary candidate entities ---
        # Graph score decays with hop distance: g = 1.0 / (1.0 + hop * 0.5)
        # hop 0 (seed entity) → 1.0 | hop 1 → 0.667 | hop 2 → 0.5 | …
        seed_entity_ids = list({
            entity_id
            for fact, _ in scored_facts.values()
            for entity_id in [fact.subject_entity_id,
                               *(([fact.object_entity_id]) if fact.object_entity_id else [])]
        })
        if seed_entity_ids:
            entity_graph = self._store.get_entity_graph(
                seed_entity_ids, scope_id, self._cfg.graph_max_hops
            )
            # Collect all non-seed entities reachable within max_hops
            neighbor_entity_ids = [eid for eid, hop in entity_graph.items() if hop > 0]
            if neighbor_entity_ids:
                neighbor_facts = self._store.get_facts_for_entities(
                    neighbor_entity_ids, scope_id
                )
                for fact in neighbor_facts:
                    subj_hop = entity_graph.get(fact.subject_entity_id, self._cfg.graph_max_hops + 1)
                    obj_hop = (
                        entity_graph.get(fact.object_entity_id, self._cfg.graph_max_hops + 1)
                        if fact.object_entity_id
                        else self._cfg.graph_max_hops + 1
                    )
                    min_hop = min(subj_hop, obj_hop)
                    g = 1.0 / (1.0 + min_hop * 0.5)
                    if fact.id in scored_facts:
                        # Already scored — add graph bonus if it improves the score
                        existing_g = graph_scores.get(fact.id, 0.0)
                        if g > existing_g:
                            graph_scores[fact.id] = g
                            old_fact, old_score = scored_facts[fact.id]
                            scored_facts[fact.id] = (
                                old_fact,
                                old_score + self._cfg.weight_graph * (g - existing_g),
                            )
                    else:
                        # New fact reachable only via graph traversal
                        graph_scores[fact.id] = g
                        t = _temporal_score(fact, now)
                        score = _fuse(0.0, 0.0, g, t, fact.salience, fact.strength, self._cfg)
                        scored_facts[fact.id] = (fact, score)
            # Also apply graph bonus for already-scored facts whose entities appear in the graph
            for fact_id, (fact, _) in list(scored_facts.items()):
                if fact_id in graph_scores:
                    continue
                subj_hop = entity_graph.get(fact.subject_entity_id, self._cfg.graph_max_hops + 1)
                obj_hop = (
                    entity_graph.get(fact.object_entity_id, self._cfg.graph_max_hops + 1)
                    if fact.object_entity_id
                    else self._cfg.graph_max_hops + 1
                )
                min_hop = min(subj_hop, obj_hop)
                if min_hop <= self._cfg.graph_max_hops:
                    g = 1.0 / (1.0 + min_hop * 0.5)
                    graph_scores[fact_id] = g

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
                semantic_score=sem_scores.get(fact.id, 0.0),
                keyword_score=kw_scores.get(fact.id, 0.0),
                graph_score=graph_scores.get(fact.id, 0.0),
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

        # Memory blocks — always fetched and surfaced first in formatted output
        memory_blocks: dict[str, str] = self._store.get_all_blocks(scope_id)

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

        formatted = _format(facts_block, entities_block, timeline_block, episodes_block, evidence_block, memory_blocks)

        return ContextResult(
            facts_block=facts_block,
            entities_block=entities_block,
            timeline_block=timeline_block,
            episodes_block=episodes_block,
            evidence_block=evidence_block,
            memory_blocks=memory_blocks,
            confidence=confidence,
            formatted=formatted,
        )


def _format(
    facts: list[FactView],
    entities: list[EntityView],
    timeline: list[TimelineEvent],
    episodes: list[EpisodeView],
    evidence: list[EvidenceView],
    memory_blocks: dict[str, str] | None = None,
) -> str:
    lines: list[str] = []

    if memory_blocks:
        lines.append("[MEMORY]")
        for key, value in sorted(memory_blocks.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

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
