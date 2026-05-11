from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from ..config import EngramConfig
from ..extraction.base import BaseExtractor, ExtractedFact
from ..models import Alias, Entity, Episode, Fact, QuarantinedFact
from ..results import StoreResult
from ..storage.sqlite_store import SQLiteStore

_log = logging.getLogger(__name__)

# Importance priors by fact type / heuristic category
_IMPORTANCE_PRIORS: dict[str, float] = {
    "identity": 1.0,
    "preference": 0.9,
    "project": 0.9,
    "event": 0.7,
    "assertion": 0.5,
    "small_talk": 0.2,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _salience(fact: ExtractedFact, novelty: float) -> float:
    base = 0.5
    importance = _IMPORTANCE_PRIORS.get(fact.fact_type, 0.5)
    return base * (1 + novelty) * (1 + importance)


class IngestionPipeline:
    def __init__(
        self,
        store: SQLiteStore,
        extractor: BaseExtractor,
        config: EngramConfig,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._config = config

    def run(
        self,
        data: str | dict,
        metadata: dict | None,
        scope_id: str,
    ) -> StoreResult:
        # Step 1 – normalize input
        raw_text = json.dumps(data) if isinstance(data, dict) else str(data)
        meta = metadata or {}

        # Step 2 – create episode (immutable)
        episode = Episode(
            id=str(uuid4()),
            scope_id=scope_id,
            source=meta.get("source", "api"),
            raw_text=raw_text,
            metadata=meta,
            created_at=datetime.utcnow(),
            checksum=Episode.compute_checksum(raw_text),
        )
        self._store.insert_episode(episode)

        result = StoreResult(episode_id=episode.id)

        # Step 3 – extract entities and candidate facts
        extraction = self._extractor.extract(raw_text)

        if not extraction.entities and not extraction.facts:
            # NullExtractor or extractor returned nothing — episode is still useful
            return result

        # Step 4 – resolve entities
        entity_map: dict[str, str] = {}  # extracted name → entity id
        for ext_entity in extraction.entities:
            entity_id = self._resolve_entity(ext_entity.name, ext_entity.type, scope_id, episode)
            entity_map[ext_entity.name] = entity_id
            result.entity_ids.append(entity_id)

        # Step 5 – process candidate facts through confidence gate
        now = datetime.now(timezone.utc)
        for ext_fact in extraction.facts:
            subject_id = entity_map.get(ext_fact.subject)
            if not subject_id:
                # Subject not resolved — quarantine
                self._quarantine(
                    ext_fact=ext_fact,
                    scope_id=scope_id,
                    episode_id=episode.id,
                    extractor_conf=extraction.confidence,
                    resolution_conf=0.0,
                    reason="subject entity not resolved",
                )
                result.quarantined_count += 1
                _log.info("Quarantined (no subject resolved): %s.%s", ext_fact.subject, ext_fact.predicate)
                continue

            # Check for existing current facts with same predicate
            existing = self._store.get_current_facts(scope_id, subject_id, ext_fact.predicate)
            supersedes_id = existing[0].id if existing else None
            is_supersession = supersedes_id is not None

            # Confidence gate
            ext_conf = extraction.confidence * ext_fact.confidence
            res_conf = ext_fact.confidence  # resolution confidence (entity was found)

            threshold = (
                self._config.supersession_confidence_min
                if is_supersession
                else self._config.resolution_confidence_min
            )

            if (
                ext_conf < self._config.extractor_confidence_min
                or res_conf < threshold
            ):
                reason = (
                    "low confidence supersession"
                    if is_supersession
                    else "low extractor or resolution confidence"
                )
                self._quarantine(
                    ext_fact=ext_fact,
                    scope_id=scope_id,
                    episode_id=episode.id,
                    extractor_conf=ext_conf,
                    resolution_conf=res_conf,
                    reason=reason,
                    supersedes_id=supersedes_id,
                )
                result.quarantined_count += 1
                _log.info(
                    "Quarantined (%s): %s.%s=%.2f (threshold=%.2f)",
                    reason, ext_fact.subject, ext_fact.predicate, ext_conf, threshold,
                )
                continue
            if is_supersession:
                self._store.supersede_fact(supersedes_id, now)

            novelty = 0.0 if existing else 0.5  # simple novelty heuristic
            sal = _salience(ext_fact, novelty)

            fact = Fact(
                id=str(uuid4()),
                scope_id=scope_id,
                subject_entity_id=subject_id,
                predicate=ext_fact.predicate,
                object_entity_id=entity_map.get(ext_fact.object),
                object_value_json={"value": ext_fact.object}
                if ext_fact.object not in entity_map
                else None,
                fact_type=ext_fact.fact_type,
                valid_from=now,
                valid_to=None,
                truth_state="current",
                source_episode_id=episode.id,
                confidence=ext_conf,
                salience=sal,
                strength=1.0,
                access_count=0,
                last_accessed_at=None,
                created_at=now,
                updated_at=now,
            )
            self._store.insert_fact(fact)
            result.fact_ids.append(fact.id)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_entity(
        self, name: str, entity_type: str, scope_id: str, episode: Episode
    ) -> str:
        normalized = _normalize(name)

        # 1. Exact alias match
        entity = self._store.find_entity_by_alias(normalized, scope_id)
        if entity:
            self._store.update_entity_seen(entity.id, episode.created_at)
            return entity.id

        # 2. Exact canonical name match
        entity = self._store.find_entity_by_name(scope_id, name)
        if entity:
            self._store.update_entity_seen(entity.id, episode.created_at)
            # Register new alias if not already known
            self._store.insert_alias(Alias(
                id=str(uuid4()),
                entity_id=entity.id,
                value=name,
                normalized_value=normalized,
                embedding_id=None,
                source_episode_id=episode.id,
                confidence=0.95,
                created_at=episode.created_at,
            ))
            return entity.id

        # 3. Create new entity
        now = episode.created_at
        entity = Entity(
            id=str(uuid4()),
            scope_id=scope_id,
            type=entity_type,
            canonical_name=name,
            summary=None,
            created_at=now,
            updated_at=now,
            first_seen_at=now,
            last_seen_at=now,
            salience=0.5,
            strength=1.0,
            confidence=1.0,
        )
        self._store.insert_entity(entity)
        self._store.insert_alias(Alias(
            id=str(uuid4()),
            entity_id=entity.id,
            value=name,
            normalized_value=normalized,
            embedding_id=None,
            source_episode_id=episode.id,
            confidence=1.0,
            created_at=now,
        ))
        return entity.id

    def _quarantine(
        self,
        ext_fact: ExtractedFact,
        scope_id: str,
        episode_id: str,
        extractor_conf: float,
        resolution_conf: float,
        reason: str,
        supersedes_id: str | None = None,
    ) -> None:
        qf = QuarantinedFact(
            id=str(uuid4()),
            scope_id=scope_id,
            source_episode_id=episode_id,
            extracted_subject=ext_fact.subject,
            extracted_predicate=ext_fact.predicate,
            extracted_object=ext_fact.object,
            candidate_supersedes_fact_id=supersedes_id,
            extractor_confidence=extractor_conf,
            resolution_confidence=resolution_conf,
            reason=reason,
            status="pending",
            created_at=datetime.utcnow(),
            reviewed_at=None,
        )
        self._store.insert_quarantined_fact(qf)
