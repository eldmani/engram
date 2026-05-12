from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from .config import EngramConfig
from .embedding.base import BaseEmbedder, NullEmbedder
from .extraction.base import BaseExtractor, NullExtractor
from .ingestion.pipeline import IngestionPipeline
from .models import Fact
from .results import (
    ContextResult,
    ExplainResult,
    ForgetResult,
    RetrievalResult,
    StoreResult,
)
from .retrieval.pipeline import RetrievalPipeline
from .storage.sqlite_store import SQLiteStore
from .workers.background import BackgroundWorkers

_log = logging.getLogger(__name__)


class Engram:
    """
    Model-agnostic temporal memory store.

    Usage::

        mem = Engram()
        mem.store("Alice prefers dark mode.")
        result = mem.get_context("what does Alice prefer?")
        print(result.formatted)

    Plug in a real extractor to activate entity/fact extraction::

        mem.set_extractor(MyLLMExtractor())
    """

    def __init__(self, config: EngramConfig | None = None) -> None:
        self.config = config or EngramConfig()
        self._store = SQLiteStore(self.config.db_path)
        self._extractor: BaseExtractor = NullExtractor()
        self._embedder: BaseEmbedder = NullEmbedder()
        self._ingest = IngestionPipeline(self._store, self._extractor, self.config)
        self._retrieve = RetrievalPipeline(self._store, self.config)
        self._workers = BackgroundWorkers(self._store, self.config)

    def set_extractor(self, extractor: BaseExtractor) -> None:
        """Replace the extractor. Takes effect on the next store() call."""
        self._extractor = extractor
        self._ingest = IngestionPipeline(self._store, extractor, self.config, self._embedder)

    def set_embedder(self, embedder: BaseEmbedder) -> None:
        """Replace the embedder. Takes effect on the next store() / retrieve() call.

        Existing stored episodes and facts are *not* retroactively embedded.
        Call ``store()`` again (or re-ingest) to populate embeddings for old
        data after swapping in a new embedder.

        Example::

            from engram import Engram, BaseEmbedder

            class MyEmbedder(BaseEmbedder):
                model_name = "my-model"
                def embed(self, texts): ...

            mem = Engram()
            mem.set_embedder(MyEmbedder())
        """
        self._embedder = embedder
        self._ingest = IngestionPipeline(self._store, self._extractor, self.config, embedder)
        self._retrieve = RetrievalPipeline(self._store, self.config, embedder)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # --- Memory blocks (always-retrieved in-context key-value slots) ---

    def set_block(self, key: str, value: str, *, scope: str | None = None) -> None:
        """Upsert a named memory block that is always surfaced by get_context().

        Memory blocks are permanently pinned to the context window and are
        returned in every :meth:`get_context` call for the given scope,
        regardless of the retrieval query.  They are intended for agent-editable
        state such as ``"persona"``, ``"current_goal"``, or ``"user_profile"``.
        """
        self._store.set_block(scope or self.config.default_scope, key, value)

    def get_block(self, key: str, *, scope: str | None = None) -> str | None:
        """Return the value of a named memory block, or ``None`` if absent."""
        return self._store.get_block(scope or self.config.default_scope, key)

    def delete_block(self, key: str, *, scope: str | None = None) -> None:
        """Delete a named memory block.  No-op if it does not exist."""
        self._store.delete_block(scope or self.config.default_scope, key)

    def get_blocks(self, *, scope: str | None = None) -> dict[str, str]:
        """Return all memory blocks for the given scope as ``{key: value}``."""
        return self._store.get_all_blocks(scope or self.config.default_scope)

    def store(
        self,
        data: str | dict,
        metadata: dict | None = None,
        scope: str | None = None,
    ) -> StoreResult:
        """
        Store a piece of information.

        Creates an immutable episode. If an extractor is configured, also
        extracts entities and temporal facts. Returns IDs of all created objects.
        """
        scope_id = scope or self.config.default_scope
        return self._ingest.run(data, metadata, scope_id)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        scope: str | None = None,
        as_of: datetime | None = None,
    ) -> RetrievalResult:
        """
        Retrieve structured results for a query.

        Returns facts, entities, and raw episodes — NOT assembled prompt text.
        Use get_context() for model-ready output.
        """
        return self._retrieve.retrieve(
            query=query,
            top_k=top_k or self.config.default_top_k,
            scope_id=scope or self.config.default_scope,
            as_of=as_of,
        )

    def get_context(
        self,
        query: str,
        top_k: int | None = None,
        scope: str | None = None,
        as_of: datetime | None = None,
        template: str = "default",
    ) -> ContextResult:
        """
        Retrieve and assemble model-ready context for a query.

        Returns structured blocks (facts, entities, timeline, episodes,
        evidence) plus a formatted string ready to insert into a prompt.
        """
        return self._retrieve.get_context(
            query=query,
            top_k=top_k or self.config.default_top_k,
            scope_id=scope or self.config.default_scope,
            as_of=as_of,
            template=template,
        )

    def update_fact(
        self,
        fact_id: str,
        value: dict | str,
        valid_from: datetime | None = None,
    ) -> Fact:
        """
        Update a fact's value without mutating history.

        Closes the existing fact (sets valid_to = now, truth_state = superseded)
        and creates a new current fact with the new value.
        """
        existing = self._store.get_fact(fact_id)
        if existing is None:
            raise ValueError(f"Fact {fact_id!r} not found")

        now = valid_from or datetime.now(timezone.utc)
        self._store.supersede_fact(fact_id, now)

        new_fact = Fact(
            id=str(uuid4()),
            scope_id=existing.scope_id,
            subject_entity_id=existing.subject_entity_id,
            predicate=existing.predicate,
            object_entity_id=existing.object_entity_id,
            object_value_json={"value": value} if isinstance(value, str) else value,
            fact_type=existing.fact_type,
            valid_from=now,
            valid_to=None,
            truth_state="current",
            source_episode_id=existing.source_episode_id,
            confidence=1.0,
            salience=existing.salience,
            strength=existing.strength,
            access_count=0,
            last_accessed_at=None,
            created_at=now,
            updated_at=now,
        )
        self._store.insert_fact(new_fact)
        return new_fact

    def forget(self, selector: dict, hard: bool = False) -> ForgetResult:
        """
        Remove or hide facts/entities/episodes.

        selector keys:
            fact_id      – target a single fact
            entity_id    – target all facts for an entity
            episode_id   – target all facts derived from an episode
            scope_id     – target everything in a scope

        hard=False marks as hidden (soft forget, stays in history).
        hard=True physically deletes rows.
        """
        result = ForgetResult()

        if "fact_id" in selector:
            fid = selector["fact_id"]
            if hard:
                self._store.delete_fact(fid)
            else:
                self._store.hide_fact(fid)
            result.affected_facts = 1

        elif "entity_id" in selector:
            eid = selector["entity_id"]
            if hard:
                self._store.delete_entity_facts(eid)
            else:
                self._store.hide_entity_facts(eid)
            result.affected_entities = 1

        elif "episode_id" in selector:
            if hard:
                self._store.delete_episode_facts(selector["episode_id"])
            result.affected_facts = 1  # approximate

        elif "scope_id" in selector:
            sid = selector["scope_id"]
            if hard:
                self._store.delete_scope_facts(sid)
            else:
                self._store.hide_scope_facts(sid)
            result.affected_facts = 1  # approximate
        else:
            raise ValueError(
                "selector must contain one of: fact_id, entity_id, episode_id, scope_id"
            )

        return result

    def explain(
        self,
        query: str,
        result_id: str | None = None,
        scope: str | None = None,
    ) -> ExplainResult:
        """
        Explain why the retrieval returned what it did.

        Returns retrieval traces and any pending quarantined facts for the scope.
        """
        scope_id = scope or self.config.default_scope
        traces = self._store.get_traces(scope_id, query)
        quarantined = self._store.get_pending_quarantined_facts(scope_id)

        lines = [f"Query: {query!r}", f"Scope: {scope_id}"]
        if traces:
            lines.append(f"Retrieval traces: {len(traces)}")
            for t in traces[:3]:
                lines.append(
                    f"  fact={t.candidate_fact_id} "
                    f"kw={t.keyword_score:.3f} "
                    f"temporal={t.temporal_score:.3f} "
                    f"final={t.final_score:.3f}"
                )
        if quarantined:
            lines.append(
                f"Quarantined facts pending review: {len(quarantined)}"
            )
            for q in quarantined[:3]:
                lines.append(
                    f"  [{q.reason}] "
                    f"{q.extracted_subject} {q.extracted_predicate} {q.extracted_object} "
                    f"(ext_conf={q.extractor_confidence:.2f})"
                )

        return ExplainResult(
            traces=traces,
            quarantined_facts=quarantined,
            summary="\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Worker access
    # ------------------------------------------------------------------

    @property
    def workers(self) -> BackgroundWorkers:
        """Access background workers for manual scheduling."""
        return self._workers

    def close(self) -> None:
        """Close the underlying database connection."""
        self._store.close()

    def __enter__(self) -> "Engram":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
