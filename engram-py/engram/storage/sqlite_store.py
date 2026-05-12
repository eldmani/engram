from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from ..models import Alias, Entity, Episode, Fact, QuarantinedFact, RetrievalTrace

_log = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    scope_id    TEXT NOT NULL,
    source      TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    checksum    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS episode_scope_time_idx ON episodes (scope_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes USING fts5(
    episode_id,
    raw_text,
    scope_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    scope_id        TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'unknown',
    canonical_name  TEXT NOT NULL,
    summary         TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    salience        REAL NOT NULL DEFAULT 0.5,
    strength        REAL NOT NULL DEFAULT 1.0,
    confidence      REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS entity_scope_idx  ON entities (scope_id);
CREATE INDEX IF NOT EXISTS entity_name_idx   ON entities (scope_id, canonical_name);

CREATE TABLE IF NOT EXISTS aliases (
    id                  TEXT PRIMARY KEY,
    entity_id           TEXT NOT NULL REFERENCES entities(id),
    value               TEXT NOT NULL,
    normalized_value    TEXT NOT NULL,
    embedding_id        TEXT,
    source_episode_id   TEXT NOT NULL REFERENCES episodes(id),
    confidence          REAL NOT NULL DEFAULT 1.0,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alias_norm_idx   ON aliases (normalized_value);
CREATE INDEX IF NOT EXISTS alias_entity_idx ON aliases (entity_id);

CREATE TABLE IF NOT EXISTS facts (
    id                  TEXT PRIMARY KEY,
    scope_id            TEXT NOT NULL,
    subject_entity_id   TEXT NOT NULL REFERENCES entities(id),
    predicate           TEXT NOT NULL,
    object_entity_id    TEXT,
    object_value_json   TEXT,
    fact_type           TEXT NOT NULL DEFAULT 'assertion',
    valid_from          TEXT NOT NULL,
    valid_to            TEXT,
    truth_state         TEXT NOT NULL DEFAULT 'current',
    source_episode_id   TEXT NOT NULL REFERENCES episodes(id),
    confidence          REAL NOT NULL DEFAULT 1.0,
    salience            REAL NOT NULL DEFAULT 0.5,
    strength            REAL NOT NULL DEFAULT 1.0,
    access_count        INTEGER NOT NULL DEFAULT 0,
    last_accessed_at    TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fact_subject_pred_idx ON facts (subject_entity_id, predicate);
CREATE INDEX IF NOT EXISTS fact_validity_idx     ON facts (valid_from, valid_to);
CREATE INDEX IF NOT EXISTS fact_truth_state_idx  ON facts (truth_state);
CREATE INDEX IF NOT EXISTS fact_scope_idx        ON facts (scope_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(
    fact_id,
    predicate,
    object_text,
    scope_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS quarantined_facts (
    id                              TEXT PRIMARY KEY,
    scope_id                        TEXT NOT NULL,
    source_episode_id               TEXT NOT NULL REFERENCES episodes(id),
    extracted_subject               TEXT NOT NULL,
    extracted_predicate             TEXT NOT NULL,
    extracted_object                TEXT NOT NULL,
    candidate_supersedes_fact_id    TEXT,
    extractor_confidence            REAL NOT NULL,
    resolution_confidence           REAL NOT NULL,
    reason                          TEXT NOT NULL,
    status                          TEXT NOT NULL DEFAULT 'pending',
    created_at                      TEXT NOT NULL,
    reviewed_at                     TEXT
);
CREATE INDEX IF NOT EXISTS quarantine_scope_status_idx ON quarantined_facts (scope_id, status);

CREATE TABLE IF NOT EXISTS retrieval_traces (
    id                  TEXT PRIMARY KEY,
    query               TEXT NOT NULL,
    scope_id            TEXT NOT NULL,
    candidate_fact_id   TEXT,
    semantic_score      REAL NOT NULL DEFAULT 0.0,
    keyword_score       REAL NOT NULL DEFAULT 0.0,
    graph_score         REAL NOT NULL DEFAULT 0.0,
    temporal_score      REAL NOT NULL DEFAULT 0.0,
    final_score         REAL NOT NULL DEFAULT 0.0,
    matched_entities    TEXT NOT NULL DEFAULT '[]',
    source_episode_ids  TEXT NOT NULL DEFAULT '[]',
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trace_scope_idx ON retrieval_traces (scope_id, created_at);

CREATE TABLE IF NOT EXISTS embeddings (
    id          TEXT PRIMARY KEY,
    ref_type    TEXT NOT NULL,
    ref_id      TEXT NOT NULL,
    vector      BLOB NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS embedding_ref_idx ON embeddings (ref_type, ref_id);

CREATE TABLE IF NOT EXISTS memory_blocks (
    scope_id    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (scope_id, key)
);
CREATE INDEX IF NOT EXISTS block_scope_idx ON memory_blocks (scope_id);
"""

_DT_FMT = "%Y-%m-%dT%H:%M:%S.%f"
_SCHEMA_VERSION = 2


def _sanitize_fts(query: str) -> str:
    """Remove FTS5 special characters to prevent OperationalError on malformed queries."""
    sanitized = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE).strip()
    return sanitized if sanitized else '""'


def _dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s, _DT_FMT)


def _ts(d: Optional[datetime]) -> Optional[str]:
    return d.isoformat() if d else None


class SQLiteStore:
    """
    SQLite-backed persistent store for engram.

    Thread safety: each thread gets its own SQLite connection via
    ``threading.local()``. SQLite WAL mode serializes concurrent writes at
    the file level. Call ``close()`` from every thread that used the store.
    """

    def __init__(self, db_path: str = "engram.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        # Bootstrap schema on the calling thread's connection.
        conn = self._conn
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()
        _log.debug("SQLiteStore opened: %s (schema v%d)", db_path, _SCHEMA_VERSION)

    @property
    def _conn(self) -> sqlite3.Connection:
        """Per-thread SQLite connection, created on first access."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        """Close the current thread's SQLite connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def insert_episode(self, ep: Episode) -> None:
        self._conn.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?)",
            (ep.id, ep.scope_id, ep.source, ep.raw_text,
             json.dumps(ep.metadata), _ts(ep.created_at), ep.checksum),
        )
        self._conn.execute(
            "INSERT INTO fts_episodes VALUES (?,?,?)",
            (ep.id, ep.raw_text, ep.scope_id),
        )
        self._conn.commit()

    def get_episode(self, ep_id: str) -> Optional[Episode]:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id=?", (ep_id,)
        ).fetchone()
        return self._row_to_episode(row) if row else None

    def fts_search_episodes(
        self, query: str, scope_id: str, limit: int
    ) -> list[tuple[Episode, float]]:
        safe_query = _sanitize_fts(query)
        if safe_query != query:
            _log.debug("FTS episode query sanitized: %r -> %r", query, safe_query)
        try:
            rows = self._conn.execute(
                """
                SELECT e.*, -fe.rank AS kw_score
                FROM fts_episodes fe
                JOIN episodes e ON e.id = fe.episode_id
                WHERE fts_episodes MATCH ? AND fe.scope_id = ?
                ORDER BY fe.rank
                LIMIT ?
                """,
                (safe_query, scope_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            _log.warning("FTS episode search failed for query %r (sanitized: %r)", query, safe_query)
            return []
        return [(self._row_to_episode(r), float(r["kw_score"])) for r in rows]

    def get_recent_episodes(self, scope_id: str, limit: int) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE scope_id=? ORDER BY created_at DESC LIMIT ?",
            (scope_id, limit),
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            scope_id=row["scope_id"],
            source=row["source"],
            raw_text=row["raw_text"],
            metadata=json.loads(row["metadata"]),
            created_at=_dt(row["created_at"]),
            checksum=row["checksum"],
        )

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def insert_entity(self, entity: Entity) -> None:
        self._conn.execute(
            """INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (entity.id, entity.scope_id, entity.type, entity.canonical_name,
             entity.summary, _ts(entity.created_at), _ts(entity.updated_at),
             _ts(entity.first_seen_at), _ts(entity.last_seen_at),
             entity.salience, entity.strength, entity.confidence),
        )
        self._conn.commit()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id=?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entity_by_name(self, scope_id: str, name: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE scope_id=? AND canonical_name=? COLLATE NOCASE",
            (scope_id, name),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entity_by_alias(
        self, normalized_value: str, scope_id: str
    ) -> Optional[Entity]:
        row = self._conn.execute(
            """
            SELECT e.* FROM aliases a
            JOIN entities e ON e.id = a.entity_id
            WHERE a.normalized_value=? AND e.scope_id=?
            LIMIT 1
            """,
            (normalized_value, scope_id),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def update_entity_seen(self, entity_id: str, when: datetime) -> None:
        self._conn.execute(
            "UPDATE entities SET last_seen_at=?, updated_at=? WHERE id=?",
            (_ts(when), _ts(when), entity_id),
        )
        self._conn.commit()

    def update_entity_summary(self, entity_id: str, summary: str) -> None:
        self._conn.execute(
            "UPDATE entities SET summary=?, updated_at=? WHERE id=?",
            (summary, _ts(datetime.now(timezone.utc)), entity_id),
        )
        self._conn.commit()

    def get_entities_for_scope(self, scope_id: str) -> list[Entity]:
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE scope_id=? ORDER BY salience DESC",
            (scope_id,),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            scope_id=row["scope_id"],
            type=row["type"],
            canonical_name=row["canonical_name"],
            summary=row["summary"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            first_seen_at=_dt(row["first_seen_at"]),
            last_seen_at=_dt(row["last_seen_at"]),
            salience=row["salience"],
            strength=row["strength"],
            confidence=row["confidence"],
        )

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------

    def insert_alias(self, alias: Alias) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?,?,?,?)",
            (alias.id, alias.entity_id, alias.value, alias.normalized_value,
             alias.embedding_id, alias.source_episode_id,
             alias.confidence, _ts(alias.created_at)),
        )
        self._conn.commit()

    def get_aliases_for_entity(self, entity_id: str) -> list[Alias]:
        rows = self._conn.execute(
            "SELECT * FROM aliases WHERE entity_id=?", (entity_id,)
        ).fetchall()
        return [self._row_to_alias(r) for r in rows]

    def _row_to_alias(self, row: sqlite3.Row) -> Alias:
        return Alias(
            id=row["id"],
            entity_id=row["entity_id"],
            value=row["value"],
            normalized_value=row["normalized_value"],
            embedding_id=row["embedding_id"],
            source_episode_id=row["source_episode_id"],
            confidence=row["confidence"],
            created_at=_dt(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def insert_fact(self, fact: Fact) -> None:
        obj_text = ""
        if fact.object_value_json:
            obj_text = json.dumps(fact.object_value_json)
        elif fact.object_entity_id:
            obj_text = fact.object_entity_id

        self._conn.execute(
            """INSERT INTO facts VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fact.id, fact.scope_id, fact.subject_entity_id, fact.predicate,
             fact.object_entity_id, json.dumps(fact.object_value_json) if fact.object_value_json else None,
             fact.fact_type, _ts(fact.valid_from), _ts(fact.valid_to),
             fact.truth_state, fact.source_episode_id,
             fact.confidence, fact.salience, fact.strength,
             fact.access_count, _ts(fact.last_accessed_at),
             _ts(fact.created_at), _ts(fact.updated_at)),
        )
        self._conn.execute(
            "INSERT INTO fts_facts VALUES (?,?,?,?)",
            (fact.id, fact.predicate, obj_text, fact.scope_id),
        )
        self._conn.commit()

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id=?", (fact_id,)
        ).fetchone()
        return self._row_to_fact(row) if row else None

    def get_current_facts(
        self, scope_id: str, subject_entity_id: str, predicate: str
    ) -> list[Fact]:
        rows = self._conn.execute(
            """SELECT * FROM facts
               WHERE scope_id=? AND subject_entity_id=? AND predicate=?
               AND truth_state='current' AND valid_to IS NULL""",
            (scope_id, subject_entity_id, predicate),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def supersede_fact(self, fact_id: str, valid_to: datetime) -> None:
        now = _ts(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE facts SET valid_to=?, truth_state='superseded', updated_at=? WHERE id=?",
            (_ts(valid_to), now, fact_id),
        )
        self._conn.commit()

    def hide_fact(self, fact_id: str) -> None:
        now = _ts(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE facts SET truth_state='hidden', updated_at=? WHERE id=?",
            (now, fact_id),
        )
        self._conn.commit()

    def delete_fact(self, fact_id: str) -> None:
        self._conn.execute("DELETE FROM fts_facts WHERE fact_id=?", (fact_id,))
        self._conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
        self._conn.commit()

    def fts_search_facts(
        self,
        query: str,
        scope_id: str,
        limit: int,
        as_of: Optional[datetime] = None,
    ) -> list[tuple[Fact, float]]:
        safe_query = _sanitize_fts(query)
        if safe_query != query:
            _log.debug("FTS facts query sanitized: %r -> %r", query, safe_query)
        try:
            rows = self._conn.execute(
                """
                SELECT f.*, -ff.rank AS kw_score
                FROM fts_facts ff
                JOIN facts f ON f.id = ff.fact_id
                WHERE fts_facts MATCH ? AND ff.scope_id = ?
                  AND f.truth_state != 'hidden'
                ORDER BY ff.rank
                LIMIT ?
                """,
                (safe_query, scope_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            _log.warning("FTS facts search failed for query %r (sanitized: %r)", query, safe_query)
            return []

        results = []
        for r in rows:
            fact = self._row_to_fact(r)
            if as_of and fact.valid_from and fact.valid_from > as_of:
                continue
            if as_of and fact.valid_to and fact.valid_to <= as_of:
                continue
            results.append((fact, float(r["kw_score"])))
        return results

    def get_current_facts_for_scope(
        self, scope_id: str, limit: int
    ) -> list[Fact]:
        rows = self._conn.execute(
            """SELECT * FROM facts
               WHERE scope_id=? AND truth_state='current'
               ORDER BY salience DESC, strength DESC
               LIMIT ?""",
            (scope_id, limit),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def reinforce_fact(self, fact_id: str, boost: float) -> None:
        now = _ts(datetime.now(timezone.utc))
        self._conn.execute(
            """UPDATE facts
               SET access_count = access_count + 1,
                   last_accessed_at = ?,
                   strength = MIN(strength + ?, 5.0),
                   updated_at = ?
               WHERE id=?""",
            (now, boost, now, fact_id),
        )
        self._conn.commit()

    def apply_decay(self, scope_id: str, stable_threshold: int) -> None:
        """Apply power-law decay to weak facts and linear decay to stable ones."""
        now = datetime.now(timezone.utc)
        rows = self._conn.execute(
            "SELECT id, strength, access_count, updated_at FROM facts WHERE scope_id=?",
            (scope_id,),
        ).fetchall()

        updates: list[tuple[float, str, str]] = []
        for r in rows:
            age_hours = max(
                1.0,
                (now - _dt(r["updated_at"])).total_seconds() / 3600,
            )
            if r["access_count"] < stable_threshold:
                # Power-law decay: s(t) = s / t^0.5
                new_strength = max(0.01, r["strength"] / math.sqrt(age_hours / 24 + 1))
            else:
                # Slow linear decay
                new_strength = max(0.01, r["strength"] - 0.001 * age_hours)
            updates.append((new_strength, _ts(now), r["id"]))

        self._conn.executemany(
            "UPDATE facts SET strength=?, updated_at=? WHERE id=?", updates
        )
        self._conn.commit()

    def delete_episode_facts(self, episode_id: str) -> None:
        rows = self._conn.execute(
            "SELECT id FROM facts WHERE source_episode_id=?", (episode_id,)
        ).fetchall()
        for r in rows:
            self.delete_fact(r["id"])

    def hide_entity_facts(self, entity_id: str) -> None:
        now = _ts(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE facts SET truth_state='hidden', updated_at=? WHERE subject_entity_id=?",
            (now, entity_id),
        )
        self._conn.commit()

    def delete_entity_facts(self, entity_id: str) -> None:
        rows = self._conn.execute(
            "SELECT id FROM facts WHERE subject_entity_id=?", (entity_id,)
        ).fetchall()
        for r in rows:
            self.delete_fact(r["id"])

    def hide_scope_facts(self, scope_id: str) -> None:
        now = _ts(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE facts SET truth_state='hidden', updated_at=? WHERE scope_id=?",
            (now, scope_id),
        )
        self._conn.commit()

    def delete_scope_facts(self, scope_id: str) -> None:
        rows = self._conn.execute(
            "SELECT id FROM facts WHERE scope_id=?", (scope_id,)
        ).fetchall()
        for r in rows:
            self.delete_fact(r["id"])

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        obj_val = None
        if row["object_value_json"]:
            try:
                obj_val = json.loads(row["object_value_json"])
            except (json.JSONDecodeError, TypeError):
                obj_val = {"value": row["object_value_json"]}
        return Fact(
            id=row["id"],
            scope_id=row["scope_id"],
            subject_entity_id=row["subject_entity_id"],
            predicate=row["predicate"],
            object_entity_id=row["object_entity_id"],
            object_value_json=obj_val,
            fact_type=row["fact_type"],
            valid_from=_dt(row["valid_from"]),
            valid_to=_dt(row["valid_to"]),
            truth_state=row["truth_state"],
            source_episode_id=row["source_episode_id"],
            confidence=row["confidence"],
            salience=row["salience"],
            strength=row["strength"],
            access_count=row["access_count"],
            last_accessed_at=_dt(row["last_accessed_at"]),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    # ------------------------------------------------------------------
    # Quarantined Facts
    # ------------------------------------------------------------------

    def insert_quarantined_fact(self, qf: QuarantinedFact) -> None:
        self._conn.execute(
            "INSERT INTO quarantined_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (qf.id, qf.scope_id, qf.source_episode_id,
             qf.extracted_subject, qf.extracted_predicate, qf.extracted_object,
             qf.candidate_supersedes_fact_id,
             qf.extractor_confidence, qf.resolution_confidence,
             qf.reason, qf.status, _ts(qf.created_at), _ts(qf.reviewed_at)),
        )
        self._conn.commit()

    def get_pending_quarantined_facts(self, scope_id: str) -> list[QuarantinedFact]:
        rows = self._conn.execute(
            "SELECT * FROM quarantined_facts WHERE scope_id=? AND status='pending'",
            (scope_id,),
        ).fetchall()
        return [self._row_to_qf(r) for r in rows]

    def update_quarantine_status(
        self, qf_id: str, status: str
    ) -> None:
        now = _ts(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE quarantined_facts SET status=?, reviewed_at=? WHERE id=?",
            (status, now, qf_id),
        )
        self._conn.commit()

    def _row_to_qf(self, row: sqlite3.Row) -> QuarantinedFact:
        return QuarantinedFact(
            id=row["id"],
            scope_id=row["scope_id"],
            source_episode_id=row["source_episode_id"],
            extracted_subject=row["extracted_subject"],
            extracted_predicate=row["extracted_predicate"],
            extracted_object=row["extracted_object"],
            candidate_supersedes_fact_id=row["candidate_supersedes_fact_id"],
            extractor_confidence=row["extractor_confidence"],
            resolution_confidence=row["resolution_confidence"],
            reason=row["reason"],
            status=row["status"],
            created_at=_dt(row["created_at"]),
            reviewed_at=_dt(row["reviewed_at"]),
        )

    # ------------------------------------------------------------------
    # Retrieval Traces
    # ------------------------------------------------------------------

    def insert_retrieval_trace(self, trace: RetrievalTrace) -> None:
        self._conn.execute(
            """INSERT INTO retrieval_traces VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (trace.id, trace.query, trace.scope_id, trace.candidate_fact_id,
             trace.semantic_score, trace.keyword_score,
             trace.graph_score, trace.temporal_score, trace.final_score,
             json.dumps(trace.matched_entities),
             json.dumps(trace.source_episode_ids),
             _ts(trace.created_at)),
        )
        self._conn.commit()

    def get_traces(self, scope_id: str, query: str) -> list[RetrievalTrace]:
        rows = self._conn.execute(
            """SELECT * FROM retrieval_traces
               WHERE scope_id=? AND query=?
               ORDER BY created_at DESC LIMIT 50""",
            (scope_id, query),
        ).fetchall()
        return [self._row_to_trace(r) for r in rows]

    def _row_to_trace(self, row: sqlite3.Row) -> RetrievalTrace:
        return RetrievalTrace(
            id=row["id"],
            query=row["query"],
            scope_id=row["scope_id"],
            candidate_fact_id=row["candidate_fact_id"],
            semantic_score=row["semantic_score"],
            keyword_score=row["keyword_score"],
            graph_score=row["graph_score"],
            temporal_score=row["temporal_score"],
            final_score=row["final_score"],
            matched_entities=json.loads(row["matched_entities"]),
            source_episode_ids=json.loads(row["source_episode_ids"]),
            created_at=_dt(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def insert_embedding(
        self,
        ref_type: str,
        ref_id: str,
        vector: list[float],
        model: str,
    ) -> None:
        """Store (or replace) a float vector for an episode or fact.

        The primary key is ``"{ref_type}:{ref_id}"`` so each reference holds
        exactly one embedding.  Calling this again with a different model
        simply overwrites the previous vector.
        """
        from ..embedding.base import vec_to_blob

        emb_id = f"{ref_type}:{ref_id}"
        blob = vec_to_blob(vector)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
            (emb_id, ref_type, ref_id, blob, model,
             _ts(datetime.now(timezone.utc))),
        )
        self._conn.commit()

    def get_embeddings_for_refs(
        self, ref_type: str, ref_ids: list[str]
    ) -> list[tuple[str, list[float]]]:
        """Return ``(ref_id, vector)`` pairs for the requested references."""
        if not ref_ids:
            return []
        from ..embedding.base import blob_to_vec

        placeholders = ",".join("?" * len(ref_ids))
        rows = self._conn.execute(
            f"SELECT ref_id, vector FROM embeddings"
            f" WHERE ref_type=? AND ref_id IN ({placeholders})",
            (ref_type, *ref_ids),
        ).fetchall()
        return [(r["ref_id"], blob_to_vec(r["vector"])) for r in rows]

    def get_embedded_fact_ids_for_scope(self, scope_id: str) -> list[str]:
        """Return IDs of *current* facts in *scope_id* that have stored embeddings."""
        rows = self._conn.execute(
            """
            SELECT e.ref_id
            FROM embeddings e
            JOIN facts f ON f.id = e.ref_id
            WHERE e.ref_type = 'fact'
              AND f.scope_id = ?
              AND f.truth_state != 'hidden'
            """,
            (scope_id,),
        ).fetchall()
        return [r["ref_id"] for r in rows]

    # ------------------------------------------------------------------
    # Graph traversal helpers
    # ------------------------------------------------------------------

    def get_connected_entity_ids(
        self, entity_ids: list[str], scope_id: str
    ) -> set[str]:
        """Return entity IDs reachable in exactly one hop from *entity_ids*.

        Traverses both outgoing (subject→object_entity) and incoming
        (object_entity→subject) edges of *current* facts in *scope_id*.
        The seed entities themselves are excluded from the result.
        """
        if not entity_ids:
            return set()
        ph = ",".join("?" * len(entity_ids))
        # Outgoing: seeds are subjects → collect objects
        rows_out = self._conn.execute(
            f"""SELECT DISTINCT object_entity_id FROM facts
                WHERE subject_entity_id IN ({ph})
                  AND object_entity_id IS NOT NULL
                  AND scope_id = ?
                  AND truth_state != 'hidden'""",
            (*entity_ids, scope_id),
        ).fetchall()
        # Incoming: seeds are objects → collect subjects
        rows_in = self._conn.execute(
            f"""SELECT DISTINCT subject_entity_id FROM facts
                WHERE object_entity_id IN ({ph})
                  AND scope_id = ?
                  AND truth_state != 'hidden'""",
            (*entity_ids, scope_id),
        ).fetchall()
        seed_set = set(entity_ids)
        neighbors = (
            {r["object_entity_id"] for r in rows_out}
            | {r["subject_entity_id"] for r in rows_in}
        )
        return neighbors - seed_set

    def get_facts_for_entities(
        self, entity_ids: list[str], scope_id: str, limit: int = 100
    ) -> list[Fact]:
        """Return current facts whose subject *or* object is in *entity_ids*."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        rows = self._conn.execute(
            f"""SELECT * FROM facts
                WHERE scope_id = ?
                  AND truth_state != 'hidden'
                  AND (
                      subject_entity_id IN ({ph})
                      OR (object_entity_id IS NOT NULL AND object_entity_id IN ({ph}))
                  )
                ORDER BY salience DESC, strength DESC
                LIMIT ?""",
            (scope_id, *entity_ids, *entity_ids, limit),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_entity_graph(
        self,
        seed_entity_ids: list[str],
        scope_id: str,
        max_hops: int = 2,
    ) -> dict[str, int]:
        """BFS from *seed_entity_ids* up to *max_hops* hops.

        Returns a mapping ``{entity_id: hop_distance}`` where seeds have
        distance 0 and each hop adds 1.  Only entities reachable via
        *current* (non-hidden) facts in *scope_id* are included.
        """
        if not seed_entity_ids:
            return {}
        distances: dict[str, int] = {eid: 0 for eid in seed_entity_ids}
        frontier = list(seed_entity_ids)
        for hop in range(1, max_hops + 1):
            if not frontier:
                break
            neighbors = self.get_connected_entity_ids(frontier, scope_id)
            new_nodes = neighbors - set(distances)
            for nid in new_nodes:
                distances[nid] = hop
            frontier = list(new_nodes)
        return distances

    # ------------------------------------------------------------------
    # Memory blocks (always-retrieved in-context key-value slots)
    # ------------------------------------------------------------------

    def set_block(self, scope_id: str, key: str, value: str) -> None:
        """Upsert a named memory block for *scope_id*."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO memory_blocks (scope_id, key, value, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(scope_id, key) DO UPDATE SET
                   value = excluded.value,
                   updated_at = excluded.updated_at""",
            (scope_id, key, value, now, now),
        )
        self._conn.commit()

    def get_block(self, scope_id: str, key: str) -> Optional[str]:
        """Return the value of a memory block, or ``None`` if it does not exist."""
        row = self._conn.execute(
            "SELECT value FROM memory_blocks WHERE scope_id = ? AND key = ?",
            (scope_id, key),
        ).fetchone()
        return row["value"] if row else None

    def get_all_blocks(self, scope_id: str) -> dict[str, str]:
        """Return all memory blocks for *scope_id* as ``{key: value}``."""
        rows = self._conn.execute(
            "SELECT key, value FROM memory_blocks WHERE scope_id = ? ORDER BY key",
            (scope_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_block(self, scope_id: str, key: str) -> None:
        """Delete a named memory block.  No-op if it does not exist."""
        self._conn.execute(
            "DELETE FROM memory_blocks WHERE scope_id = ? AND key = ?",
            (scope_id, key),
        )
        self._conn.commit()
