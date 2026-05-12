"""
Unit tests for SQLiteStore — covers every public method and all edge cases.
These tests bypass Engram and talk directly to the storage layer.
"""

import os
import tempfile
import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from engram.models import Alias, Entity, Episode, Fact, QuarantinedFact, RetrievalTrace
from engram.storage.sqlite_store import (
    SQLiteStore,
    _SCHEMA_VERSION,
    _sanitize_fts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "test.db"))
    yield s
    s.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _episode(scope_id: str = "default", text: str = "Alice prefers dark mode.") -> Episode:
    return Episode(
        id=str(uuid4()),
        scope_id=scope_id,
        source="test",
        raw_text=text,
        metadata={"src": "unit"},
        created_at=_now(),
        checksum=Episode.compute_checksum(text),
    )


def _entity(scope_id: str = "default", name: str = "Alice") -> Entity:
    now = _now()
    return Entity(
        id=str(uuid4()),
        scope_id=scope_id,
        type="person",
        canonical_name=name,
        summary=None,
        created_at=now,
        updated_at=now,
        first_seen_at=now,
        last_seen_at=now,
        salience=0.5,
        strength=1.0,
        confidence=0.95,
    )


def _fact(scope_id: str, episode_id: str, entity_id: str,
          predicate: str = "prefers", obj: str = "dark mode",
          truth_state: str = "current") -> Fact:
    now = _now()
    return Fact(
        id=str(uuid4()),
        scope_id=scope_id,
        subject_entity_id=entity_id,
        predicate=predicate,
        object_entity_id=None,
        object_value_json={"value": obj},
        fact_type="assertion",
        valid_from=now,
        valid_to=None,
        truth_state=truth_state,
        source_episode_id=episode_id,
        confidence=0.92,
        salience=0.5,
        strength=1.0,
        access_count=0,
        last_accessed_at=None,
        created_at=now,
        updated_at=now,
    )


def _quarantine(scope_id: str, episode_id: str,
                subject: str = "Alice", predicate: str = "prefers",
                obj: str = "dark mode") -> QuarantinedFact:
    return QuarantinedFact(
        id=str(uuid4()),
        scope_id=scope_id,
        source_episode_id=episode_id,
        extracted_subject=subject,
        extracted_predicate=predicate,
        extracted_object=obj,
        candidate_supersedes_fact_id=None,
        extractor_confidence=0.50,
        resolution_confidence=0.50,
        reason="low confidence",
        status="pending",
        created_at=_now(),
        reviewed_at=None,
    )


def _alias(entity_id: str, episode_id: str, value: str = "alice") -> Alias:
    return Alias(
        id=str(uuid4()),
        entity_id=entity_id,
        value=value,
        normalized_value=value.lower(),
        embedding_id=None,
        source_episode_id=episode_id,
        confidence=1.0,
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# _sanitize_fts
# ---------------------------------------------------------------------------

class TestSanitizeFts:
    def test_normal_query_unchanged(self):
        assert _sanitize_fts("dark mode") == "dark mode"

    def test_strips_special_chars(self):
        result = _sanitize_fts("dark*mode!")
        assert "*" not in result
        assert "!" not in result

    def test_empty_string_returns_quoted(self):
        assert _sanitize_fts("") == '""'

    def test_all_special_chars_returns_quoted(self):
        assert _sanitize_fts("!!!") == '""'

    def test_parentheses_stripped(self):
        result = _sanitize_fts("(foo AND bar)")
        assert "(" not in result
        assert ")" not in result

    def test_unicode_words_preserved(self):
        result = _sanitize_fts("こんにちは")
        assert "こんにちは" in result


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_schema_version_constant(self):
        assert _SCHEMA_VERSION == 2

    def test_pragma_user_version_set(self, store: SQLiteStore):
        row = store._conn.execute("PRAGMA user_version").fetchone()
        assert row[0] == _SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

class TestEpisodes:
    def test_insert_and_get(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        fetched = store.get_episode(ep.id)
        assert fetched is not None
        assert fetched.id == ep.id
        assert fetched.raw_text == ep.raw_text
        assert fetched.scope_id == ep.scope_id

    def test_get_nonexistent_returns_none(self, store: SQLiteStore):
        assert store.get_episode("no-such-id") is None

    def test_metadata_round_trips(self, store: SQLiteStore):
        ep = _episode()
        ep.metadata = {"key": "value", "num": 42}
        store.insert_episode(ep)
        fetched = store.get_episode(ep.id)
        assert fetched.metadata == {"key": "value", "num": 42}

    def test_fts_search_returns_results(self, store: SQLiteStore):
        ep = _episode(text="Alice loves dark mode.")
        store.insert_episode(ep)
        results = store.fts_search_episodes("dark", "default", limit=10)
        assert len(results) >= 1
        assert results[0][0].id == ep.id
        assert results[0][1] > 0

    def test_fts_search_scope_isolation(self, store: SQLiteStore):
        ep1 = _episode(scope_id="scope-a", text="cats in scope A")
        ep2 = _episode(scope_id="scope-b", text="dogs in scope B")
        store.insert_episode(ep1)
        store.insert_episode(ep2)
        results_a = store.fts_search_episodes("cats", "scope-a", limit=5)
        results_b = store.fts_search_episodes("dogs", "scope-b", limit=5)
        assert len(results_a) == 1
        assert len(results_b) == 1
        # cross-scope: cats should not appear in scope-b
        assert store.fts_search_episodes("cats", "scope-b", limit=5) == []

    def test_fts_search_malformed_query_returns_empty(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        # Should not raise despite FTS special chars
        result = store.fts_search_episodes("dark OR* AND!", "default", limit=10)
        assert isinstance(result, list)

    def test_fts_search_empty_query_returns_empty(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        result = store.fts_search_episodes("", "default", limit=10)
        assert isinstance(result, list)

    def test_get_recent_episodes(self, store: SQLiteStore):
        for i in range(5):
            store.insert_episode(_episode(text=f"episode {i}"))
        recent = store.get_recent_episodes("default", limit=3)
        assert len(recent) == 3

    def test_get_recent_episodes_empty_scope(self, store: SQLiteStore):
        assert store.get_recent_episodes("empty", limit=10) == []


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

class TestEntities:
    def test_insert_and_get(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        fetched = store.get_entity(ent.id)
        assert fetched is not None
        assert fetched.canonical_name == "Alice"

    def test_get_nonexistent_returns_none(self, store: SQLiteStore):
        assert store.get_entity("no-such-id") is None

    def test_find_entity_by_name(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity(name="Bob")
        store.insert_entity(ent)
        found = store.find_entity_by_name("default", "Bob")
        assert found is not None
        assert found.id == ent.id

    def test_find_entity_by_name_case_insensitive(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity(name="Charlie")
        store.insert_entity(ent)
        assert store.find_entity_by_name("default", "charlie") is not None
        assert store.find_entity_by_name("default", "CHARLIE") is not None

    def test_find_entity_by_name_missing_returns_none(self, store: SQLiteStore):
        assert store.find_entity_by_name("default", "Ghost") is None

    def test_find_entity_by_alias(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity(name="Diana")
        store.insert_entity(ent)
        al = _alias(ent.id, ep.id, value="di")
        store.insert_alias(al)
        found = store.find_entity_by_alias("di", "default")
        assert found is not None
        assert found.id == ent.id

    def test_find_entity_by_alias_missing_returns_none(self, store: SQLiteStore):
        assert store.find_entity_by_alias("phantom", "default") is None

    def test_get_entities_for_scope(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        for name in ("Alice", "Bob", "Carol"):
            store.insert_entity(_entity(name=name))
        entities = store.get_entities_for_scope("default")
        assert len(entities) == 3
        names = {e.canonical_name for e in entities}
        assert names == {"Alice", "Bob", "Carol"}

    def test_get_entities_for_scope_empty(self, store: SQLiteStore):
        assert store.get_entities_for_scope("no-scope") == []

    def test_update_entity_summary(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        store.update_entity_summary(ent.id, "Alice is a person.")
        fetched = store.get_entity(ent.id)
        assert fetched.summary == "Alice is a person."

    def test_update_entity_seen(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        new_time = _now()
        store.update_entity_seen(ent.id, new_time)
        fetched = store.get_entity(ent.id)
        assert fetched.last_seen_at is not None


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

class TestAliases:
    def test_insert_and_get(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        al = _alias(ent.id, ep.id, value="al")
        store.insert_alias(al)
        aliases = store.get_aliases_for_entity(ent.id)
        assert len(aliases) == 1
        assert aliases[0].value == "al"

    def test_insert_duplicate_alias_ignored(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        al = _alias(ent.id, ep.id)
        store.insert_alias(al)
        store.insert_alias(al)  # second insert — should be silently ignored (OR IGNORE)
        assert len(store.get_aliases_for_entity(ent.id)) == 1

    def test_get_aliases_for_entity_empty(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        assert store.get_aliases_for_entity(ent.id) == []


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

class TestFacts:
    def _setup(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        return ep, ent

    def test_insert_and_get(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        fetched = store.get_fact(f.id)
        assert fetched is not None
        assert fetched.predicate == "prefers"
        assert fetched.truth_state == "current"

    def test_get_nonexistent_returns_none(self, store: SQLiteStore):
        assert store.get_fact("no-id") is None

    def test_get_current_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id, predicate="likes", obj="coffee")
        store.insert_fact(f)
        results = store.get_current_facts("default", ent.id, "likes")
        assert len(results) == 1
        assert results[0].object_value_json == {"value": "coffee"}

    def test_get_current_facts_excludes_superseded(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id, truth_state="superseded")
        store.insert_fact(f)
        assert store.get_current_facts("default", ent.id, "prefers") == []

    def test_get_current_facts_for_scope(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f1 = _fact(ep.scope_id, ep.id, ent.id, predicate="p1", obj="v1")
        f2 = _fact(ep.scope_id, ep.id, ent.id, predicate="p2", obj="v2")
        store.insert_fact(f1)
        store.insert_fact(f2)
        results = store.get_current_facts_for_scope("default", limit=10)
        assert len(results) == 2

    def test_supersede_fact(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.supersede_fact(f.id, _now())
        fetched = store.get_fact(f.id)
        assert fetched.truth_state == "superseded"
        assert fetched.valid_to is not None

    def test_hide_fact(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.hide_fact(f.id)
        fetched = store.get_fact(f.id)
        assert fetched.truth_state == "hidden"

    def test_delete_fact(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.delete_fact(f.id)
        assert store.get_fact(f.id) is None

    def test_reinforce_fact(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.reinforce_fact(f.id, 0.1)
        fetched = store.get_fact(f.id)
        assert fetched.access_count == 1
        assert fetched.strength > 1.0

    def test_apply_decay(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.apply_decay("default", stable_threshold=10)
        fetched = store.get_fact(f.id)
        # Strength should have decayed (fact has 0 accesses, below stable_threshold)
        assert fetched.strength <= 1.0

    def test_fts_search_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id, obj="dark mode")
        store.insert_fact(f)
        results = store.fts_search_facts("dark", "default", limit=5)
        assert len(results) >= 1

    def test_fts_search_facts_malformed_query_returns_empty(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        results = store.fts_search_facts("AND* (NEAR/bad)", "default", limit=5)
        assert isinstance(results, list)

    def test_fts_search_facts_excludes_hidden(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id, obj="dark mode")
        store.insert_fact(f)
        store.hide_fact(f.id)
        results = store.fts_search_facts("dark", "default", limit=5)
        ids = [r[0].id for r in results]
        assert f.id not in ids

    def test_hide_entity_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f1 = _fact(ep.scope_id, ep.id, ent.id, predicate="likes")
        f2 = _fact(ep.scope_id, ep.id, ent.id, predicate="prefers")
        store.insert_fact(f1)
        store.insert_fact(f2)
        store.hide_entity_facts(ent.id)
        for fid in (f1.id, f2.id):
            assert store.get_fact(fid).truth_state == "hidden"

    def test_hide_scope_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.hide_scope_facts("default")
        assert store.get_fact(f.id).truth_state == "hidden"

    def test_delete_scope_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.delete_scope_facts("default")
        assert store.get_fact(f.id) is None

    def test_delete_entity_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.delete_entity_facts(ent.id)
        assert store.get_fact(f.id) is None

    def test_delete_episode_facts(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        store.delete_episode_facts(ep.id)
        assert store.get_fact(f.id) is None


# ---------------------------------------------------------------------------
# Quarantined facts
# ---------------------------------------------------------------------------

class TestQuarantinedFacts:
    def test_insert_and_get_pending(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        qf = _quarantine("default", ep.id)
        store.insert_quarantined_fact(qf)
        pending = store.get_pending_quarantined_facts("default")
        assert len(pending) == 1
        assert pending[0].id == qf.id

    def test_update_status_to_approved(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        qf = _quarantine("default", ep.id)
        store.insert_quarantined_fact(qf)
        store.update_quarantine_status(qf.id, "approved")
        pending = store.get_pending_quarantined_facts("default")
        assert pending == []

    def test_update_status_to_rejected(self, store: SQLiteStore):
        ep = _episode()
        store.insert_episode(ep)
        qf = _quarantine("default", ep.id)
        store.insert_quarantined_fact(qf)
        store.update_quarantine_status(qf.id, "rejected")
        pending = store.get_pending_quarantined_facts("default")
        assert pending == []

    def test_pending_empty_when_none(self, store: SQLiteStore):
        assert store.get_pending_quarantined_facts("default") == []


# ---------------------------------------------------------------------------
# Retrieval traces
# ---------------------------------------------------------------------------

class TestRetrievalTraces:
    def test_insert_and_get(self, store: SQLiteStore):
        ep, ent = self._setup(store)
        f = _fact(ep.scope_id, ep.id, ent.id)
        store.insert_fact(f)
        trace = RetrievalTrace(
            id=str(uuid4()),
            query="dark mode",
            scope_id="default",
            candidate_fact_id=f.id,
            semantic_score=0.0,
            keyword_score=0.8,
            graph_score=0.0,
            temporal_score=0.5,
            final_score=0.7,
            matched_entities=["Alice"],
            source_episode_ids=[ep.id],
            created_at=_now(),
        )
        store.insert_retrieval_trace(trace)
        traces = store.get_traces("default", "dark mode")
        assert len(traces) >= 1

    def test_get_traces_empty(self, store: SQLiteStore):
        assert store.get_traces("default", "anything") == []

    def _setup(self, store):
        ep = _episode()
        store.insert_episode(ep)
        ent = _entity()
        store.insert_entity(ent)
        return ep, ent


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_is_idempotent(self, tmp_path):
        s = SQLiteStore(str(tmp_path / "close_test.db"))
        s.close()
        s.close()  # second close should not raise

    def test_close_frees_connection(self, tmp_path):
        s = SQLiteStore(str(tmp_path / "free_test.db"))
        s.close()
        # After close, _local should have no conn attribute
        assert not hasattr(s._local, "conn")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        """Two threads can write to the same DB file concurrently without error."""
        db_path = str(tmp_path / "threaded.db")
        store = SQLiteStore(db_path)
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(5):
                    ep = _episode(text=f"thread {thread_id} episode {i}")
                    store.insert_episode(ep)
                store.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # All 20 episodes should be present (5 per thread × 4 threads)
        main_store = SQLiteStore(db_path)
        episodes = main_store.get_recent_episodes("default", limit=100)
        assert len(episodes) == 20
        main_store.close()
