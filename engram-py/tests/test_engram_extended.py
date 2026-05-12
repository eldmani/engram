"""
Extended integration tests for Engram — covers all paths not exercised by
the baseline test_engram.py suite: forget selectors, update_fact variants,
all worker methods, config, __version__, NullExtractor, and thread safety.
"""

import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import pytest

import engram as engram_pkg
from engram import (
    Engram,
    EngramConfig,
    BaseExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    NullExtractor,
)
from engram.config import EngramConfig as Cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tmp_engram(**kwargs) -> Engram:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Engram(EngramConfig(db_path=path, **kwargs))


class FixedExtractor(BaseExtractor):
    def __init__(self, entities, facts, confidence=0.95):
        self._entities = entities
        self._facts = facts
        self._confidence = confidence

    def extract(self, data):
        return ExtractionResult(
            entities=self._entities,
            facts=self._facts,
            confidence=self._confidence,
        )


def _alice_mem(extra_facts=None, confidence=0.95) -> Engram:
    facts = [ExtractedFact("Alice", "prefers", "dark mode", 0.95)]
    if extra_facts:
        facts += extra_facts
    mem = tmp_engram()
    mem.set_extractor(FixedExtractor(
        [ExtractedEntity("Alice", "person", 0.95)],
        facts,
        confidence,
    ))
    return mem


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    def test_version_is_string(self):
        assert isinstance(engram_pkg.__version__, str)

    def test_version_value(self):
        assert engram_pkg.__version__ == "0.1.0"

    def test_public_names_exported(self):
        for name in ("Engram", "EngramConfig", "BaseExtractor", "NullExtractor",
                     "ExtractionResult", "ExtractedEntity", "ExtractedFact"):
            assert hasattr(engram_pkg, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# EngramConfig
# ---------------------------------------------------------------------------

class TestEngramConfig:
    def test_defaults(self):
        cfg = Cfg()
        assert cfg.default_scope == "default"
        assert cfg.extractor_confidence_min == 0.70
        assert cfg.resolution_confidence_min == 0.75
        assert cfg.supersession_confidence_min == 0.85
        assert cfg.weight_keyword == 0.30
        assert cfg.weight_semantic == 0.40
        assert cfg.weight_graph == 0.10
        assert cfg.weight_temporal == 0.10

    def test_custom_values(self):
        cfg = Cfg(default_scope="bot-1", default_top_k=5)
        assert cfg.default_scope == "bot-1"
        assert cfg.default_top_k == 5

    def test_engram_uses_config(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cfg = Cfg(db_path=path, default_scope="custom")
        with Engram(cfg) as mem:
            result = mem.store("anything", scope=None)  # should use custom scope
            # The episode should exist in "custom" scope
            r = mem.retrieve("anything", scope="custom")
            assert len(r.episodes) >= 1


# ---------------------------------------------------------------------------
# NullExtractor
# ---------------------------------------------------------------------------

class TestNullExtractor:
    def test_extract_returns_empty(self):
        ex = NullExtractor()
        result = ex.extract("anything")
        assert isinstance(result, ExtractionResult)
        assert result.entities == []
        assert result.facts == []

    def test_extract_with_dict(self):
        ex = NullExtractor()
        result = ex.extract({"key": "value"})
        assert result.entities == []


# ---------------------------------------------------------------------------
# store() variations
# ---------------------------------------------------------------------------

class TestStoreVariations:
    def test_store_with_metadata(self):
        with tmp_engram() as mem:
            result = mem.store("event happened", metadata={"source": "log", "ts": 123})
            assert result.episode_id
            ep = mem._store.get_episode(result.episode_id)
            assert ep.metadata["source"] == "log"
            assert ep.metadata["ts"] == 123

    def test_store_multiple_entities_and_facts(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            entities=[
                ExtractedEntity("Alice", "person", 0.95),
                ExtractedEntity("Bob", "person", 0.95),
            ],
            facts=[
                ExtractedFact("Alice", "knows", "Bob", 0.92),
                ExtractedFact("Bob", "likes", "chess", 0.90),
            ],
        ))
        with mem:
            result = mem.store("Alice knows Bob. Bob likes chess.")
            assert len(result.entity_ids) == 2
            assert len(result.fact_ids) == 2

    def test_store_same_content_has_consistent_checksum(self):
        """Same raw text produces the same checksum on both episodes."""
        with tmp_engram() as mem:
            r1 = mem.store("Exactly the same text.")
            r2 = mem.store("Exactly the same text.")
            ep1 = mem._store.get_episode(r1.episode_id)
            ep2 = mem._store.get_episode(r2.episode_id)
            # Checksums must be deterministic and identical
            assert ep1.checksum == ep2.checksum
            # They are separate episode records (pipeline does not deduplicate)
            assert ep1.id != ep2.id


# ---------------------------------------------------------------------------
# retrieve() / get_context() with additional options
# ---------------------------------------------------------------------------

class TestRetrieveOptions:
    def test_retrieve_with_top_k(self):
        with tmp_engram() as mem:
            for i in range(10):
                mem.store(f"fact about topic {i}")
            result = mem.retrieve("topic", top_k=3)
            assert len(result.episodes) <= 3

    def test_retrieve_with_as_of(self):
        """as_of is accepted by retrieve() without error; FTS fact search respects it."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mem = Engram(EngramConfig(db_path=path))
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "likes", "coffee", 0.95)],
        ))
        mem.store("Alice likes coffee.")
        result = mem.retrieve("coffee", as_of=past)
        # as_of is accepted without error; result is a valid RetrievalResult
        assert hasattr(result, "facts")
        assert hasattr(result, "episodes")
        # The FTS keyword search layer filters by as_of — verify directly
        fact_hits = mem._store.fts_search_facts("coffee", "default", 10, as_of=past)
        assert fact_hits == []  # valid_from is now > past => filtered
        mem.close()

    def test_get_context_custom_scope(self):
        with tmp_engram() as mem:
            mem.store("Alice is in scope X.", scope="scope-x")
            ctx = mem.get_context("Alice", scope="scope-x")
            assert ctx.formatted

    def test_get_context_top_k(self):
        with tmp_engram() as mem:
            for i in range(10):
                mem.store(f"alpha item {i}")
            ctx = mem.get_context("alpha", top_k=2)
            assert ctx.formatted


# ---------------------------------------------------------------------------
# update_fact() variations
# ---------------------------------------------------------------------------

class TestUpdateFactVariations:
    def _stored_fact_id(self, mem: Engram) -> str:
        result = mem.store("Alice prefers Adidas.")
        return result.fact_ids[0]

    def test_update_with_dict_value(self):
        mem = _alice_mem([ExtractedFact("Alice", "prefers", "Adidas", 0.95)])
        with mem:
            result = mem.store("Alice prefers Adidas.")
            fid = result.fact_ids[0]
            new_fact = mem.update_fact(fid, {"value": "New Balance", "certainty": 0.9})
            assert new_fact.object_value_json == {"value": "New Balance", "certainty": 0.9}

    def test_update_with_explicit_valid_from(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            explicit_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
            new_fact = mem.update_fact(fid, "light mode", valid_from=explicit_time)
            assert new_fact.valid_from == explicit_time

    def test_update_preserves_scope(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "dark mode", 0.95)],
        ))
        with mem:
            result = mem.store("Alice prefers dark mode.", scope="scope-a")
            fid = result.fact_ids[0]
            new_fact = mem.update_fact(fid, "light mode")
            assert new_fact.scope_id == "scope-a"


# ---------------------------------------------------------------------------
# forget() — all selector paths
# ---------------------------------------------------------------------------

class TestForgetSelectors:
    def _mem_with_fact(self):
        mem = _alice_mem()
        result = mem.store("Alice prefers dark mode.")
        return mem, result.fact_ids[0], result.episode_id

    def test_soft_forget_entity(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            entities = mem._store.get_entities_for_scope("default")
            eid = entities[0].id
            forget_result = mem.forget({"entity_id": eid}, hard=False)
            assert forget_result.affected_entities >= 1
            # The specific fact should now be hidden
            assert mem._store.get_fact(fid).truth_state == "hidden"

    def test_hard_forget_entity(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            entities = mem._store.get_entities_for_scope("default")
            eid = entities[0].id
            mem.forget({"entity_id": eid}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_hard_forget_episode(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            mem.forget({"episode_id": result.episode_id}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_soft_forget_episode_is_noop_for_facts(self):
        """Soft forget by episode_id doesn't hide facts (by design — only hard does)."""
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            mem.forget({"episode_id": result.episode_id}, hard=False)
            fact = mem._store.get_fact(fid)
            # Fact still exists (soft episode forget is a no-op on facts)
            assert fact is not None

    def test_soft_forget_scope(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            forget_result = mem.forget({"scope_id": "default"}, hard=False)
            assert forget_result.affected_facts >= 1
            assert mem._store.get_fact(fid).truth_state == "hidden"

    def test_hard_forget_scope(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            mem.forget({"scope_id": "default"}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_forget_result_fields(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            fr = mem.forget({"fact_id": fid})
            assert hasattr(fr, "affected_facts")
            assert hasattr(fr, "affected_entities")
            assert hasattr(fr, "affected_aliases")


# ---------------------------------------------------------------------------
# Background workers — full coverage
# ---------------------------------------------------------------------------

class TestWorkersExtended:
    def test_reinforce_boosts_access_count(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            before = mem._store.get_fact(fid)
            mem.workers.reinforce([fid])
            after = mem._store.get_fact(fid)
            assert after.access_count == before.access_count + 1
            assert after.strength >= before.strength

    def test_reinforce_empty_list_is_noop(self):
        with tmp_engram() as mem:
            mem.workers.reinforce([])  # should not raise

    def test_compute_heat_returns_float(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            fact = mem._store.get_fact(fid)
            heat = mem.workers.compute_heat(fact)
            assert isinstance(heat, float)
            assert heat >= 0.0

    def test_run_heat_promotion_returns_list(self):
        mem = _alice_mem()
        with mem:
            mem.store("Alice prefers dark mode.")
            hot = mem.workers.run_heat_promotion("default")
            assert isinstance(hot, list)

    def test_run_decay_reduces_strength(self):
        mem = _alice_mem()
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fid = result.fact_ids[0]
            before_strength = mem._store.get_fact(fid).strength
            mem.workers.run_decay("default")
            after_strength = mem._store.get_fact(fid).strength
            assert after_strength <= before_strength

    def test_run_reconsolidation_rejects_superseded_quarantine(self):
        """
        A quarantined fact whose predicate now has a confirmed current fact
        should be rejected during reconsolidation.
        """
        mem = tmp_engram()
        # First: store with high confidence to commit a current fact
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Adidas", 0.95)],
        ))
        mem.store("Alice prefers Adidas.")

        # Second: store with low confidence → quarantined
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Nike", 0.50)],
            confidence=0.50,
        ))
        mem.store("Alice might prefer Nike.")
        pending_before = mem._store.get_pending_quarantined_facts("default")
        assert len(pending_before) >= 1

        # Reconsolidate — predicate "prefers" already confirmed → should reject
        mem.workers.run_reconsolidation("default")
        pending_after = mem._store.get_pending_quarantined_facts("default")
        assert len(pending_after) < len(pending_before)
        mem.close()

    def test_run_cleanup_runs_without_error(self):
        mem = _alice_mem()
        with mem:
            mem.store("Alice prefers dark mode.")
            mem.workers.run_cleanup("default")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_enter_returns_self(self):
        mem = tmp_engram()
        with mem as m:
            assert m is mem

    def test_exit_closes_store(self):
        mem = tmp_engram()
        with mem:
            mem.store("test")
        # After __exit__, the connection should be closed
        assert not hasattr(mem._store._local, "conn")

    def test_nested_stores_independent(self):
        with tmp_engram() as m1, tmp_engram() as m2:
            m1.store("for m1")
            m2.store("for m2")
            assert m1.retrieve("m1").episodes[0].raw_text != \
                   m2.retrieve("m2").episodes[0].raw_text


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_no_quarantine(self):
        with tmp_engram() as mem:
            mem.store("Alice likes coffee.")
            expl = mem.explain("Alice")
            assert "0" in expl.summary or "No" in expl.summary or expl.quarantined_facts == []

    def test_explain_returns_traces_after_retrieve(self):
        with tmp_engram() as mem:
            mem.store("Alice likes coffee.")
            mem.retrieve("Alice likes coffee")
            expl = mem.explain("Alice likes coffee")
            assert isinstance(expl.traces, list)
            assert isinstance(expl.quarantined_facts, list)
            assert isinstance(expl.summary, str)


# ---------------------------------------------------------------------------
# FTS robustness via Engram.retrieve()
# ---------------------------------------------------------------------------

class TestFtsRobustness:
    def test_retrieve_with_special_chars_does_not_raise(self):
        with tmp_engram() as mem:
            mem.store("Alice prefers dark mode.")
            # FTS special characters — must not raise
            result = mem.retrieve("dark AND* OR(mode!)")
            assert isinstance(result.episodes, list)

    def test_retrieve_empty_query(self):
        with tmp_engram() as mem:
            mem.store("some content")
            result = mem.retrieve("")
            assert isinstance(result.episodes, list)


# ---------------------------------------------------------------------------
# Thread safety at Engram level
# ---------------------------------------------------------------------------

class TestEngramThreadSafety:
    def test_concurrent_store_calls(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mem = Engram(EngramConfig(db_path=path))
        errors = []

        def worker(n: int) -> None:
            try:
                for i in range(5):
                    mem.store(f"thread {n} item {i}", scope=f"scope-{n}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        mem.close()

        assert errors == [], f"Thread errors: {errors}"
