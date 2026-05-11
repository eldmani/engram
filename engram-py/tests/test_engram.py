"""Tests for engram Python library."""

import os
import tempfile

import pytest

from engram import (
    Engram,
    EngramConfig,
    BaseExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tmp_engram() -> Engram:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Engram(EngramConfig(db_path=path))


class FixedExtractor(BaseExtractor):
    """Deterministic extractor for testing."""

    def __init__(
        self,
        entities: list[ExtractedEntity],
        facts: list[ExtractedFact],
        confidence: float = 0.95,
    ) -> None:
        self._entities = entities
        self._facts = facts
        self._confidence = confidence

    def extract(self, data: str | dict) -> ExtractionResult:
        return ExtractionResult(
            entities=self._entities,
            facts=self._facts,
            confidence=self._confidence,
        )


# ---------------------------------------------------------------------------
# Basic store / retrieve with NullExtractor
# ---------------------------------------------------------------------------

class TestBasicStoreRetrieve:
    def test_store_returns_episode_id(self):
        with tmp_engram() as mem:
            result = mem.store("Alice prefers dark mode.")
            assert result.episode_id
            assert result.fact_ids == []
            assert result.entity_ids == []
            assert result.quarantined_count == 0

    def test_store_dict(self):
        with tmp_engram() as mem:
            result = mem.store({"key": "value"})
            assert result.episode_id

    def test_retrieve_returns_episode(self):
        with tmp_engram() as mem:
            mem.store("Alice prefers dark mode.")
            result = mem.retrieve("dark mode")
            assert len(result.episodes) >= 1
            assert any("dark mode" in ep.raw_text for ep in result.episodes)

    def test_retrieve_empty_scope(self):
        with tmp_engram() as mem:
            result = mem.retrieve("anything", scope="empty-scope")
            assert result.facts == []
            assert result.episodes == []

    def test_get_context_returns_formatted(self):
        with tmp_engram() as mem:
            mem.store("Alice prefers dark mode and vim keybindings.")
            ctx = mem.get_context("what does Alice prefer?")
            assert ctx.formatted
            assert "EPISODES" in ctx.formatted or "FACTS" in ctx.formatted

    def test_scope_isolation(self):
        with tmp_engram() as mem:
            mem.store("Alice likes cats.", scope="user-1")
            mem.store("Bob likes dogs.", scope="user-2")
            r1 = mem.retrieve("likes", scope="user-1")
            r2 = mem.retrieve("likes", scope="user-2")
            ep_texts_1 = [ep.raw_text for ep in r1.episodes]
            ep_texts_2 = [ep.raw_text for ep in r2.episodes]
            assert all("cats" in t for t in ep_texts_1)
            assert all("dogs" in t for t in ep_texts_2)


# ---------------------------------------------------------------------------
# Extractor + entity resolution + facts
# ---------------------------------------------------------------------------

class TestWithExtractor:
    def _make_mem(self, entities, facts, confidence=0.95) -> Engram:
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(entities, facts, confidence))
        return mem

    def test_store_creates_entities_and_facts(self):
        mem = self._make_mem(
            entities=[ExtractedEntity("Alice", "person", 0.95)],
            facts=[ExtractedFact("Alice", "prefers", "dark mode", 0.92)],
        )
        with mem:
            result = mem.store("Alice prefers dark mode.")
            assert len(result.entity_ids) == 1
            assert len(result.fact_ids) == 1
            assert result.quarantined_count == 0

    def test_retrieve_returns_facts(self):
        mem = self._make_mem(
            entities=[ExtractedEntity("Alice", "person", 0.95)],
            facts=[ExtractedFact("Alice", "prefers", "dark mode", 0.92)],
        )
        with mem:
            mem.store("Alice prefers dark mode.")
            result = mem.retrieve("Alice preference")
            assert len(result.facts) >= 1
            assert result.facts[0].predicate == "prefers"

    def test_entity_resolved_on_second_store(self):
        mem = self._make_mem(
            entities=[ExtractedEntity("Alice", "person", 0.95)],
            facts=[ExtractedFact("Alice", "likes", "coffee", 0.90)],
        )
        with mem:
            mem.store("Alice likes coffee.")
            mem.store("Alice likes coffee.")  # same entity, same predicate
            # Should resolve to existing entity, not create a new one
            entities = mem._store.get_entities_for_scope("default")
            assert len(entities) == 1

    def test_temporal_supersession(self):
        mem = tmp_engram()
        ext1 = FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Adidas", 0.92)],
        )
        mem.set_extractor(ext1)
        mem.store("Alice prefers Adidas.")

        ext2 = FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Nike", 0.92)],
        )
        mem.set_extractor(ext2)
        mem.store("Alice prefers Nike now.")

        result = mem.retrieve("Alice preference")
        current_facts = [f for f in result.facts if f.truth_state == "current"]
        assert len(current_facts) == 1
        assert current_facts[0].object_value_json == {"value": "Nike"}
        mem.close()


# ---------------------------------------------------------------------------
# Confidence gate / quarantine
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    def test_low_confidence_quarantines_fact(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            entities=[ExtractedEntity("Alice", "person", 0.95)],
            facts=[ExtractedFact("Alice", "prefers", "dark mode", 0.50)],
            confidence=0.50,  # below extractor_confidence_min=0.70
        ))
        with mem:
            result = mem.store("Alice maybe prefers dark mode.")
            assert result.quarantined_count == 1
            assert result.fact_ids == []

    def test_low_supersession_confidence_quarantines(self):
        """A supersession with confidence < 0.85 should not overwrite the existing fact."""
        mem = tmp_engram()
        # First store with high confidence
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Adidas", 0.95)],
            confidence=0.95,
        ))
        mem.store("Alice prefers Adidas.")

        # Second store with borderline confidence — should quarantine the supersession
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Nike", 0.80)],
            confidence=0.80,  # below supersession_confidence_min=0.85
        ))
        result = mem.store("Alice might prefer Nike.")
        assert result.quarantined_count >= 1

        # Original fact should still be current
        entities = mem._store.get_entities_for_scope("default")
        alice = entities[0]
        current = mem._store.get_current_facts("default", alice.id, "prefers")
        assert len(current) == 1
        assert current[0].object_value_json == {"value": "Adidas"}
        mem.close()

    def test_explain_surfaces_quarantine(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "dark mode", 0.50)],
            confidence=0.50,
        ))
        with mem:
            mem.store("Alice maybe prefers dark mode.")
            expl = mem.explain("Alice preference")
            assert len(expl.quarantined_facts) >= 1
            assert "quarantine" in expl.summary.lower() or "Quarantine" in expl.summary


# ---------------------------------------------------------------------------
# update_fact
# ---------------------------------------------------------------------------

class TestUpdateFact:
    def test_update_fact_creates_new_version(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "Adidas", 0.95)],
            confidence=0.95,
        ))
        with mem:
            result = mem.store("Alice prefers Adidas.")
            fact_id = result.fact_ids[0]
            new_fact = mem.update_fact(fact_id, "Nike")
            assert new_fact.truth_state == "current"
            assert new_fact.object_value_json == {"value": "Nike"}
            old = mem._store.get_fact(fact_id)
            assert old.truth_state == "superseded"
            assert old.valid_to is not None

    def test_update_nonexistent_fact_raises(self):
        with tmp_engram() as mem:
            with pytest.raises(ValueError):
                mem.update_fact("nonexistent-id", "value")


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------

class TestForget:
    def test_soft_forget_hides_fact(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "dark mode", 0.95)],
            confidence=0.95,
        ))
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fact_id = result.fact_ids[0]
            mem.forget({"fact_id": fact_id}, hard=False)
            fact = mem._store.get_fact(fact_id)
            assert fact.truth_state == "hidden"

    def test_hard_forget_deletes_fact(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "dark mode", 0.95)],
            confidence=0.95,
        ))
        with mem:
            result = mem.store("Alice prefers dark mode.")
            fact_id = result.fact_ids[0]
            mem.forget({"fact_id": fact_id}, hard=True)
            assert mem._store.get_fact(fact_id) is None

    def test_forget_invalid_selector_raises(self):
        with tmp_engram() as mem:
            with pytest.raises(ValueError):
                mem.forget({"unknown_key": "x"})


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class TestWorkers:
    def test_decay_runs_without_error(self):
        with tmp_engram() as mem:
            mem.store("Alice prefers dark mode.")
            mem.workers.run_decay("default")  # should not raise

    def test_reconsolidation_runs_without_error(self):
        mem = tmp_engram()
        mem.set_extractor(FixedExtractor(
            [ExtractedEntity("Alice", "person", 0.95)],
            [ExtractedFact("Alice", "prefers", "dark mode", 0.50)],
            confidence=0.50,
        ))
        with mem:
            mem.store("Alice maybe prefers dark mode.")
            mem.workers.run_reconsolidation("default")  # should not raise
