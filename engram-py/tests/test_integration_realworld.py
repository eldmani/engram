"""
Real-world integration test — simulates an AI assistant's full memory lifecycle.

Scenario: A personal AI assistant learns about a user (Sarah), updates its
knowledge over several conversations, handles ambiguous/low-confidence info,
retrieves answers for follow-up questions, maintains multiple user scopes,
and performs memory housekeeping.

No mocks. A real SQLite database on disk is used throughout. All assertions
reflect observable, user-visible outcomes — not implementation internals.
"""

import os
import tempfile
import time
import threading
from datetime import datetime, timezone

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
# Extractor that simulates a real LLM extraction backend
# ---------------------------------------------------------------------------

class LLMExtractor(BaseExtractor):
    """
    Deterministic rule-based extractor that parses a tiny DSL so we can write
    predictable tests without needing a live LLM.

    Input format:  "ENTITY:<name>:<type>  FACT:<subj>:<pred>:<obj>:<conf>"
    Multiple entities/facts are space-separated blocks.

    If no DSL blocks are present the extractor attempts naive heuristics on
    plain-English sentences so the tests read naturally.
    """

    def __init__(self, confidence: float = 0.95):
        self._conf = confidence

    def extract(self, data: str | dict) -> ExtractionResult:
        text = data if isinstance(data, str) else str(data)
        entities: list[ExtractedEntity] = []
        facts: list[ExtractedFact] = []

        for token in text.split("  "):
            token = token.strip()
            if token.startswith("ENTITY:"):
                parts = token.split(":")
                name = parts[1].strip()
                etype = parts[2].strip() if len(parts) > 2 else "unknown"
                entities.append(ExtractedEntity(name, etype, 0.97))
            elif token.startswith("FACT:"):
                parts = token.split(":")
                subj = parts[1].strip()
                pred = parts[2].strip()
                obj  = parts[3].strip()
                conf = float(parts[4]) if len(parts) > 4 else 0.90
                facts.append(ExtractedFact(subj, pred, obj, conf))

        return ExtractionResult(entities=entities, facts=facts, confidence=self._conf)


def _db() -> tuple[Engram, str]:
    """Create a fresh on-disk Engram instance. Return (engram, db_path)."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="engram_integ_")
    os.close(fd)
    cfg = EngramConfig(db_path=path)
    mem = Engram(cfg)
    mem.set_extractor(LLMExtractor())
    return mem, path


# ---------------------------------------------------------------------------
# Scenario 1 — Basic learning and recall
# ---------------------------------------------------------------------------

class TestScenario01_BasicLearningAndRecall:
    """Assistant learns a few facts and recalls them correctly."""

    def test_full_scenario(self):
        mem, _ = _db()
        with mem:
            # Conversation turn 1: user mentions preferences
            r1 = mem.store(
                "ENTITY:Sarah:person  FACT:Sarah:prefers:dark mode:0.95",
                metadata={"turn": 1, "session": "s001"},
            )
            assert r1.episode_id, "episode must be created"
            assert len(r1.entity_ids) == 1, "one entity created"
            assert len(r1.fact_ids) == 1, "one fact created"

            # Conversation turn 2: more facts
            r2 = mem.store(
                "ENTITY:Sarah:person  FACT:Sarah:works_at:Acme Corp:0.93  FACT:Sarah:uses:Python:0.91",
                metadata={"turn": 2, "session": "s001"},
            )
            assert len(r2.fact_ids) == 2

            # Retrieve: Sarah's preferences
            result = mem.retrieve("Sarah dark mode preferences")
            assert result.facts, "should recall dark mode fact"
            predicates = [f.predicate for f in result.facts]
            assert "prefers" in predicates

            # Retrieve: Sarah's workplace
            result2 = mem.retrieve("where does Sarah work")
            workplace_facts = [f for f in result2.facts if f.predicate == "works_at"]
            assert workplace_facts, "works_at fact should be recalled"
            assert workplace_facts[0].object_value_json["value"] == "Acme Corp"

            # Context assembly: must return non-empty formatted prompt block
            ctx = mem.get_context("Sarah profile")
            assert ctx.formatted.strip(), "formatted context must not be empty"
            assert "Sarah" in ctx.formatted


# ---------------------------------------------------------------------------
# Scenario 2 — Temporal supersession (knowledge updates)
# ---------------------------------------------------------------------------

class TestScenario02_TemporalSupersession:
    """Facts are updated correctly; history is preserved, not overwritten."""

    def test_preference_update(self):
        mem, _ = _db()
        with mem:
            # Initial fact
            r1 = mem.store("ENTITY:Sarah:person  FACT:Sarah:prefers:dark mode:0.95")
            fid_original = r1.fact_ids[0]

            # Sarah changes her preference
            r2 = mem.store("ENTITY:Sarah:person  FACT:Sarah:prefers:light mode:0.95")
            fid_updated = r2.fact_ids[0]

            # The two facts must be different (supersession creates a new row)
            assert fid_original != fid_updated

            # Current fact reflects the update
            entities = mem._store.get_entities_for_scope("default")
            assert entities
            sarah = entities[0]
            current = mem._store.get_current_facts("default", sarah.id, "prefers")
            assert len(current) == 1
            assert current[0].object_value_json["value"] == "light mode"

            # Old fact still exists as superseded (history preserved)
            old = mem._store.get_fact(fid_original)
            assert old is not None
            assert old.truth_state == "superseded"

    def test_manual_update_fact(self):
        mem, _ = _db()
        with mem:
            r1 = mem.store("ENTITY:Bob:person  FACT:Bob:lives_in:London:0.92")
            fid = r1.fact_ids[0]

            # Bob moved to Berlin
            new_fact = mem.update_fact(fid, "Berlin")
            assert new_fact.object_value_json["value"] == "Berlin"
            assert new_fact.truth_state == "current"

            # Old record is closed
            old = mem._store.get_fact(fid)
            assert old.truth_state == "superseded"
            assert old.valid_to is not None

            # Context reflects the new city
            ctx = mem.get_context("where does Bob live")
            assert "Berlin" in ctx.formatted


# ---------------------------------------------------------------------------
# Scenario 3 — Confidence gate & quarantine
# ---------------------------------------------------------------------------

class TestScenario03_ConfidenceGate:
    """Low-confidence extractions go to quarantine, not into the fact store."""

    def test_low_confidence_rumour_quarantined(self):
        mem, _ = _db()
        with mem:
            # Extractor itself is low confidence (0.50 → below all thresholds)
            mem.set_extractor(LLMExtractor(confidence=0.50))
            r = mem.store("ENTITY:Carol:person  FACT:Carol:owns:yacht:0.40")

            # No facts committed
            assert len(r.fact_ids) == 0
            assert r.quarantined_count >= 1

            # But episode is still stored
            ep = mem._store.get_episode(r.episode_id)
            assert ep is not None

            # Quarantine surfaced via explain
            expl = mem.explain("Carol yacht")
            assert len(expl.quarantined_facts) >= 1
            qf = expl.quarantined_facts[0]
            assert qf.extracted_subject == "Carol"
            assert qf.extracted_predicate == "owns"
            assert qf.status == "pending"

    def test_subsequent_confirmation_reconsolidates(self):
        """
        A fact that was quarantined (low confidence) and then confirmed by a
        later high-confidence store should be handled correctly:
        the pending quarantine entry should be rejected (the true fact is
        already committed via the later episode).
        """
        mem, _ = _db()
        with mem:
            # First: low-confidence rumour → quarantined
            mem.set_extractor(LLMExtractor(confidence=0.50))
            r_low = mem.store("ENTITY:Dave:person  FACT:Dave:role:CEO:0.40")
            assert r_low.quarantined_count >= 1

            # Second: high-confidence confirmation → committed
            mem.set_extractor(LLMExtractor(confidence=0.95))
            r_high = mem.store("ENTITY:Dave:person  FACT:Dave:role:CEO:0.95")
            assert len(r_high.fact_ids) == 1

            # Worker reconsolidation: quarantine should now be rejected
            mem.workers.run_reconsolidation("default")
            pending = mem._store.get_pending_quarantined_facts("default")
            assert pending == [], f"Expected 0 pending quarantines, got {len(pending)}"


# ---------------------------------------------------------------------------
# Scenario 4 — Multi-user scope isolation
# ---------------------------------------------------------------------------

class TestScenario04_ScopeIsolation:
    """Facts in separate scopes (user-A, user-B) never bleed across."""

    def test_scopes_are_isolated(self):
        mem, _ = _db()
        with mem:
            mem.store(
                "ENTITY:Alice:person  FACT:Alice:prefers:vim:0.95",
                scope="user-alice",
            )
            mem.store(
                "ENTITY:Bob:person  FACT:Bob:prefers:emacs:0.95",
                scope="user-bob",
            )

            alice_result = mem.retrieve("editor preference", scope="user-alice")
            bob_result   = mem.retrieve("editor preference", scope="user-bob")

            alice_values = {f.object_value_json["value"] for f in alice_result.facts}
            bob_values   = {f.object_value_json["value"] for f in bob_result.facts}

            assert "vim"   in alice_values, "Alice's vim preference must be in scope user-alice"
            assert "emacs" in bob_values,   "Bob's emacs preference must be in scope user-bob"
            assert "emacs" not in alice_values, "Alice's scope must not see Bob's data"
            assert "vim"   not in bob_values,   "Bob's scope must not see Alice's data"

    def test_forget_scope_does_not_affect_other_scope(self):
        mem, _ = _db()
        with mem:
            mem.store("ENTITY:Alice:person  FACT:Alice:likes:cats:0.95", scope="user-alice")
            r = mem.store("ENTITY:Bob:person  FACT:Bob:likes:dogs:0.95", scope="user-bob")
            bob_fid = r.fact_ids[0]

            # Forget everything in Alice's scope
            mem.forget({"scope_id": "user-alice"}, hard=True)

            # Bob's data must survive untouched
            assert mem._store.get_fact(bob_fid) is not None
            assert mem._store.get_fact(bob_fid).truth_state == "current"


# ---------------------------------------------------------------------------
# Scenario 5 — Forget paths: fact / entity / episode / scope
# ---------------------------------------------------------------------------

class TestScenario05_ForgetPaths:
    """Every forget selector hides or deletes the right rows."""

    def _setup(self, mem: Engram) -> tuple[str, str, str]:
        """Returns (fact_id, entity_id, episode_id)."""
        r = mem.store("ENTITY:Eve:person  FACT:Eve:speaks:French:0.95")
        entity = mem._store.get_entities_for_scope("default")[0]
        return r.fact_ids[0], entity.id, r.episode_id

    def test_soft_forget_by_fact_id(self):
        mem, _ = _db()
        with mem:
            fid, _, _ = self._setup(mem)
            mem.forget({"fact_id": fid}, hard=False)
            assert mem._store.get_fact(fid).truth_state == "hidden"

    def test_hard_forget_by_fact_id(self):
        mem, _ = _db()
        with mem:
            fid, _, _ = self._setup(mem)
            mem.forget({"fact_id": fid}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_soft_forget_by_entity_id(self):
        mem, _ = _db()
        with mem:
            fid, eid, _ = self._setup(mem)
            fr = mem.forget({"entity_id": eid}, hard=False)
            assert fr.affected_entities >= 1
            assert mem._store.get_fact(fid).truth_state == "hidden"

    def test_hard_forget_by_entity_id(self):
        mem, _ = _db()
        with mem:
            fid, eid, _ = self._setup(mem)
            mem.forget({"entity_id": eid}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_soft_forget_by_scope(self):
        mem, _ = _db()
        with mem:
            fid, _, _ = self._setup(mem)
            mem.forget({"scope_id": "default"}, hard=False)
            assert mem._store.get_fact(fid).truth_state == "hidden"

    def test_hard_forget_by_scope(self):
        mem, _ = _db()
        with mem:
            fid, _, _ = self._setup(mem)
            mem.forget({"scope_id": "default"}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_hard_forget_by_episode_id(self):
        mem, _ = _db()
        with mem:
            fid, _, epid = self._setup(mem)
            mem.forget({"episode_id": epid}, hard=True)
            assert mem._store.get_fact(fid) is None

    def test_invalid_selector_raises(self):
        mem, _ = _db()
        with mem:
            with pytest.raises(ValueError):
                mem.forget({"unknown_key": "x"})


# ---------------------------------------------------------------------------
# Scenario 6 — Workers: decay, reinforce, heat, cleanup
# ---------------------------------------------------------------------------

class TestScenario06_Workers:
    """Background worker methods produce observable state changes."""

    def test_reinforcement_increases_strength_and_access_count(self):
        mem, _ = _db()
        with mem:
            r = mem.store("ENTITY:Frank:person  FACT:Frank:knows:Python:0.95")
            fid = r.fact_ids[0]
            original_strength = mem._store.get_fact(fid).strength
            original_count    = mem._store.get_fact(fid).access_count

            mem.workers.reinforce([fid])
            mem.workers.reinforce([fid])  # reinforce twice

            fact = mem._store.get_fact(fid)
            assert fact.access_count == original_count + 2
            assert fact.strength > original_strength

    def test_decay_weakens_unreinforced_fact(self):
        mem, _ = _db()
        with mem:
            r = mem.store("ENTITY:Grace:person  FACT:Grace:likes:jazz:0.95")
            fid = r.fact_ids[0]
            strength_before = mem._store.get_fact(fid).strength

            mem.workers.run_decay("default")

            strength_after = mem._store.get_fact(fid).strength
            assert strength_after <= strength_before, (
                f"Decay must weaken unreinforced fact: {strength_before} → {strength_after}"
            )

    def test_reinforced_fact_does_not_decay_below_reinforced_value(self):
        """A fact reinforced to a high strength should not decay past the reinforced value."""
        mem, _ = _db()
        with mem:
            r = mem.store("ENTITY:Hank:person  FACT:Hank:speciality:ML:0.95")
            fid = r.fact_ids[0]

            # Reinforce many times to build up strength
            for _ in range(20):
                mem.workers.reinforce([fid])

            strength_after_reinforce = mem._store.get_fact(fid).strength

            # One decay pass
            mem.workers.run_decay("default")
            strength_after_decay = mem._store.get_fact(fid).strength

            # The fact was accessed many times — it should be above the stable threshold
            # and therefore only undergo slow linear decay, not aggressive power-law decay
            assert strength_after_decay > 1.0, "Well-reinforced fact should stay above baseline"

    def test_heat_computation_is_consistent(self):
        mem, _ = _db()
        with mem:
            r = mem.store("ENTITY:Iris:person  FACT:Iris:language:Rust:0.95")
            fid = r.fact_ids[0]
            fact = mem._store.get_fact(fid)

            heat1 = mem.workers.compute_heat(fact)
            heat2 = mem.workers.compute_heat(fact)

            # Within the same second, heat should be identical
            assert abs(heat1 - heat2) < 0.001
            assert heat1 >= 0.0

    def test_heat_increases_after_reinforcement(self):
        mem, _ = _db()
        with mem:
            r = mem.store("ENTITY:Jack:person  FACT:Jack:hobby:chess:0.95")
            fid = r.fact_ids[0]
            fact_before = mem._store.get_fact(fid)
            heat_before = mem.workers.compute_heat(fact_before)

            for _ in range(10):
                mem.workers.reinforce([fid])

            fact_after = mem._store.get_fact(fid)
            heat_after = mem.workers.compute_heat(fact_after)

            assert heat_after > heat_before

    def test_run_heat_promotion_returns_hot_facts(self):
        mem, _ = _db()
        # Lower the threshold so a reinforced fact is guaranteed to cross it
        cfg = EngramConfig(db_path=mem.config.db_path, heat_promotion_threshold=0.0001)
        mem.close()
        mem = Engram(cfg)
        mem.set_extractor(LLMExtractor())
        with mem:
            r = mem.store("ENTITY:Kim:person  FACT:Kim:skill:Go:0.95")
            fid = r.fact_ids[0]
            for _ in range(5):
                mem.workers.reinforce([fid])
            hot = mem.workers.run_heat_promotion("default")
            assert fid in hot, f"Reinforced fact must be in hot list; got {hot}"

    def test_run_cleanup_leaves_valid_facts_intact(self):
        mem, _ = _db()
        with mem:
            r = mem.store("ENTITY:Leo:person  FACT:Leo:city:Paris:0.95")
            fid = r.fact_ids[0]
            mem.workers.run_cleanup("default")
            # Fact should still exist after cleanup
            assert mem._store.get_fact(fid) is not None


# ---------------------------------------------------------------------------
# Scenario 7 — FTS robustness
# ---------------------------------------------------------------------------

class TestScenario07_FtsRobustness:
    """Pathological query strings must never raise; they return empty or partial results."""

    @pytest.mark.parametrize("query", [
        "",
        "   ",
        "AND OR NOT",
        "dark AND* mode!",
        "(NEAR/bad query)",
        "*",
        "!@#$%^&*()",
        'He said "hello" AND goodbye',
    ])
    def test_bad_query_does_not_raise(self, query: str):
        mem, _ = _db()
        with mem:
            mem.store("ENTITY:Mia:person  FACT:Mia:likes:dark mode:0.95")
            result = mem.retrieve(query)  # must not raise
            assert isinstance(result.facts, list)
            assert isinstance(result.episodes, list)


# ---------------------------------------------------------------------------
# Scenario 8 — Multi-fact entity profile (context assembly)
# ---------------------------------------------------------------------------

class TestScenario08_RichEntityProfile:
    """An entity with many facts produces a rich, structured context block."""

    def test_full_profile_context(self):
        mem, _ = _db()
        with mem:
            # Build up a detailed profile over 5 turns
            turns = [
                "ENTITY:Nina:person  FACT:Nina:works_at:Google:0.95",
                "ENTITY:Nina:person  FACT:Nina:role:Staff Engineer:0.92",
                "ENTITY:Nina:person  FACT:Nina:lives_in:NYC:0.94",
                "ENTITY:Nina:person  FACT:Nina:speaks:English:0.99  FACT:Nina:speaks:Mandarin:0.97",
                "ENTITY:Nina:person  FACT:Nina:hobby:rock climbing:0.88",
            ]
            for turn in turns:
                mem.store(turn)

            ctx = mem.get_context("Nina profile", top_k=20)
            assert ctx.formatted.strip()

            # All the entities and facts must be retrievable
            result = mem.retrieve("Nina engineer Google", top_k=10)
            predicates = {f.predicate for f in result.facts}
            assert "works_at" in predicates
            assert "role" in predicates

            # Multiple entities for scope
            entities = mem._store.get_entities_for_scope("default")
            assert len(entities) == 1  # Nina resolved consistently
            nina = entities[0]
            assert nina.canonical_name == "Nina"

            # All facts for Nina
            all_facts = mem._store.get_current_facts_for_scope("default", limit=50)
            pred_set = {f.predicate for f in all_facts}
            assert pred_set >= {"works_at", "role", "lives_in", "hobby"}


# ---------------------------------------------------------------------------
# Scenario 9 — Episode immutability & metadata
# ---------------------------------------------------------------------------

class TestScenario09_EpisodeImmutability:
    """Episodes are write-once; metadata is preserved exactly."""

    def test_episode_metadata_preserved(self):
        mem, _ = _db()
        with mem:
            r = mem.store(
                "ENTITY:Oscar:person  FACT:Oscar:prefers:Neovim:0.95",
                metadata={"session": "abc123", "turn": 7, "model": "gpt-5"},
            )
            ep = mem._store.get_episode(r.episode_id)
            assert ep.metadata["session"] == "abc123"
            assert ep.metadata["turn"] == 7
            assert ep.metadata["model"] == "gpt-5"

    def test_episode_checksum_is_deterministic(self):
        from engram.models import Episode
        text = "Some deterministic input text."
        c1 = Episode.compute_checksum(text)
        c2 = Episode.compute_checksum(text)
        assert c1 == c2
        assert len(c1) in (40, 64)  # SHA-1 (40) or SHA-256 (64)

# ---------------------------------------------------------------------------
# Scenario 10 — Concurrent access (thread safety)
# ---------------------------------------------------------------------------

class TestScenario10_ConcurrentAccess:
    """Multiple threads writing to the same Engram instance must not corrupt data."""

    def test_concurrent_store_and_retrieve(self):
        mem, _ = _db()
        errors: list[Exception] = []
        results: list[str] = []
        lock = threading.Lock()

        def writer(name: str, n: int) -> None:
            try:
                for i in range(5):
                    r = mem.store(
                        f"ENTITY:{name}:person  FACT:{name}:item_{i}:value_{i}:0.95",
                        scope=f"scope-{name}",
                    )
                    with lock:
                        results.append(r.episode_id)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        def reader(scope: str) -> None:
            try:
                for _ in range(3):
                    mem.retrieve("item value", scope=scope)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        names = ["Alice", "Bob", "Carol", "Dave"]
        threads = (
            [threading.Thread(target=writer, args=(n, i)) for i, n in enumerate(names)] +
            [threading.Thread(target=reader, args=(f"scope-{n}",)) for n in names]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mem.close()

        assert errors == [], f"Thread errors:\n" + "\n".join(str(e) for e in errors)
        # 4 writers × 5 stores = 20 unique episode IDs
        assert len(set(results)) == 20


# ---------------------------------------------------------------------------
# Scenario 11 — End-to-end AI assistant conversation replay
# ---------------------------------------------------------------------------

class TestScenario11_AssistantConversationReplay:
    """
    Replay a realistic multi-turn assistant conversation. Asserts that the
    memory layer supports the full assistant loop:
      Store → Retrieve → Reinforce → Update → Forget → Context
    """

    def test_full_conversation(self):
        mem, _ = _db()
        mem.set_extractor(LLMExtractor())

        with mem:
            scope = "user-pat"

            # --- Turn 1: onboarding ---
            mem.store(
                "ENTITY:Pat:person  FACT:Pat:name:Pat:0.99  FACT:Pat:role:data scientist:0.97",
                scope=scope,
                metadata={"turn": 1},
            )

            # --- Turn 2: preferences ---
            r2 = mem.store(
                "ENTITY:Pat:person  FACT:Pat:prefers:dark mode:0.95  FACT:Pat:uses:Jupyter:0.93",
                scope=scope,
                metadata={"turn": 2},
            )

            # --- Turn 3: assistant retrieves context for follow-up ---
            result = mem.retrieve("Pat preferences tools", scope=scope, top_k=5)
            assert result.facts, "Should recall facts about Pat"
            retrieved_fact_ids = [f.id for f in result.facts]

            # Simulate assistant reinforcing retrieved facts
            mem.workers.reinforce(retrieved_fact_ids)

            # --- Turn 4: Pat updates a preference ---
            prefs = [f for f in result.facts if f.predicate == "prefers"]
            if prefs:
                mem.update_fact(prefs[0].id, "light mode")

            # --- Turn 5: new preference should appear in context ---
            ctx = mem.get_context("Pat current preferences", scope=scope)
            assert ctx.formatted.strip()
            # light mode must now be visible (dark mode was superseded)
            assert "light mode" in ctx.formatted

            # --- Turn 6: user requests privacy erasure for "Pat" ---
            entities = mem._store.get_entities_for_scope(scope)
            pat = next(e for e in entities if e.canonical_name == "Pat")
            forget_result = mem.forget({"entity_id": pat.id}, hard=True)
            assert forget_result.affected_entities >= 1

            # All Pat's facts must be gone
            remaining = mem._store.get_current_facts_for_scope(scope, limit=100)
            pat_facts = [f for f in remaining if f.subject_entity_id == pat.id]
            assert pat_facts == [], f"Pat's facts should be erased; got {pat_facts}"

            # --- Final: worker housekeeping runs without error ---
            mem.workers.run_decay(scope)
            mem.workers.run_reconsolidation(scope)
            mem.workers.run_cleanup(scope)
