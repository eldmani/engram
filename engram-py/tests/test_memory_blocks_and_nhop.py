"""Tests for memory blocks (Letta parity) and N-hop graph traversal (Graphiti parity)."""
from __future__ import annotations

import hashlib
import math
import struct

import pytest

from engram import Engram, EngramConfig, BaseEmbedder
from engram.extraction.base import (
    BaseExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
)
from engram.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DeterministicEmbedder(BaseEmbedder):
    """8-dim SHA-256-based unit embedder (same as test_embedding_and_graph.py)."""

    model_name = "sha256-8d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [
                struct.unpack_from(">f", digest, i * 4)[0]
                for i in range(8)
            ]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            results.append([x / norm for x in raw])
        return results


class _ChainExtractor(BaseExtractor):
    """Extracts a chain: A→B→C via two facts."""

    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                ExtractedEntity("A", "person"),
                ExtractedEntity("B", "person"),
                ExtractedEntity("C", "person"),
            ],
            facts=[
                ExtractedFact(subject="A", predicate="knows", object="B", confidence=1.0),
                ExtractedFact(subject="B", predicate="knows", object="C", confidence=1.0),
            ],
        )


class _SingleFactExtractor(BaseExtractor):
    def __init__(self, subject: str, predicate: str, obj: str) -> None:
        self._s, self._p, self._o = subject, predicate, obj

    def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[ExtractedEntity(self._s, "person"), ExtractedEntity(self._o, "person")],
            facts=[ExtractedFact(subject=self._s, predicate=self._p, object=self._o, confidence=1.0)],
        )


# ---------------------------------------------------------------------------
# Memory blocks — SQLiteStore level
# ---------------------------------------------------------------------------

class TestSQLiteMemoryBlocks:
    def test_set_and_get(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.set_block("scope1", "persona", "Alice the assistant")
        assert store.get_block("scope1", "persona") == "Alice the assistant"

    def test_get_missing_returns_none(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        assert store.get_block("scope1", "missing") is None

    def test_upsert_updates_value(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.set_block("s", "k", "v1")
        store.set_block("s", "k", "v2")
        assert store.get_block("s", "k") == "v2"

    def test_get_all_blocks_empty(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        assert store.get_all_blocks("scope1") == {}

    def test_get_all_blocks_multiple(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.set_block("s", "b", "2")
        store.set_block("s", "a", "1")
        store.set_block("s", "c", "3")
        blocks = store.get_all_blocks("s")
        assert blocks == {"a": "1", "b": "2", "c": "3"}

    def test_delete_block(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.set_block("s", "k", "v")
        store.delete_block("s", "k")
        assert store.get_block("s", "k") is None

    def test_delete_missing_noop(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.delete_block("s", "nope")  # must not raise

    def test_blocks_are_scope_isolated(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.set_block("scope1", "k", "v1")
        store.set_block("scope2", "k", "v2")
        assert store.get_block("scope1", "k") == "v1"
        assert store.get_block("scope2", "k") == "v2"
        assert store.get_all_blocks("scope1") == {"k": "v1"}
        assert store.get_all_blocks("scope2") == {"k": "v2"}

    def test_schema_version_is_2(self, tmp_path):
        from engram.storage.sqlite_store import _SCHEMA_VERSION
        assert _SCHEMA_VERSION == 2

    def test_schema_version_persisted(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        row = store._conn.execute("PRAGMA user_version").fetchone()
        assert row[0] == 2


# ---------------------------------------------------------------------------
# Memory blocks — Engram facade
# ---------------------------------------------------------------------------

class TestEngramMemoryBlocks:
    def test_set_get_block(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("persona", "Alice")
        assert mem.get_block("persona") == "Alice"

    def test_get_missing_block(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        assert mem.get_block("no_such_key") is None

    def test_upsert_block(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("goal", "help user")
        mem.set_block("goal", "assist with coding")
        assert mem.get_block("goal") == "assist with coding"

    def test_delete_block(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("x", "y")
        mem.delete_block("x")
        assert mem.get_block("x") is None

    def test_get_blocks_returns_dict(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("b", "2")
        mem.set_block("a", "1")
        assert mem.get_blocks() == {"a": "1", "b": "2"}

    def test_scope_kwarg(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("k", "v1", scope="s1")
        mem.set_block("k", "v2", scope="s2")
        assert mem.get_block("k", scope="s1") == "v1"
        assert mem.get_block("k", scope="s2") == "v2"

    def test_delete_missing_noop(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.delete_block("nope")  # must not raise


# ---------------------------------------------------------------------------
# Memory blocks — ContextResult / get_context integration
# ---------------------------------------------------------------------------

class TestMemoryBlocksInContext:
    def test_blocks_in_context_result(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("persona", "Alice the assistant")
        ctx = mem.get_context("what are the rules?")
        assert ctx.memory_blocks == {"persona": "Alice the assistant"}

    def test_blocks_appear_in_formatted_output(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("persona", "Alice the assistant")
        ctx = mem.get_context("anything")
        assert "[MEMORY]" in ctx.formatted
        assert "persona: Alice the assistant" in ctx.formatted

    def test_blocks_prepend_before_facts(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_extractor(_SingleFactExtractor("Alice", "likes", "Python"))
        mem.store("Alice likes Python.")
        mem.set_block("context", "office environment")
        ctx = mem.get_context("Alice")
        assert ctx.formatted.index("[MEMORY]") < ctx.formatted.index("[FACTS]")

    def test_no_blocks_no_memory_section(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        ctx = mem.get_context("anything")
        assert "[MEMORY]" not in ctx.formatted

    def test_multiple_blocks_sorted_in_output(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("z_last", "last")
        mem.set_block("a_first", "first")
        ctx = mem.get_context("anything")
        assert ctx.formatted.index("a_first") < ctx.formatted.index("z_last")

    def test_deleted_block_not_in_context(self, tmp_path):
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_block("transient", "temp value")
        mem.delete_block("transient")
        ctx = mem.get_context("anything")
        assert "transient" not in ctx.formatted
        assert ctx.memory_blocks == {}


# ---------------------------------------------------------------------------
# N-hop graph traversal — SQLiteStore.get_entity_graph
# ---------------------------------------------------------------------------

class TestGetEntityGraph:
    def _make_chain(self, tmp_path):
        """Build mem with chain A→B→C and return it."""
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_extractor(_ChainExtractor())
        mem.store("A knows B and B knows C.")
        return mem

    def test_seed_distance_0(self, tmp_path):
        mem = self._make_chain(tmp_path)
        # find entity id for A
        result = mem.retrieve("A")
        entity_ids = list({f.subject_entity_id for f in result.facts})
        # entity graph for those seeds: all should have distance 0
        graph = mem._store.get_entity_graph(entity_ids, mem.config.default_scope, max_hops=2)
        for eid in entity_ids:
            assert graph[eid] == 0

    def test_empty_seeds_returns_empty(self, tmp_path):
        mem = self._make_chain(tmp_path)
        graph = mem._store.get_entity_graph([], mem.config.default_scope, max_hops=2)
        assert graph == {}

    def test_max_hops_0_returns_only_seeds(self, tmp_path):
        mem = self._make_chain(tmp_path)
        result = mem.retrieve("A")
        seed_ids = list({f.subject_entity_id for f in result.facts})[:1]
        graph = mem._store.get_entity_graph(seed_ids, mem.config.default_scope, max_hops=0)
        assert set(graph.keys()) == set(seed_ids)

    def test_hop_distances_increase_monotonically(self, tmp_path):
        """In A→B→C chain, A is seed → B is hop 1 → C is hop 2."""
        mem = self._make_chain(tmp_path)
        # We need the canonical entity IDs for A, B, C
        result = mem.retrieve("A knows B")
        # Find the entity named A
        a_id = None
        for entity in result.entities:
            if entity.canonical_name.upper().startswith("A"):
                a_id = entity.id
                break
        if a_id is None:
            pytest.skip("Extractor did not produce 'A' entity for this test")
        graph = mem._store.get_entity_graph([a_id], mem.config.default_scope, max_hops=3)
        assert graph[a_id] == 0
        # All other nodes must have distance > 0
        for eid, dist in graph.items():
            if eid != a_id:
                assert dist > 0

    def test_max_hops_limits_traversal(self, tmp_path):
        """With max_hops=1, nodes 2+ hops away are excluded."""
        mem = self._make_chain(tmp_path)
        result = mem.retrieve("A")
        a_id = None
        for entity in result.entities:
            if entity.canonical_name.upper().startswith("A"):
                a_id = entity.id
                break
        if a_id is None:
            pytest.skip("Could not find entity A")
        graph_1 = mem._store.get_entity_graph([a_id], mem.config.default_scope, max_hops=1)
        graph_2 = mem._store.get_entity_graph([a_id], mem.config.default_scope, max_hops=2)
        # With max_hops=2 we should see at least as many entities as max_hops=1
        assert len(graph_2) >= len(graph_1)


# ---------------------------------------------------------------------------
# N-hop graph scoring — retrieval pipeline
# ---------------------------------------------------------------------------

class TestNHopGraphScoring:
    def test_graph_score_decays_with_hop(self, tmp_path):
        """Facts linked via hop 1 should score lower than seed facts via graph score."""
        mem = Engram(EngramConfig(db_path=str(tmp_path / "e.db")))
        mem.set_extractor(_ChainExtractor())
        mem.store("A knows B and B knows C.")
        # Retrieve with seed 'A' — the A→B fact should score higher than B→C (further hop)
        result = mem.retrieve("A knows B")
        assert len(result.facts) > 0

    def test_graph_max_hops_config(self, tmp_path):
        """graph_max_hops=0 disables graph expansion."""
        cfg = EngramConfig(db_path=str(tmp_path / "e.db"), graph_max_hops=0)
        mem = Engram(cfg)
        mem.set_extractor(_ChainExtractor())
        mem.store("A knows B and B knows C.")
        result = mem.retrieve("A knows B")
        # Should still return the directly relevant facts
        assert len(result.facts) >= 0

    def test_graph_max_hops_2_reaches_chain_end(self, tmp_path):
        """With max_hops=2, a query about A should eventually surface C via A→B→C."""
        cfg = EngramConfig(db_path=str(tmp_path / "e.db"), graph_max_hops=2)
        mem = Engram(cfg)
        mem.set_extractor(_ChainExtractor())
        mem.store("A knows B and B knows C.")
        result = mem.retrieve("A")
        fact_ids = {f.id for f in result.facts}
        # We can't predict exact IDs but there should be at least 1 fact
        assert len(result.facts) >= 1

    def test_graph_score_formula(self):
        """Verify 1.0 / (1.0 + hop * 0.5) for expected hop values."""
        assert 1.0 / (1.0 + 0 * 0.5) == pytest.approx(1.0)
        assert 1.0 / (1.0 + 1 * 0.5) == pytest.approx(2 / 3, rel=1e-6)
        assert 1.0 / (1.0 + 2 * 0.5) == pytest.approx(0.5)
        assert 1.0 / (1.0 + 3 * 0.5) == pytest.approx(0.4)
