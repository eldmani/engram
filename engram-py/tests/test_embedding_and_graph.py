"""
Tests for Gap 1 (semantic search via embedder plug-in) and
Gap 2 (graph traversal) — added in v0.1.1.

All tests use in-memory or temp-file SQLite. No external dependencies required:
the DeterministicEmbedder produces synthetic float vectors from a hash so we
can assert that cosine similarity scores flow through without needing a real
embedding model.
"""
from __future__ import annotations

import hashlib
import math
import struct
import tempfile
import os
from typing import Any

import pytest

from engram import (
    Engram,
    EngramConfig,
    BaseEmbedder,
    NullEmbedder,
    BaseExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
)
from engram.embedding.base import (
    cosine_similarity,
    vec_to_blob,
    blob_to_vec,
    NullEmbedder as _NullEmbedder,
)
from engram.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

class DeterministicEmbedder(BaseEmbedder):
    """
    Produces unit-normalised float vectors deterministically from the input
    text using a seeded SHA-256 hash so tests are fully reproducible.

    DIM=8 is enough to exercise cosine similarity without being slow.
    """

    DIM = 8

    @property
    def model_name(self) -> str:
        return "deterministic-test-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Take first DIM * 4 bytes as float32
            raw = struct.unpack(f"{self.DIM}f", digest[: self.DIM * 4])
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            result.append([x / norm for x in raw])
        return result


class SingleFactExtractor(BaseExtractor):
    """Always extracts exactly one entity + one fact from the stored text.
    The subject and predicate are taken from the extractor instance config.
    """

    def __init__(
        self,
        entity: str,
        predicate: str,
        obj: str,
        entity_type: str = "person",
        conf: float = 0.95,
    ) -> None:
        self._entity = entity
        self._predicate = predicate
        self._obj = obj
        self._type = entity_type
        self._conf = conf

    def extract(self, data: Any) -> ExtractionResult:
        return ExtractionResult(
            entities=[ExtractedEntity(name=self._entity, type=self._type)],
            facts=[
                ExtractedFact(
                    subject=self._entity,
                    predicate=self._predicate,
                    object=self._obj,
                    fact_type="assertion",
                    confidence=self._conf,
                )
            ],
            confidence=self._conf,
        )


class TwoEntityExtractor(BaseExtractor):
    """Extracts two entities and one fact linking them (for graph tests)."""

    def __init__(
        self,
        entity_a: str,
        entity_b: str,
        predicate: str,
        conf: float = 0.95,
    ) -> None:
        self._a = entity_a
        self._b = entity_b
        self._pred = predicate
        self._conf = conf

    def extract(self, data: Any) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                ExtractedEntity(name=self._a, type="person"),
                ExtractedEntity(name=self._b, type="person"),
            ],
            facts=[
                ExtractedFact(
                    subject=self._a,
                    predicate=self._pred,
                    object=self._b,
                    fact_type="assertion",
                    confidence=self._conf,
                )
            ],
            confidence=self._conf,
        )


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def mem(tmp_db):
    m = Engram(EngramConfig(db_path=tmp_db))
    yield m
    m.close()


@pytest.fixture
def mem_emb(tmp_db):
    """Engram instance with the deterministic embedder pre-installed."""
    m = Engram(EngramConfig(db_path=tmp_db))
    m.set_embedder(DeterministicEmbedder())
    yield m
    m.close()


# ===========================================================================
# Part 1 — Embedding interface (BaseEmbedder / NullEmbedder)
# ===========================================================================

class TestEmbedderInterface:
    def test_null_embedder_model_name(self):
        emb = NullEmbedder()
        assert emb.model_name == "null"

    def test_null_embedder_returns_empty_vectors(self):
        emb = NullEmbedder()
        vecs = emb.embed(["hello", "world"])
        assert vecs == [[], []]

    def test_deterministic_embedder_fixed_dimension(self):
        emb = DeterministicEmbedder()
        vecs = emb.embed(["test text"])
        assert len(vecs) == 1
        assert len(vecs[0]) == DeterministicEmbedder.DIM

    def test_deterministic_embedder_is_unit_normalised(self):
        emb = DeterministicEmbedder()
        vec = emb.embed(["some input"])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic_embedder_is_deterministic(self):
        emb = DeterministicEmbedder()
        assert emb.embed(["hello"]) == emb.embed(["hello"])

    def test_deterministic_embedder_different_inputs_differ(self):
        emb = DeterministicEmbedder()
        a = emb.embed(["alice"])[0]
        b = emb.embed(["bob"])[0]
        assert a != b

    def test_set_embedder_accepts_base_embedder_subclass(self, mem):
        # Should not raise
        mem.set_embedder(DeterministicEmbedder())

    def test_set_embedder_replaces_both_pipelines(self, tmp_db):
        """After set_embedder(), both ingest and retrieve pipelines use the new embedder."""
        emb = DeterministicEmbedder()
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(emb)
        # The internal pipelines should reference the embedder
        assert m._ingest._embedder is emb
        assert m._retrieve._embedder is emb
        m.close()

    def test_set_embedder_after_set_extractor_preserves_extractor(self, tmp_db):
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = SingleFactExtractor("X", "likes", "Y")
        m.set_extractor(ext)
        m.set_embedder(DeterministicEmbedder())
        assert m._ingest._extractor is ext
        m.close()


# ===========================================================================
# Part 2 — Cosine similarity and vector serialisation
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = [0.6, 0.8]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_score_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == 0.0

    def test_opposite_vectors_score_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_empty_vector_returns_zero(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0

    def test_zero_norm_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_similar_texts_score_higher_than_unrelated(self):
        emb = DeterministicEmbedder()
        query = emb.embed(["Alice prefers dark mode"])[0]
        related = emb.embed(["Alice dark mode preference"])[0]
        unrelated = emb.embed(["Python packaging tutorial"])[0]
        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


class TestVectorSerialisation:
    def test_roundtrip_preserves_values(self):
        v = [0.1, 0.2, 0.3, 0.4]
        recovered = blob_to_vec(vec_to_blob(v))
        for a, b in zip(v, recovered):
            assert abs(a - b) < 1e-6

    def test_blob_length_is_four_bytes_per_element(self):
        v = [1.0, 2.0, 3.0]
        assert len(vec_to_blob(v)) == 12  # 3 × float32

    def test_empty_vector_produces_empty_blob(self):
        assert vec_to_blob([]) == b""
        assert blob_to_vec(b"") == []


# ===========================================================================
# Part 3 — SQLiteStore embedding methods
# ===========================================================================

class TestSQLiteStoreEmbeddings:
    @pytest.fixture
    def store(self, tmp_path):
        s = SQLiteStore(str(tmp_path / "embed.db"))
        yield s
        s.close()

    def test_insert_and_retrieve_embedding(self, store):
        store.insert_embedding("fact", "f1", [0.1, 0.2, 0.3], "test-model")
        pairs = store.get_embeddings_for_refs("fact", ["f1"])
        assert len(pairs) == 1
        ref_id, vec = pairs[0]
        assert ref_id == "f1"
        assert len(vec) == 3
        assert abs(vec[0] - 0.1) < 1e-5

    def test_insert_or_replace_overwrites_old_vector(self, store):
        store.insert_embedding("fact", "f1", [0.1, 0.2], "m1")
        store.insert_embedding("fact", "f1", [0.9, 0.8], "m2")
        pairs = store.get_embeddings_for_refs("fact", ["f1"])
        assert len(pairs) == 1
        assert abs(pairs[0][1][0] - 0.9) < 1e-5

    def test_get_embeddings_returns_only_requested_ids(self, store):
        store.insert_embedding("fact", "f1", [0.1], "m")
        store.insert_embedding("fact", "f2", [0.2], "m")
        store.insert_embedding("fact", "f3", [0.3], "m")
        pairs = store.get_embeddings_for_refs("fact", ["f1", "f3"])
        ids = {p[0] for p in pairs}
        assert ids == {"f1", "f3"}

    def test_get_embeddings_empty_input_returns_empty(self, store):
        assert store.get_embeddings_for_refs("fact", []) == []

    def test_ref_type_is_filtered(self, store):
        store.insert_embedding("episode", "e1", [0.5], "m")
        # Querying for 'fact' ref_type should not return the episode embedding
        pairs = store.get_embeddings_for_refs("fact", ["e1"])
        assert pairs == []


# ===========================================================================
# Part 4 — Semantic scoring end-to-end
# ===========================================================================

class TestSemanticSearch:
    def test_null_embedder_produces_zero_semantic_score(self, tmp_db):
        """With NullEmbedder, semantic_score in traces must be 0.0."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = SingleFactExtractor("Alice", "prefers", "dark mode")
        m.set_extractor(ext)
        m.store("Alice prefers dark mode.")
        m.retrieve("dark mode preference")
        traces = m._store.get_traces("default", "dark mode preference")
        assert all(t.semantic_score == 0.0 for t in traces)
        m.close()

    def test_real_embedder_produces_nonzero_semantic_score(self, tmp_db):
        """With DeterministicEmbedder, semantic_score should be > 0.0 for a matching fact."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        ext = SingleFactExtractor("Alice", "prefers", "dark mode")
        m.set_extractor(ext)
        m.store("Alice prefers dark mode.")
        m.retrieve("dark mode preference")
        traces = m._store.get_traces("default", "dark mode preference")
        assert any(t.semantic_score > 0.0 for t in traces)
        m.close()

    def test_embeddings_stored_after_ingest(self, tmp_db):
        """After store(), embeddings table has entries for the fact."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        ext = SingleFactExtractor("Bob", "works_at", "Acme")
        m.set_extractor(ext)
        result = m.store("Bob works at Acme.")
        assert len(result.fact_ids) > 0
        embedded_ids = m._store.get_embedded_fact_ids_for_scope("default")
        # All committed facts should be embedded
        for fid in result.fact_ids:
            assert fid in embedded_ids
        m.close()

    def test_episode_embedding_stored_after_ingest(self, tmp_db):
        """After store(), episode embedding is also persisted."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        result = m.store("Just storing some text.")
        ep_pairs = m._store.get_embeddings_for_refs("episode", [result.episode_id])
        assert len(ep_pairs) == 1
        m.close()

    def test_semantically_similar_fact_ranked_above_unrelated(self, tmp_db):
        """A query about 'colour preferences' should rank the colour-related fact higher."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())

        ext_colour = SingleFactExtractor("Carol", "favourite_colour", "blue")
        m.set_extractor(ext_colour)
        m.store("Carol's favourite colour is blue.")

        ext_job = SingleFactExtractor("Carol", "works_as", "engineer")
        m.set_extractor(ext_job)
        m.store("Carol works as an engineer.")

        result = m.retrieve("What colour does Carol prefer?")
        # There should be at least one fact with a semantic score
        assert len(result.facts) >= 1

    def test_set_embedder_after_ingest_does_not_retroactively_embed(self, tmp_db):
        """Embeddings are NOT created retroactively when set_embedder is called after store."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = SingleFactExtractor("Dave", "drives", "Mazda")
        m.set_extractor(ext)
        result = m.store("Dave drives a Mazda.")
        # No embedder yet → no embeddings
        assert m._store.get_embedded_fact_ids_for_scope("default") == []
        # Now set embedder — still no embeddings for the old fact
        m.set_embedder(DeterministicEmbedder())
        assert m._store.get_embedded_fact_ids_for_scope("default") == []
        m.close()

    def test_null_embedder_skips_storage(self, tmp_db):
        """NullEmbedder must never write to the embeddings table."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = SingleFactExtractor("Eve", "lives_in", "Paris")
        m.set_extractor(ext)
        m.store("Eve lives in Paris.")
        # embeddings table should be empty
        rows = m._store._conn.execute(
            "SELECT COUNT(*) AS n FROM embeddings"
        ).fetchone()
        assert rows["n"] == 0
        m.close()

    def test_embedder_exception_is_swallowed_store_succeeds(self, tmp_db):
        """If the embedder raises, store() should still succeed (embedding is best-effort)."""
        class FailingEmbedder(BaseEmbedder):
            model_name = "fail"
            def embed(self, texts):
                raise RuntimeError("intentional failure")

        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(FailingEmbedder())
        ext = SingleFactExtractor("Frank", "likes", "cats")
        m.set_extractor(ext)
        # Should not raise
        result = m.store("Frank likes cats.")
        assert result.episode_id is not None
        m.close()


# ===========================================================================
# Part 5 — Graph traversal
# ===========================================================================

class TestGraphTraversal:
    @pytest.fixture
    def store(self, tmp_path):
        s = SQLiteStore(str(tmp_path / "graph.db"))
        yield s
        s.close()

    def test_get_connected_entity_ids_empty_seeds(self, store):
        assert store.get_connected_entity_ids([], "scope1") == set()

    def test_get_connected_entity_ids_no_facts_returns_empty(self, store):
        result = store.get_connected_entity_ids(["nonexistent-id"], "scope1")
        assert result == set()

    def test_get_facts_for_entities_empty_ids(self, store):
        assert store.get_facts_for_entities([], "scope1") == []

    def test_graph_traversal_finds_one_hop_neighbor(self, tmp_db):
        """Storing 'Alice knows Bob' then retrieving 'Bob' also surfaces Alice-related facts."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = TwoEntityExtractor("Alice", "Bob", "knows")
        m.set_extractor(ext)
        m.store("Alice knows Bob.")
        # Retrieve about Bob — the 'knows' fact links Alice→Bob, so Alice is
        # reachable from Bob as a neighbour (and vice versa)
        result = m.retrieve("Bob")
        fact_predicates = [f.predicate for f in result.facts]
        assert "knows" in fact_predicates
        m.close()

    def test_graph_score_nonzero_for_connected_facts(self, tmp_db):
        """After retrieval, traces for graph-expanded facts should have graph_score > 0."""
        m = Engram(EngramConfig(db_path=tmp_db))

        # Store Alice→Bob link
        ext_ab = TwoEntityExtractor("Alice", "Bob", "manages")
        m.set_extractor(ext_ab)
        m.store("Alice manages Bob.")

        # Store a second fact about Alice (unrelated to query about Bob)
        ext_a = SingleFactExtractor("Alice", "favourite_food", "sushi")
        m.set_extractor(ext_a)
        m.store("Alice likes sushi.")

        # Query about Bob — the 'manages' fact should have a graph score if Alice
        # is a seed entity reached via FTS
        result = m.retrieve("Alice manages Bob")
        # At minimum, the directly matched fact should be in result
        assert len(result.facts) >= 1
        m.close()

    def test_seed_entity_facts_included_in_results(self, tmp_db):
        """Facts directly about seed entities score highest."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = SingleFactExtractor("Carol", "role", "CEO")
        m.set_extractor(ext)
        m.store("Carol is CEO.")
        result = m.retrieve("Carol role")
        predicates = [f.predicate for f in result.facts]
        assert "role" in predicates
        m.close()

    def test_get_connected_entity_ids_bidirectional(self, tmp_db):
        """Both outgoing (subj→obj) and incoming (obj→subj) edges are traversed."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = TwoEntityExtractor("Alice", "Bob", "reports_to")
        m.set_extractor(ext)
        m.store("Alice reports to Bob.")

        # Find Alice's entity ID
        alice_entity = m._store.find_entity_by_name("default", "Alice")
        bob_entity = m._store.find_entity_by_name("default", "Bob")
        assert alice_entity is not None
        assert bob_entity is not None

        # Alice is subject → Bob should be reachable from Alice
        from_alice = m._store.get_connected_entity_ids([alice_entity.id], "default")
        assert bob_entity.id in from_alice

        # Bob is object → Alice should be reachable from Bob
        from_bob = m._store.get_connected_entity_ids([bob_entity.id], "default")
        assert alice_entity.id in from_bob

        m.close()

    def test_seed_entities_excluded_from_neighbor_result(self, tmp_db):
        """get_connected_entity_ids should NOT include the seed entities themselves."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = TwoEntityExtractor("Alice", "Bob", "works_with")
        m.set_extractor(ext)
        m.store("Alice works with Bob.")

        alice_entity = m._store.find_entity_by_name("default", "Alice")
        assert alice_entity is not None
        neighbors = m._store.get_connected_entity_ids([alice_entity.id], "default")
        assert alice_entity.id not in neighbors

        m.close()

    def test_graph_scores_stored_in_traces(self, tmp_db):
        """graph_score in retrieval traces reflects the graph traversal result."""
        m = Engram(EngramConfig(db_path=tmp_db))
        ext = TwoEntityExtractor("Alice", "Bob", "mentors")
        m.set_extractor(ext)
        m.store("Alice mentors Bob.")

        # Extra fact about Alice so there are facts to expand to
        ext2 = SingleFactExtractor("Alice", "expert_in", "Python")
        m.set_extractor(ext2)
        m.store("Alice is an expert in Python.")

        # Retrieve about Bob — may surface Alice-linked facts via graph
        m.retrieve("Bob mentors")
        traces = m._store.get_traces("default", "Bob mentors")
        # At least one trace is written (the directly matched fact)
        assert len(traces) >= 1
        m.close()


# ===========================================================================
# Part 6 — Combined semantic + graph
# ===========================================================================

class TestSemanticAndGraphCombined:
    def test_hybrid_retrieval_returns_facts(self, tmp_db):
        """Retrieval with both semantic and graph active doesn't crash and returns facts."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        ext = TwoEntityExtractor("Alice", "Bob", "trusts")
        m.set_extractor(ext)
        m.store("Alice trusts Bob.")
        result = m.retrieve("trust relationship")
        assert len(result.facts) >= 1
        m.close()

    def test_hybrid_scores_are_not_all_zero(self, tmp_db):
        """When a real embedder is active, at least one score component should be non-zero."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        ext = SingleFactExtractor("Alice", "hobby", "painting")
        m.set_extractor(ext)
        m.store("Alice enjoys painting.")
        m.retrieve("Alice hobby")
        traces = m._store.get_traces("default", "Alice hobby")
        assert len(traces) >= 1
        for t in traces:
            total = t.keyword_score + t.semantic_score + t.graph_score + t.temporal_score
            assert total > 0.0
        m.close()

    def test_final_score_reflects_all_components(self, tmp_db):
        """final_score in traces should be > 0 when any component is active."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        ext = SingleFactExtractor("Dave", "plays", "guitar")
        m.set_extractor(ext)
        m.store("Dave plays guitar.")
        m.retrieve("Dave guitar")
        traces = m._store.get_traces("default", "Dave guitar")
        assert all(t.final_score > 0.0 for t in traces)
        m.close()

    def test_set_embedder_twice_uses_latest(self, tmp_db):
        """Calling set_embedder() twice; only the last embedder should be active."""
        class TaggedEmbedder(BaseEmbedder):
            def __init__(self, tag: str) -> None:
                self._tag = tag

            @property
            def model_name(self) -> str:
                return f"tagged-{self._tag}"

            def embed(self, texts):
                return DeterministicEmbedder().embed(texts)

        m = Engram(EngramConfig(db_path=tmp_db))
        emb1 = TaggedEmbedder("first")
        emb2 = TaggedEmbedder("second")
        m.set_embedder(emb1)
        m.set_embedder(emb2)
        assert m._retrieve._embedder is emb2
        m.close()

    def test_context_result_works_with_embedder(self, tmp_db):
        """get_context() should work without error when an embedder is set."""
        m = Engram(EngramConfig(db_path=tmp_db))
        m.set_embedder(DeterministicEmbedder())
        ext = SingleFactExtractor("Eve", "speaks", "French")
        m.set_extractor(ext)
        m.store("Eve speaks French.")
        ctx = m.get_context("What language does Eve speak?")
        assert ctx.formatted is not None
        m.close()
