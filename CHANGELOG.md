# Changelog

All notable changes to engram are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.1.0] - 2025-01-01

### Added

**Core**
- `SQLiteStore` — WAL-mode SQLite backend with FTS5 full-text search, temporal
  fact validity windows, and confidence-gated quarantine.
- `IngestionPipeline` — 11-step ingestion: episode creation → entity resolution
  → alias normalisation → confidence gate → fact supersession → quarantine.
- `RetrievalPipeline` — Hybrid scoring (keyword + graph + temporal + salience +
  strength). Semantic scoring is reserved for a future embedder plug-in.
- `BackgroundWorkers` — Decay, heat promotion, and reconsolidation passes.
- `Engram` — High-level public API: `store()`, `retrieve()`, `get_context()`,
  `update_fact()`, `forget()`, `explain()`.

**Python package** (`engram-py`)
- Python ≥ 3.10, stdlib-only core; optional extras `graph` (networkx) and
  `vector` (numpy).
- PEP 561 `py.typed` marker included.
- Thread safety via `threading.local()` per-thread connections.

**Node.js package** (`engram-node`)
- Node.js ≥ 23.0.0, zero npm dependencies.
- Uses built-in `node:sqlite` (unflagged since Node 23).
- TypeScript-first with full type declarations.

**Safety**
- Three-tier confidence gate (extractor / resolution / supersession thresholds).
- Failed gate writes `QuarantinedFact` — never silently corrupts history.
- FTS query sanitisation prevents `OperationalError` on malformed input.
- `hide_scope_facts` / `delete_scope_facts` scope-level forget operations.

**Developer experience**
- 20 Python tests, 20 Node.js tests (all green).
- Schema version pragma (`user_version`) for future migration support.
- `CONTRIBUTING.md` and this `CHANGELOG.md`.

[0.1.0]: https://github.com/your-org/engram/releases/tag/v0.1.0
