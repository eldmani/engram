# Contributing to engram

Thank you for your interest in contributing!

## Getting started

### Python

```bash
cd engram-py
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

### Node.js

```bash
cd engram-node
npm install
npm test
```

> **Requires Node ≥ 23.0.0** — `node:sqlite` is only unflagged on Node 23+.

---

## Development guidelines

### Code style

- **Python**: PEP 8 + `from __future__ import annotations`. Use built-in
  generics (`list[T]`, `tuple[A, B]`) instead of `typing.List`/`typing.Tuple`.
  All public functions should have docstrings.
- **Node.js**: TypeScript strict mode. No `any` outside of SQLite row types.
  Prefer `node:` protocol imports for built-ins.

### Adding a new feature

1. Open an issue describing the motivation.
2. Create a branch: `git checkout -b feat/my-feature`.
3. Write tests first — both Python and Node implementations must stay in parity.
4. Implement in both `engram-py/` and `engram-node/src/`.
5. Ensure all tests pass: `pytest` + `npm test`.
6. Open a pull request against `main`.

### Confidence gate

The three thresholds (`extractor_confidence_min`, `resolution_confidence_min`,
`supersession_confidence_min`) are intentionally conservative. If you lower
them, add tests that verify quarantine still fires for genuinely low-confidence
extractions.

### Semantic scoring

`weight_semantic` (default 0.40) is a **placeholder**. Semantic scores are
always `0.0` until an embedder is wired in via `set_extractor()` /
`setExtractor()`. If you add embedder support, document the expected interface
in `extraction/base.py` and `src/extraction/base.ts`.

### Schema changes

Bump `_SCHEMA_VERSION` / `SCHEMA_VERSION` and add a migration helper in
`sqlite_store.py` / `sqliteStore.ts`. The schema uses `PRAGMA user_version`
to track the current version.

---

## Reporting issues

Please include:
- Python / Node.js version
- OS
- Minimal reproduction script
- Full traceback or error output

---

## License

By contributing you agree that your contributions will be licensed under the
[Apache 2.0 License](LICENSE).
