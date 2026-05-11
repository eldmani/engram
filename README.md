# engram

Model-agnostic temporal memory store for AI agents.

**Core principle:** truth is temporal · importance is dynamic · relevance is query-time

## What is engram?

engram is a library that gives AI agents durable, queryable memory. It stores
conversations and facts in a temporal knowledge graph backed by SQLite. Any model
can write to it and read from it via a simple `store()` / `retrieve()` API.

Key properties:
- **Temporal facts** — every fact has `valid_from` / `valid_to`. Old facts are preserved, not deleted.
- **Confidence gate** — uncertain extractions are quarantined, never silently corrupting history.
- **Hybrid retrieval** — keyword (FTS), semantic (pluggable embedder), graph traversal, and temporal filters fused into one score.
- **Memory dynamics** — salience, reinforcement on access, power-law strength decay, heat-based promotion.
- **Explainability** — every retrieval result traces back to source episodes.
- **Model-agnostic** — plug in any LLM extractor; the storage layer never calls an LLM directly.

## Packages

| Package | Language | Path |
|---|---|---|
| `engram-memory` | Python ≥ 3.10 | [`engram-py/`](engram-py/) |
| `engram-memory` | Node.js / TypeScript | [`engram-node/`](engram-node/) |

Both packages share the same SQLite schema — a `.db` file written by Python is
readable by the Node.js library and vice versa.

## Python quickstart

```bash
cd engram-py
pip install -e .
```

```python
from engram import Engram

mem = Engram()  # defaults to engram.db in cwd

mem.store("Alice prefers dark mode and uses vim keybindings.")
mem.store("Alice is working on a project called Nightwatch.")

result = mem.get_context("what does Alice prefer?")
print(result.formatted)
# [FACTS]
# (raw episodes — plug in an extractor for structured facts)
# [EPISODES]
# - Alice prefers dark mode and uses vim keybindings.
# [EVIDENCE]
# - Derived from episodes ep_...
```

### Plug in a real extractor

```python
from engram import Engram
from engram.extraction import BaseExtractor, ExtractionResult, ExtractedEntity, ExtractedFact

class MyLLMExtractor(BaseExtractor):
    def extract(self, data: str | dict) -> ExtractionResult:
        # Call your LLM here, parse entities and facts
        return ExtractionResult(
            entities=[ExtractedEntity(name="Alice", type="person", confidence=0.95)],
            facts=[ExtractedFact(subject="Alice", predicate="prefers", object="dark mode", confidence=0.92)],
            confidence=0.92,
        )

mem = Engram()
mem.set_extractor(MyLLMExtractor())
mem.store("Alice prefers dark mode.")
```

## Node.js quickstart

```bash
cd engram-node
npm install
npm run build
```

```typescript
import { Engram } from 'engram-memory';

const mem = new Engram();

mem.store('Alice prefers dark mode and uses vim keybindings.');
const result = mem.getContext('what does Alice prefer?');
console.log(result.formatted);
```

## Architecture

See [engram_architecture_v1.txt](engram_architecture_v1.txt) for the full specification.

## License

Apache 2.0
