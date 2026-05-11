import * as os from 'node:os';
import * as path from 'node:path';
import * as fs from 'node:fs';
import {
  Engram,
  BaseExtractor,
  type EngramConfig,
  type ExtractionResult,
  type ExtractedEntity,
  type ExtractedFact,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tmpEngram(overrides?: Partial<EngramConfig>): Engram {
  const dbPath = path.join(os.tmpdir(), `engram-test-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
  const mem = new Engram({ dbPath, ...overrides });
  return mem;
}

class FixedExtractor extends BaseExtractor {
  constructor(
    private entities: ExtractedEntity[],
    private facts: ExtractedFact[],
    private confidence = 0.95,
  ) {
    super();
  }

  extract(_data: string | Record<string, unknown>): ExtractionResult {
    return { entities: this.entities, facts: this.facts, confidence: this.confidence };
  }
}

// ---------------------------------------------------------------------------
// Basic store / retrieve
// ---------------------------------------------------------------------------

describe('TestBasicStoreRetrieve', () => {
  let mem: Engram;

  beforeEach(() => { mem = tmpEngram(); });
  afterEach(() => mem.close());

  it('store() returns episode_id', () => {
    const result = mem.store('Alice prefers dark mode.');
    expect(result.episodeId).toBeTruthy();
    expect(result.factIds).toHaveLength(0);
    expect(result.entityIds).toHaveLength(0);
    expect(result.quarantinedCount).toBe(0);
  });

  it('store() accepts dict', () => {
    const result = mem.store({ key: 'value' });
    expect(result.episodeId).toBeTruthy();
  });

  it('retrieve() returns episode', () => {
    mem.store('Alice prefers dark mode.');
    const result = mem.retrieve('dark mode');
    expect(result.episodes.length).toBeGreaterThanOrEqual(1);
    expect(result.episodes.some(ep => ep.rawText.includes('dark mode'))).toBe(true);
  });

  it('retrieve() on empty scope returns empty', () => {
    const result = mem.retrieve('anything', undefined, 'empty-scope');
    expect(result.facts).toHaveLength(0);
    expect(result.episodes).toHaveLength(0);
  });

  it('getContext() returns formatted string', () => {
    mem.store('Alice prefers dark mode and vim keybindings.');
    const ctx = mem.getContext('what does Alice prefer?');
    expect(ctx.formatted).toBeTruthy();
    expect(ctx.formatted).toMatch(/EPISODES|FACTS/);
  });

  it('scope isolation', () => {
    mem.store('Alice likes cats.', {}, 'user-1');
    mem.store('Bob likes dogs.', {}, 'user-2');
    const r1 = mem.retrieve('likes', undefined, 'user-1');
    const r2 = mem.retrieve('likes', undefined, 'user-2');
    expect(r1.episodes.every(ep => ep.rawText.includes('cats'))).toBe(true);
    expect(r2.episodes.every(ep => ep.rawText.includes('dogs'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Extractor + entity resolution
// ---------------------------------------------------------------------------

describe('TestWithExtractor', () => {
  it('store() creates entities and facts', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.92 }],
    ));
    const result = mem.store('Alice prefers dark mode.');
    expect(result.entityIds).toHaveLength(1);
    expect(result.factIds).toHaveLength(1);
    expect(result.quarantinedCount).toBe(0);
    mem.close();
  });

  it('retrieve() returns facts', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.92 }],
    ));
    mem.store('Alice prefers dark mode.');
    const result = mem.retrieve('Alice preference');
    expect(result.facts.length).toBeGreaterThanOrEqual(1);
    expect(result.facts[0].predicate).toBe('prefers');
    mem.close();
  });

  it('entity deduplication on second store', () => {
    const mem = tmpEngram();
    const ext = new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'likes', object: 'coffee', confidence: 0.90 }],
    );
    mem.setExtractor(ext);
    mem.store('Alice likes coffee.');
    mem.store('Alice likes coffee.');
    const entities = mem._store.getEntitiesForScope('default');
    expect(entities).toHaveLength(1);
    mem.close();
  });

  it('temporal supersession', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Adidas', confidence: 0.92 }],
    ));
    mem.store('Alice prefers Adidas.');

    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Nike', confidence: 0.92 }],
    ));
    mem.store('Alice prefers Nike now.');

    const result = mem.retrieve('Alice preference');
    const current = result.facts.filter(f => f.truthState === 'current');
    expect(current).toHaveLength(1);
    expect((current[0].objectValueJson as { value: string }).value).toBe('Nike');
    mem.close();
  });
});

// ---------------------------------------------------------------------------
// Confidence gate / quarantine
// ---------------------------------------------------------------------------

describe('TestConfidenceGate', () => {
  it('low confidence quarantines fact', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.50 }],
      0.50, // below extractorConfidenceMin=0.70
    ));
    const result = mem.store('Alice maybe prefers dark mode.');
    expect(result.quarantinedCount).toBe(1);
    expect(result.factIds).toHaveLength(0);
    mem.close();
  });

  it('low supersession confidence quarantines', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Adidas', confidence: 0.95 }],
      0.95,
    ));
    mem.store('Alice prefers Adidas.');

    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Nike', confidence: 0.80 }],
      0.80, // below supersessionConfidenceMin=0.85
    ));
    const result = mem.store('Alice might prefer Nike.');
    expect(result.quarantinedCount).toBeGreaterThanOrEqual(1);

    // Original should still be current
    const entities = mem._store.getEntitiesForScope('default');
    const alice = entities[0];
    const current = mem._store.getCurrentFacts('default', alice.id, 'prefers');
    expect(current).toHaveLength(1);
    expect((current[0].objectValueJson as { value: string }).value).toBe('Adidas');
    mem.close();
  });

  it('explain() surfaces quarantined facts', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.50 }],
      0.50,
    ));
    mem.store('Alice maybe prefers dark mode.');
    const expl = mem.explain('Alice preference');
    expect(expl.quarantinedFacts.length).toBeGreaterThanOrEqual(1);
    expect(expl.summary.toLowerCase()).toMatch(/quarantine/);
    mem.close();
  });
});

// ---------------------------------------------------------------------------
// updateFact
// ---------------------------------------------------------------------------

describe('TestUpdateFact', () => {
  it('creates new version, old becomes superseded', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Adidas', confidence: 0.95 }],
      0.95,
    ));
    const result = mem.store('Alice prefers Adidas.');
    const factId = result.factIds[0];
    const newFact = mem.updateFact(factId, 'Nike');
    expect(newFact.truthState).toBe('current');
    expect((newFact.objectValueJson as { value: string }).value).toBe('Nike');
    const old = mem._store.getFact(factId);
    expect(old?.truthState).toBe('superseded');
    expect(old?.validTo).not.toBeNull();
    mem.close();
  });

  it('throws on nonexistent fact', () => {
    const mem = tmpEngram();
    expect(() => mem.updateFact('nonexistent-id', 'value')).toThrow();
    mem.close();
  });
});

// ---------------------------------------------------------------------------
// forget
// ---------------------------------------------------------------------------

describe('TestForget', () => {
  it('soft forget hides fact', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.95 }],
    ));
    const result = mem.store('Alice prefers dark mode.');
    const factId = result.factIds[0];
    mem.forget({ factId }, false);
    const fact = mem._store.getFact(factId);
    expect(fact?.truthState).toBe('hidden');
    mem.close();
  });

  it('hard forget deletes fact', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.95 }],
    ));
    const result = mem.store('Alice prefers dark mode.');
    const factId = result.factIds[0];
    mem.forget({ factId }, true);
    expect(mem._store.getFact(factId)).toBeNull();
    mem.close();
  });

  it('throws on invalid selector', () => {
    const mem = tmpEngram();
    expect(() => mem.forget({})).toThrow();
    mem.close();
  });
});

// ---------------------------------------------------------------------------
// Background workers
// ---------------------------------------------------------------------------

describe('TestWorkers', () => {
  it('runDecay runs without error', () => {
    const mem = tmpEngram();
    mem.store('Alice prefers dark mode.');
    expect(() => mem.workers.runDecay('default')).not.toThrow();
    mem.close();
  });

  it('runReconsolidation runs without error', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.50 }],
      0.50,
    ));
    mem.store('Alice maybe prefers dark mode.');
    expect(() => mem.workers.runReconsolidation('default')).not.toThrow();
    mem.close();
  });
});
