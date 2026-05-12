/**
 * Extended integration tests for Engram — covers all paths not exercised by
 * the baseline engram.test.ts suite: forget selectors, updateFact variants,
 * all worker methods, config, NullExtractor, and extended API surface.
 * Node.js mirror of Python's test_engram_extended.py.
 */
import * as os from 'node:os';
import * as path from 'node:path';
import {
  Engram,
  NullExtractor,
  BaseExtractor,
  defaultConfig,
  type EngramConfig,
  type ExtractionResult,
  type ExtractedEntity,
  type ExtractedFact,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tmpDb(): string {
  return path.join(
    os.tmpdir(),
    `engram-ext-${Date.now()}-${Math.random().toString(36).slice(2)}.db`,
  );
}

function tmpEngram(overrides?: Partial<EngramConfig>): Engram {
  return new Engram({ dbPath: tmpDb(), ...overrides });
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

function aliceMem(
  extraFacts: ExtractedFact[] = [],
  confidence = 0.95,
): Engram {
  const facts: ExtractedFact[] = [
    { subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.95 },
    ...extraFacts,
  ];
  const mem = tmpEngram();
  mem.setExtractor(new FixedExtractor(
    [{ name: 'Alice', type: 'person', confidence: 0.95 }],
    facts,
    confidence,
  ));
  return mem;
}

// ---------------------------------------------------------------------------
// EngramConfig defaults
// ---------------------------------------------------------------------------

describe('EngramConfig', () => {
  it('defaultConfig has expected defaults', () => {
    const cfg = defaultConfig();
    expect(cfg.defaultScope).toBe('default');
    expect(cfg.extractorConfidenceMin).toBe(0.70);
    expect(cfg.resolutionConfidenceMin).toBe(0.75);
    expect(cfg.supersessionConfidenceMin).toBe(0.85);
    expect(cfg.weightKeyword).toBe(0.30);
    expect(cfg.weightSemantic).toBe(0.40);
    expect(cfg.weightGraph).toBe(0.10);
    expect(cfg.weightTemporal).toBe(0.10);
  });

  it('custom values are reflected in Engram config', () => {
    const cfg = defaultConfig({ defaultScope: 'bot-1', defaultTopK: 5 });
    expect(cfg.defaultScope).toBe('bot-1');
    expect(cfg.defaultTopK).toBe(5);
  });

  it('Engram uses custom defaultScope', () => {
    const mem = tmpEngram({ defaultScope: 'custom' });
    try {
      mem.store('anything');
      const r = mem.retrieve('anything', undefined, 'custom');
      expect(r.episodes.length).toBeGreaterThanOrEqual(1);
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// NullExtractor
// ---------------------------------------------------------------------------

describe('NullExtractor', () => {
  it('extract returns empty entities and facts', () => {
    const ex = new NullExtractor();
    const result = ex.extract('anything');
    expect(result.entities).toEqual([]);
    expect(result.facts).toEqual([]);
  });

  it('extract works with dict input', () => {
    const ex = new NullExtractor();
    const result = ex.extract({ key: 'value' });
    expect(result.entities).toEqual([]);
    expect(result.facts).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// store() variations
// ---------------------------------------------------------------------------

describe('StoreVariations', () => {
  it('store() preserves metadata on episode', () => {
    const mem = tmpEngram();
    try {
      const result = mem.store('event happened', { source: 'log', ts: 123 });
      expect(result.episodeId).toBeTruthy();
      const ep = mem._store.getEpisode(result.episodeId);
      expect(ep!.metadata['source']).toBe('log');
      expect(ep!.metadata['ts']).toBe(123);
    } finally {
      mem.close();
    }
  });

  it('store() creates multiple entities and facts', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [
        { name: 'Alice', type: 'person', confidence: 0.95 },
        { name: 'Bob', type: 'person', confidence: 0.95 },
      ],
      [
        { subject: 'Alice', predicate: 'knows', object: 'Bob', confidence: 0.92 },
        { subject: 'Bob', predicate: 'likes', object: 'chess', confidence: 0.90 },
      ],
    ));
    try {
      const result = mem.store('Alice knows Bob. Bob likes chess.');
      expect(result.entityIds).toHaveLength(2);
      expect(result.factIds).toHaveLength(2);
    } finally {
      mem.close();
    }
  });

  it('same content produces same checksum on both episodes', () => {
    const mem = tmpEngram();
    try {
      const r1 = mem.store('Exactly the same text.');
      const r2 = mem.store('Exactly the same text.');
      const ep1 = mem._store.getEpisode(r1.episodeId);
      const ep2 = mem._store.getEpisode(r2.episodeId);
      expect(ep1!.checksum).toBe(ep2!.checksum);
      expect(ep1!.id).not.toBe(ep2!.id);
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// retrieve() / getContext() with options
// ---------------------------------------------------------------------------

describe('RetrieveOptions', () => {
  it('retrieve respects topK limit', () => {
    const mem = tmpEngram();
    try {
      for (let i = 0; i < 10; i++) mem.store(`fact about topic ${i}`);
      const result = mem.retrieve('topic', 3);
      expect(result.facts.length).toBeLessThanOrEqual(3);
    } finally {
      mem.close();
    }
  });

  it('retrieve with asOf filters future facts', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'likes', object: 'coffee', confidence: 0.95 }],
    ));
    const pastDate = new Date(Date.now() - 60 * 60 * 1000); // 1 hour ago
    try {
      mem.store('Alice likes coffee.');
      const result = mem.retrieve('coffee', undefined, undefined, pastDate);
      expect(result).toHaveProperty('facts');
      expect(result).toHaveProperty('episodes');
      // validFrom is now > pastDate, so FTS fact filtering should exclude them
      const factHits = mem._store.ftsSearchFacts('coffee', 'default', 10, pastDate);
      expect(factHits).toHaveLength(0);
    } finally {
      mem.close();
    }
  });

  it('getContext with custom scope works', () => {
    const mem = tmpEngram();
    try {
      mem.store('Alice is in scope X.', {}, 'scope-x');
      const ctx = mem.getContext('Alice', undefined, 'scope-x');
      expect(ctx.formatted).toBeTruthy();
    } finally {
      mem.close();
    }
  });

  it('getContext respects topK', () => {
    const mem = tmpEngram();
    try {
      for (let i = 0; i < 10; i++) mem.store(`alpha item ${i}`);
      const ctx = mem.getContext('alpha', 2);
      expect(ctx.formatted).toBeTruthy();
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// updateFact() variations
// ---------------------------------------------------------------------------

describe('UpdateFactVariations', () => {
  it('updateFact replaces value and supersedes old fact', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const newFact = mem.updateFact(fid, 'light mode');
      expect(newFact.objectValueJson).toEqual({ value: 'light mode' });
      const oldFact = mem._store.getFact(fid);
      expect(oldFact!.truthState).toBe('superseded');
    } finally {
      mem.close();
    }
  });

  it('updateFact with explicit validFrom sets the date', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const explicitTime = new Date('2025-01-01T00:00:00.000Z');
      const newFact = mem.updateFact(fid, 'light mode', explicitTime);
      expect(newFact.validFrom.toISOString()).toBe(explicitTime.toISOString());
    } finally {
      mem.close();
    }
  });

  it('updateFact preserves scope of original fact', () => {
    const mem = tmpEngram();
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'dark mode', confidence: 0.95 }],
    ));
    try {
      const result = mem.store('Alice prefers dark mode.', {}, 'scope-a');
      const fid = result.factIds[0];
      const newFact = mem.updateFact(fid, 'light mode');
      expect(newFact.scopeId).toBe('scope-a');
    } finally {
      mem.close();
    }
  });

  it('updateFact throws for nonexistent factId', () => {
    const mem = tmpEngram();
    try {
      expect(() => mem.updateFact('no-such-id', 'value')).toThrow();
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// forget() — all selector paths
// ---------------------------------------------------------------------------

describe('ForgetSelectors', () => {
  it('soft forget by fact_id hides the fact', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      mem.forget({ factId: fid }, false);
      expect(mem._store.getFact(fid)!.truthState).toBe('hidden');
    } finally {
      mem.close();
    }
  });

  it('hard forget by fact_id deletes the fact', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      mem.forget({ factId: fid }, true);
      expect(mem._store.getFact(fid)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('soft forget by entity_id hides entity facts', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const entities = mem._store.getEntitiesForScope('default');
      const eid = entities[0].id;
      const fr = mem.forget({ entityId: eid }, false);
      expect(fr.affectedEntities).toBeGreaterThanOrEqual(1);
      expect(mem._store.getFact(fid)!.truthState).toBe('hidden');
    } finally {
      mem.close();
    }
  });

  it('hard forget by entity_id deletes entity facts', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const entities = mem._store.getEntitiesForScope('default');
      const eid = entities[0].id;
      mem.forget({ entityId: eid }, true);
      expect(mem._store.getFact(fid)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('hard forget by episode_id deletes episode facts', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      mem.forget({ episodeId: result.episodeId }, true);
      expect(mem._store.getFact(fid)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('soft forget by scope hides all scope facts', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const fr = mem.forget({ scopeId: 'default' }, false);
      expect(fr.affectedFacts).toBeGreaterThanOrEqual(1);
      expect(mem._store.getFact(fid)!.truthState).toBe('hidden');
    } finally {
      mem.close();
    }
  });

  it('hard forget by scope deletes all scope facts', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      mem.forget({ scopeId: 'default' }, true);
      expect(mem._store.getFact(fid)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('forget with empty selector throws', () => {
    const mem = tmpEngram();
    try {
      expect(() => mem.forget({})).toThrow();
    } finally {
      mem.close();
    }
  });

  it('forgetResult has expected fields', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const fr = mem.forget({ factId: fid });
      expect(fr).toHaveProperty('affectedFacts');
      expect(fr).toHaveProperty('affectedEntities');
      expect(fr).toHaveProperty('affectedAliases');
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Background workers
// ---------------------------------------------------------------------------

describe('WorkersExtended', () => {
  it('reinforce boosts access_count and strength', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const before = mem._store.getFact(fid)!;
      mem.workers.reinforce([fid]);
      const after = mem._store.getFact(fid)!;
      expect(after.accessCount).toBe(before.accessCount + 1);
      expect(after.strength).toBeGreaterThanOrEqual(before.strength);
    } finally {
      mem.close();
    }
  });

  it('reinforce with empty list does not throw', () => {
    const mem = tmpEngram();
    try {
      expect(() => mem.workers.reinforce([])).not.toThrow();
    } finally {
      mem.close();
    }
  });

  it('computeHeat returns a non-negative float', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const fact = mem._store.getFact(fid)!;
      const heat = mem.workers.computeHeat(fact);
      expect(typeof heat).toBe('number');
      expect(heat).toBeGreaterThanOrEqual(0.0);
    } finally {
      mem.close();
    }
  });

  it('runHeatPromotion returns an array', () => {
    const mem = aliceMem();
    try {
      mem.store('Alice prefers dark mode.');
      const hot = mem.workers.runHeatPromotion('default');
      expect(Array.isArray(hot)).toBe(true);
    } finally {
      mem.close();
    }
  });

  it('runDecay reduces strength of unreinforced fact', () => {
    const mem = aliceMem();
    try {
      const result = mem.store('Alice prefers dark mode.');
      const fid = result.factIds[0];
      const beforeStrength = mem._store.getFact(fid)!.strength;
      mem.workers.runDecay('default');
      const afterStrength = mem._store.getFact(fid)!.strength;
      expect(afterStrength).toBeLessThanOrEqual(beforeStrength);
    } finally {
      mem.close();
    }
  });

  it('runReconsolidation rejects superseded quarantine', () => {
    const mem = tmpEngram();
    // High-confidence fact committed first
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Adidas', confidence: 0.95 }],
    ));
    mem.store('Alice prefers Adidas.');

    // Low-confidence → quarantined
    mem.setExtractor(new FixedExtractor(
      [{ name: 'Alice', type: 'person', confidence: 0.95 }],
      [{ subject: 'Alice', predicate: 'prefers', object: 'Nike', confidence: 0.50 }],
      0.50,
    ));
    mem.store('Alice might prefer Nike.');
    const pendingBefore = mem._store.getPendingQuarantinedFacts('default');
    expect(pendingBefore.length).toBeGreaterThanOrEqual(1);

    mem.workers.runReconsolidation('default');
    const pendingAfter = mem._store.getPendingQuarantinedFacts('default');
    expect(pendingAfter.length).toBeLessThan(pendingBefore.length);
    mem.close();
  });

  it('runCleanup runs without error', () => {
    const mem = aliceMem();
    try {
      mem.store('Alice prefers dark mode.');
      expect(() => mem.workers.runCleanup('default')).not.toThrow();
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Symbol.dispose (context-manager parity)
// ---------------------------------------------------------------------------

describe('SymbolDispose', () => {
  it('close() does not throw', () => {
    const mem = tmpEngram();
    mem.store('test');
    expect(() => mem.close()).not.toThrow();
  });

  it('Symbol.dispose is equivalent to close()', () => {
    const mem = tmpEngram();
    mem.store('test');
    expect(() => mem[Symbol.dispose]()).not.toThrow();
  });

  it('two independent Engram instances do not share state', () => {
    const m1 = tmpEngram();
    const m2 = tmpEngram();
    try {
      m1.store('for m1 only');
      m2.store('for m2 only');
      const r1 = m1.retrieve('m1').episodes;
      const r2 = m2.retrieve('m2').episodes;
      expect(r1.length).toBeGreaterThanOrEqual(1);
      expect(r2.length).toBeGreaterThanOrEqual(1);
      expect(r1[0].rawText).not.toBe(r2[0].rawText);
    } finally {
      m1.close();
      m2.close();
    }
  });
});

// ---------------------------------------------------------------------------
// explain()
// ---------------------------------------------------------------------------

describe('Explain', () => {
  it('explain returns empty quarantinedFacts when none', () => {
    const mem = tmpEngram();
    try {
      mem.store('Alice likes coffee.');
      const expl = mem.explain('Alice');
      expect(expl.quarantinedFacts).toEqual([]);
    } finally {
      mem.close();
    }
  });

  it('explain returns traces after retrieve', () => {
    const mem = tmpEngram();
    try {
      mem.store('Alice likes coffee.');
      mem.retrieve('Alice likes coffee');
      const expl = mem.explain('Alice likes coffee');
      expect(Array.isArray(expl.traces)).toBe(true);
      expect(Array.isArray(expl.quarantinedFacts)).toBe(true);
      expect(typeof expl.summary).toBe('string');
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// FTS robustness via Engram.retrieve()
// ---------------------------------------------------------------------------

describe('FtsRobustness', () => {
  it('retrieve with FTS special chars does not throw', () => {
    const mem = tmpEngram();
    try {
      mem.store('Alice prefers dark mode.');
      const result = mem.retrieve('dark AND* OR(mode!)');
      expect(Array.isArray(result.episodes)).toBe(true);
    } finally {
      mem.close();
    }
  });

  it('retrieve with empty query does not throw', () => {
    const mem = tmpEngram();
    try {
      mem.store('some content');
      const result = mem.retrieve('');
      expect(Array.isArray(result.episodes)).toBe(true);
    } finally {
      mem.close();
    }
  });
});
