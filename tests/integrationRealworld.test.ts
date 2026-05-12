/**
 * Real-world integration tests — simulates an AI assistant's full memory lifecycle.
 *
 * Scenario: A personal AI assistant learns about users, updates its knowledge
 * over several conversations, handles ambiguous/low-confidence info, retrieves
 * answers for follow-up questions, maintains multiple user scopes, and performs
 * memory housekeeping.
 *
 * No mocks. A real SQLite database on disk is used throughout.
 * Node.js mirror of Python's test_integration_realworld.py.
 */
import * as os from 'node:os';
import * as path from 'node:path';
import {
  Engram,
  BaseExtractor,
  type EngramConfig,
  type ExtractionResult,
  type ExtractedEntity,
  type ExtractedFact,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// DSL-based deterministic extractor
// ---------------------------------------------------------------------------

/**
 * Parses a tiny DSL so we can write predictable tests without needing a live LLM.
 * Input format: "ENTITY:name:type  FACT:subj:pred:obj:conf"
 * Multiple entities/facts are double-space-separated blocks.
 */
class LLMExtractor extends BaseExtractor {
  constructor(private confidence = 0.95) {
    super();
  }

  extract(data: string | Record<string, unknown>): ExtractionResult {
    const text = typeof data === 'string' ? data : JSON.stringify(data);
    const entities: ExtractedEntity[] = [];
    const facts: ExtractedFact[] = [];

    for (const token of text.split('  ')) {
      const t = token.trim();
      if (t.startsWith('ENTITY:')) {
        const parts = t.split(':');
        const name = parts[1]?.trim() ?? '';
        const etype = parts[2]?.trim() ?? 'unknown';
        if (name) entities.push({ name, type: etype, confidence: 0.97 });
      } else if (t.startsWith('FACT:')) {
        const parts = t.split(':');
        const subj = parts[1]?.trim() ?? '';
        const pred = parts[2]?.trim() ?? '';
        const obj  = parts[3]?.trim() ?? '';
        const conf = parts[4] ? parseFloat(parts[4]) : 0.90;
        if (subj && pred && obj) {
          facts.push({ subject: subj, predicate: pred, object: obj, confidence: conf });
        }
      }
    }

    return { entities, facts, confidence: this.confidence };
  }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function newMem(overrides?: Partial<EngramConfig>): Engram {
  const dbPath = path.join(
    os.tmpdir(),
    `engram-integ-${Date.now()}-${Math.random().toString(36).slice(2)}.db`,
  );
  const mem = new Engram({ dbPath, ...overrides });
  mem.setExtractor(new LLMExtractor());
  return mem;
}

// ---------------------------------------------------------------------------
// Scenario 1 — Basic learning and recall
// ---------------------------------------------------------------------------

describe('Scenario01_BasicLearningAndRecall', () => {
  it('learns facts and recalls them correctly', () => {
    const mem = newMem();
    try {
      const r1 = mem.store(
        'ENTITY:Sarah:person  FACT:Sarah:prefers:dark mode:0.95',
        { turn: 1, session: 's001' },
      );
      expect(r1.episodeId).toBeTruthy();
      expect(r1.entityIds).toHaveLength(1);
      expect(r1.factIds).toHaveLength(1);

      const r2 = mem.store(
        'ENTITY:Sarah:person  FACT:Sarah:works_at:Acme Corp:0.93  FACT:Sarah:uses:Python:0.91',
        { turn: 2, session: 's001' },
      );
      expect(r2.factIds).toHaveLength(2);

      const result = mem.retrieve('Sarah dark mode preferences');
      expect(result.facts.length).toBeGreaterThanOrEqual(1);
      const predicates = result.facts.map(f => f.predicate);
      expect(predicates).toContain('prefers');

      const result2 = mem.retrieve('where does Sarah work');
      const workplaceFacts = result2.facts.filter(f => f.predicate === 'works_at');
      expect(workplaceFacts.length).toBeGreaterThanOrEqual(1);
      expect(workplaceFacts[0].objectValueJson!['value']).toBe('Acme Corp');

      const ctx = mem.getContext('Sarah profile');
      expect(ctx.formatted.trim()).toBeTruthy();
      expect(ctx.formatted).toContain('Sarah');
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 2 — Temporal supersession (knowledge updates)
// ---------------------------------------------------------------------------

describe('Scenario02_TemporalSupersession', () => {
  it('updates preference and preserves history as superseded', () => {
    const mem = newMem();
    try {
      const r1 = mem.store('ENTITY:Sarah:person  FACT:Sarah:prefers:dark mode:0.95');
      const fidOriginal = r1.factIds[0];

      const r2 = mem.store('ENTITY:Sarah:person  FACT:Sarah:prefers:light mode:0.95');
      const fidUpdated = r2.factIds[0];

      expect(fidOriginal).not.toBe(fidUpdated);

      const entities = mem._store.getEntitiesForScope('default');
      expect(entities.length).toBeGreaterThanOrEqual(1);
      const sarah = entities[0];
      const current = mem._store.getCurrentFacts('default', sarah.id, 'prefers');
      expect(current).toHaveLength(1);
      expect(current[0].objectValueJson!['value']).toBe('light mode');

      const old = mem._store.getFact(fidOriginal);
      expect(old).not.toBeNull();
      expect(old!.truthState).toBe('superseded');
    } finally {
      mem.close();
    }
  });

  it('manual updateFact reflects new value in context', () => {
    const mem = newMem();
    try {
      const r1 = mem.store('ENTITY:Bob:person  FACT:Bob:lives_in:London:0.92');
      const fid = r1.factIds[0];

      const newFact = mem.updateFact(fid, 'Berlin');
      expect(newFact.objectValueJson!['value']).toBe('Berlin');
      expect(newFact.truthState).toBe('current');

      const old = mem._store.getFact(fid);
      expect(old!.truthState).toBe('superseded');
      expect(old!.validTo).not.toBeNull();

      const ctx = mem.getContext('where does Bob live');
      expect(ctx.formatted).toContain('Berlin');
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 3 — Confidence gate & quarantine
// ---------------------------------------------------------------------------

describe('Scenario03_ConfidenceGate', () => {
  it('low-confidence extraction goes to quarantine not fact store', () => {
    const mem = newMem();
    mem.setExtractor(new LLMExtractor(0.50)); // below threshold
    try {
      const r = mem.store('ENTITY:Carol:person  FACT:Carol:owns:yacht:0.40');
      expect(r.factIds).toHaveLength(0);
      expect(r.quarantinedCount).toBeGreaterThanOrEqual(1);

      const ep = mem._store.getEpisode(r.episodeId);
      expect(ep).not.toBeNull();

      const expl = mem.explain('Carol yacht');
      expect(expl.quarantinedFacts.length).toBeGreaterThanOrEqual(1);
      const qf = expl.quarantinedFacts[0];
      expect(qf.extractedSubject).toBe('Carol');
      expect(qf.extractedPredicate).toBe('owns');
      expect(qf.status).toBe('pending');
    } finally {
      mem.close();
    }
  });

  it('subsequent high-confidence confirmation reconsolidates quarantine', () => {
    const mem = newMem();
    try {
      mem.setExtractor(new LLMExtractor(0.50));
      const rLow = mem.store('ENTITY:Dave:person  FACT:Dave:role:CEO:0.40');
      expect(rLow.quarantinedCount).toBeGreaterThanOrEqual(1);

      mem.setExtractor(new LLMExtractor(0.95));
      const rHigh = mem.store('ENTITY:Dave:person  FACT:Dave:role:CEO:0.95');
      expect(rHigh.factIds).toHaveLength(1);

      mem.workers.runReconsolidation('default');
      const pending = mem._store.getPendingQuarantinedFacts('default');
      expect(pending).toHaveLength(0);
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 4 — Multi-user scope isolation
// ---------------------------------------------------------------------------

describe('Scenario04_ScopeIsolation', () => {
  it('facts in separate scopes never bleed across', () => {
    const mem = newMem();
    try {
      mem.store('ENTITY:Alice:person  FACT:Alice:prefers:vim:0.95', {}, 'user-alice');
      mem.store('ENTITY:Bob:person  FACT:Bob:prefers:emacs:0.95', {}, 'user-bob');

      const aliceResult = mem.retrieve('editor preference', undefined, 'user-alice');
      const bobResult   = mem.retrieve('editor preference', undefined, 'user-bob');

      const aliceValues = new Set(aliceResult.facts.map(f => f.objectValueJson?.['value']));
      const bobValues   = new Set(bobResult.facts.map(f => f.objectValueJson?.['value']));

      expect(aliceValues).toContain('vim');
      expect(bobValues).toContain('emacs');
      expect(aliceValues).not.toContain('emacs');
      expect(bobValues).not.toContain('vim');
    } finally {
      mem.close();
    }
  });

  it('forget scope does not affect other scopes', () => {
    const mem = newMem();
    try {
      mem.store('ENTITY:Alice:person  FACT:Alice:likes:cats:0.95', {}, 'user-alice');
      const r = mem.store('ENTITY:Bob:person  FACT:Bob:likes:dogs:0.95', {}, 'user-bob');
      const bobFid = r.factIds[0];

      mem.forget({ scopeId: 'user-alice' }, true);

      expect(mem._store.getFact(bobFid)).not.toBeNull();
      expect(mem._store.getFact(bobFid)!.truthState).toBe('current');
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 5 — Forget paths: fact / entity / episode / scope
// ---------------------------------------------------------------------------

describe('Scenario05_ForgetPaths', () => {
  function setup(mem: Engram): { factId: string; entityId: string; episodeId: string } {
    const r = mem.store('ENTITY:Eve:person  FACT:Eve:speaks:French:0.95');
    const entity = mem._store.getEntitiesForScope('default')[0];
    return { factId: r.factIds[0], entityId: entity.id, episodeId: r.episodeId };
  }

  it('soft forget by factId hides the fact', () => {
    const mem = newMem();
    try {
      const { factId } = setup(mem);
      mem.forget({ factId }, false);
      expect(mem._store.getFact(factId)!.truthState).toBe('hidden');
    } finally {
      mem.close();
    }
  });

  it('hard forget by factId deletes the fact', () => {
    const mem = newMem();
    try {
      const { factId } = setup(mem);
      mem.forget({ factId }, true);
      expect(mem._store.getFact(factId)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('soft forget by entityId hides entity facts', () => {
    const mem = newMem();
    try {
      const { factId, entityId } = setup(mem);
      const fr = mem.forget({ entityId }, false);
      expect(fr.affectedEntities).toBeGreaterThanOrEqual(1);
      expect(mem._store.getFact(factId)!.truthState).toBe('hidden');
    } finally {
      mem.close();
    }
  });

  it('hard forget by entityId deletes entity facts', () => {
    const mem = newMem();
    try {
      const { factId, entityId } = setup(mem);
      mem.forget({ entityId }, true);
      expect(mem._store.getFact(factId)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('soft forget by scope hides all scope facts', () => {
    const mem = newMem();
    try {
      const { factId } = setup(mem);
      mem.forget({ scopeId: 'default' }, false);
      expect(mem._store.getFact(factId)!.truthState).toBe('hidden');
    } finally {
      mem.close();
    }
  });

  it('hard forget by scope deletes all scope facts', () => {
    const mem = newMem();
    try {
      const { factId } = setup(mem);
      mem.forget({ scopeId: 'default' }, true);
      expect(mem._store.getFact(factId)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('hard forget by episodeId deletes episode facts', () => {
    const mem = newMem();
    try {
      const { factId, episodeId } = setup(mem);
      mem.forget({ episodeId }, true);
      expect(mem._store.getFact(factId)).toBeNull();
    } finally {
      mem.close();
    }
  });

  it('invalid selector throws', () => {
    const mem = newMem();
    try {
      expect(() => mem.forget({})).toThrow();
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 6 — Workers: decay, reinforce, heat, cleanup
// ---------------------------------------------------------------------------

describe('Scenario06_Workers', () => {
  it('reinforcement twice increases strength and access_count by 2', () => {
    const mem = newMem();
    try {
      const r = mem.store('ENTITY:Frank:person  FACT:Frank:knows:Python:0.95');
      const fid = r.factIds[0];
      const originalStrength = mem._store.getFact(fid)!.strength;
      const originalCount    = mem._store.getFact(fid)!.accessCount;

      mem.workers.reinforce([fid]);
      mem.workers.reinforce([fid]);

      const fact = mem._store.getFact(fid)!;
      expect(fact.accessCount).toBe(originalCount + 2);
      expect(fact.strength).toBeGreaterThan(originalStrength);
    } finally {
      mem.close();
    }
  });

  it('decay weakens unreinforced fact', () => {
    const mem = newMem();
    try {
      const r = mem.store('ENTITY:Grace:person  FACT:Grace:likes:jazz:0.95');
      const fid = r.factIds[0];
      const strengthBefore = mem._store.getFact(fid)!.strength;

      mem.workers.runDecay('default');

      const strengthAfter = mem._store.getFact(fid)!.strength;
      expect(strengthAfter).toBeLessThanOrEqual(strengthBefore);
    } finally {
      mem.close();
    }
  });

  it('well-reinforced fact stays above 1.0 after one decay pass', () => {
    const mem = newMem();
    try {
      const r = mem.store('ENTITY:Hank:person  FACT:Hank:speciality:ML:0.95');
      const fid = r.factIds[0];

      for (let i = 0; i < 20; i++) mem.workers.reinforce([fid]);
      const strengthAfterReinforce = mem._store.getFact(fid)!.strength;

      mem.workers.runDecay('default');
      const strengthAfterDecay = mem._store.getFact(fid)!.strength;

      expect(strengthAfterDecay).toBeGreaterThan(1.0);
    } finally {
      mem.close();
    }
  });

  it('computeHeat is consistent within the same second', () => {
    const mem = newMem();
    try {
      const r = mem.store('ENTITY:Iris:person  FACT:Iris:language:Rust:0.95');
      const fid = r.factIds[0];
      const fact = mem._store.getFact(fid)!;

      const heat1 = mem.workers.computeHeat(fact);
      const heat2 = mem.workers.computeHeat(fact);

      expect(Math.abs(heat1 - heat2)).toBeLessThan(0.001);
      expect(heat1).toBeGreaterThanOrEqual(0.0);
    } finally {
      mem.close();
    }
  });

  it('computeHeat increases after reinforcement', () => {
    const mem = newMem();
    try {
      const r = mem.store('ENTITY:Jack:person  FACT:Jack:hobby:chess:0.95');
      const fid = r.factIds[0];
      const heatBefore = mem.workers.computeHeat(mem._store.getFact(fid)!);

      for (let i = 0; i < 10; i++) mem.workers.reinforce([fid]);

      const heatAfter = mem.workers.computeHeat(mem._store.getFact(fid)!);
      expect(heatAfter).toBeGreaterThan(heatBefore);
    } finally {
      mem.close();
    }
  });

  it('runHeatPromotion returns hot facts when threshold is very low', () => {
    const mem = newMem({ heatPromotionThreshold: 0.0001 });
    try {
      const r = mem.store('ENTITY:Kim:person  FACT:Kim:skill:Go:0.95');
      const fid = r.factIds[0];
      for (let i = 0; i < 5; i++) mem.workers.reinforce([fid]);
      const hot = mem.workers.runHeatPromotion('default');
      expect(hot).toContain(fid);
    } finally {
      mem.close();
    }
  });

  it('runCleanup leaves valid facts intact', () => {
    const mem = newMem();
    try {
      const r = mem.store('ENTITY:Leo:person  FACT:Leo:city:Paris:0.95');
      const fid = r.factIds[0];
      mem.workers.runCleanup('default');
      expect(mem._store.getFact(fid)).not.toBeNull();
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 7 — FTS robustness
// ---------------------------------------------------------------------------

const badQueries = [
  '',
  '   ',
  'AND OR NOT',
  'dark AND* mode!',
  '(NEAR/bad query)',
  '*',
  '!@#$%^&*()',
  'He said "hello" AND goodbye',
];

describe('Scenario07_FtsRobustness', () => {
  test.each(badQueries)('bad query %j does not throw', (query) => {
    const mem = newMem();
    try {
      mem.store('ENTITY:Mia:person  FACT:Mia:likes:dark mode:0.95');
      const result = mem.retrieve(query);
      expect(Array.isArray(result.facts)).toBe(true);
      expect(Array.isArray(result.episodes)).toBe(true);
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 8 — Multi-fact entity profile (context assembly)
// ---------------------------------------------------------------------------

describe('Scenario08_RichEntityProfile', () => {
  it('entity with many facts produces a rich context block', () => {
    const mem = newMem();
    try {
      const turns = [
        'ENTITY:Nina:person  FACT:Nina:works_at:Google:0.95',
        'ENTITY:Nina:person  FACT:Nina:role:Staff Engineer:0.92',
        'ENTITY:Nina:person  FACT:Nina:lives_in:NYC:0.94',
        'ENTITY:Nina:person  FACT:Nina:speaks:English:0.99  FACT:Nina:speaks:Mandarin:0.97',
        'ENTITY:Nina:person  FACT:Nina:hobby:rock climbing:0.88',
      ];
      for (const turn of turns) mem.store(turn);

      const ctx = mem.getContext('Nina profile', 20);
      expect(ctx.formatted.trim()).toBeTruthy();

      const result = mem.retrieve('Nina engineer Google', 10);
      const predicates = new Set(result.facts.map(f => f.predicate));
      expect(predicates).toContain('works_at');
      expect(predicates).toContain('role');

      const entities = mem._store.getEntitiesForScope('default');
      expect(entities).toHaveLength(1);
      expect(entities[0].canonicalName).toBe('Nina');

      const allFacts = mem._store.getCurrentFactsForScope('default', 50);
      const predSet = new Set(allFacts.map(f => f.predicate));
      expect(predSet).toContain('works_at');
      expect(predSet).toContain('role');
      expect(predSet).toContain('lives_in');
      expect(predSet).toContain('hobby');
    } finally {
      mem.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Scenario 9 — Episode immutability & metadata
// ---------------------------------------------------------------------------

describe('Scenario09_EpisodeImmutability', () => {
  it('episode metadata is preserved exactly', () => {
    const mem = newMem();
    try {
      const r = mem.store(
        'ENTITY:Oscar:person  FACT:Oscar:prefers:Neovim:0.95',
        { session: 'abc123', turn: 7, model: 'gpt-5' },
      );
      const ep = mem._store.getEpisode(r.episodeId)!;
      expect(ep.metadata['session']).toBe('abc123');
      expect(ep.metadata['turn']).toBe(7);
      expect(ep.metadata['model']).toBe('gpt-5');
    } finally {
      mem.close();
    }
  });

  it('checksum is deterministic for same input', () => {
    const { createHash } = require('node:crypto');
    const text = 'Some deterministic input text.';
    const c1 = createHash('sha256').update(text).digest('hex');
    const c2 = createHash('sha256').update(text).digest('hex');
    expect(c1).toBe(c2);
    expect(c1).toHaveLength(64); // SHA-256
  });
});

// ---------------------------------------------------------------------------
// Scenario 10 — End-to-end AI assistant conversation replay
// ---------------------------------------------------------------------------

describe('Scenario10_AssistantConversationReplay', () => {
  it('full assistant conversation: store → retrieve → reinforce → update → forget → context', () => {
    const mem = newMem();
    const scope = 'user-pat';

    try {
      // Turn 1: onboarding
      mem.store(
        'ENTITY:Pat:person  FACT:Pat:name:Pat:0.99  FACT:Pat:role:data scientist:0.97',
        { turn: 1 },
        scope,
      );

      // Turn 2: preferences
      mem.store(
        'ENTITY:Pat:person  FACT:Pat:prefers:dark mode:0.95  FACT:Pat:uses:Jupyter:0.93',
        { turn: 2 },
        scope,
      );

      // Turn 3: retrieve and reinforce
      const result = mem.retrieve('Pat preferences tools', undefined, scope);
      expect(result.facts.length).toBeGreaterThanOrEqual(1);
      const retrievedFactIds = result.facts.map(f => f.id);
      mem.workers.reinforce(retrievedFactIds);

      // Turn 4: update preference
      const prefs = result.facts.filter(f => f.predicate === 'prefers');
      if (prefs.length > 0) {
        mem.updateFact(prefs[0].id, 'light mode');
      }

      // Turn 5: context should reflect update
      const ctx = mem.getContext('Pat current preferences', undefined, scope);
      expect(ctx.formatted.trim()).toBeTruthy();
      expect(ctx.formatted).toContain('light mode');

      // Turn 6: privacy erasure
      const entities = mem._store.getEntitiesForScope(scope);
      const pat = entities.find(e => e.canonicalName === 'Pat');
      expect(pat).toBeDefined();
      const forgetResult = mem.forget({ entityId: pat!.id }, true);
      expect(forgetResult.affectedEntities).toBeGreaterThanOrEqual(1);

      const remaining = mem._store.getCurrentFactsForScope(scope, 100);
      const patFacts = remaining.filter(f => f.subjectEntityId === pat!.id);
      expect(patFacts).toHaveLength(0);

      // Housekeeping
      mem.workers.runDecay(scope);
      mem.workers.runReconsolidation(scope);
      mem.workers.runCleanup(scope);
    } finally {
      mem.close();
    }
  });
});
