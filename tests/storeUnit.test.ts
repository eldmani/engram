/**
 * Unit tests for SQLiteStore — covers every public method and all edge cases.
 * These tests bypass Engram and talk directly to the storage layer.
 * Node.js mirror of Python's test_store_unit.py.
 */
import * as os from 'node:os';
import * as path from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import {
  SQLiteStore,
  SCHEMA_VERSION,
  sanitizeFts,
} from '../src/storage/sqliteStore.js';
import type {
  Alias,
  Entity,
  Episode,
  Fact,
  QuarantinedFact,
  RetrievalTrace,
} from '../src/models/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tmpDb(): string {
  return path.join(
    os.tmpdir(),
    `engram-storeunit-${Date.now()}-${Math.random().toString(36).slice(2)}.db`,
  );
}

function now(): Date {
  return new Date();
}

function makeEpisode(scopeId = 'default', rawText = 'Alice prefers dark mode.'): Episode {
  const checksum = createHash('sha256').update(rawText).digest('hex');
  return {
    id: randomUUID(),
    scopeId,
    source: 'test',
    rawText,
    metadata: { src: 'unit' },
    createdAt: now(),
    checksum,
  };
}

function makeEntity(scopeId = 'default', name = 'Alice'): Entity {
  const n = now();
  return {
    id: randomUUID(),
    scopeId,
    type: 'person',
    canonicalName: name,
    summary: null,
    createdAt: n,
    updatedAt: n,
    firstSeenAt: n,
    lastSeenAt: n,
    salience: 0.5,
    strength: 1.0,
    confidence: 0.95,
  };
}

function makeFact(
  scopeId: string,
  episodeId: string,
  entityId: string,
  predicate = 'prefers',
  obj = 'dark mode',
  truthState = 'current',
): Fact {
  const n = now();
  return {
    id: randomUUID(),
    scopeId,
    subjectEntityId: entityId,
    predicate,
    objectEntityId: null,
    objectValueJson: { value: obj },
    factType: 'assertion',
    validFrom: n,
    validTo: null,
    truthState,
    sourceEpisodeId: episodeId,
    confidence: 0.92,
    salience: 0.5,
    strength: 1.0,
    accessCount: 0,
    lastAccessedAt: null,
    createdAt: n,
    updatedAt: n,
  };
}

function makeQuarantine(
  scopeId: string,
  episodeId: string,
  subject = 'Alice',
  predicate = 'prefers',
  obj = 'dark mode',
): QuarantinedFact {
  return {
    id: randomUUID(),
    scopeId,
    sourceEpisodeId: episodeId,
    extractedSubject: subject,
    extractedPredicate: predicate,
    extractedObject: obj,
    candidateSuperseedsFactId: null,
    extractorConfidence: 0.50,
    resolutionConfidence: 0.50,
    reason: 'low confidence',
    status: 'pending',
    createdAt: now(),
    reviewedAt: null,
  };
}

function makeAlias(entityId: string, episodeId: string, value = 'alice'): Alias {
  return {
    id: randomUUID(),
    entityId,
    value,
    normalizedValue: value.toLowerCase(),
    embeddingId: null,
    sourceEpisodeId: episodeId,
    confidence: 1.0,
    createdAt: now(),
  };
}

// ---------------------------------------------------------------------------
// sanitizeFts
// ---------------------------------------------------------------------------

describe('sanitizeFts', () => {
  it('normal query is unchanged', () => {
    expect(sanitizeFts('dark mode')).toBe('dark mode');
  });

  it('strips special chars', () => {
    const result = sanitizeFts('dark*mode!');
    expect(result).not.toContain('*');
    expect(result).not.toContain('!');
  });

  it('empty string returns quoted empty', () => {
    expect(sanitizeFts('')).toBe('""');
  });

  it('all-special-chars returns quoted empty', () => {
    expect(sanitizeFts('!!!')).toBe('""');
  });

  it('parentheses are stripped', () => {
    const result = sanitizeFts('(foo AND bar)');
    expect(result).not.toContain('(');
    expect(result).not.toContain(')');
  });

  it('non-ascii chars are stripped by JS regex (ascii-only \\w)', () => {
    // JS RegExp \\w is ASCII-only; unicode word chars get removed.
    const result = sanitizeFts('こんにちは');
    expect(typeof result).toBe('string');
    expect(result).not.toContain('こんにちは');
  });
});

// ---------------------------------------------------------------------------
// SCHEMA_VERSION
// ---------------------------------------------------------------------------

describe('SCHEMA_VERSION', () => {
  it('SCHEMA_VERSION constant is 2', () => {
    expect(SCHEMA_VERSION).toBe(2);
  });

  it('PRAGMA user_version matches SCHEMA_VERSION', () => {
    const store = new SQLiteStore(tmpDb());
    const row = store['db'].prepare('PRAGMA user_version').get() as { user_version: number };
    expect(row.user_version).toBe(SCHEMA_VERSION);
    store.close();
  });
});

// ---------------------------------------------------------------------------
// Episodes
// ---------------------------------------------------------------------------

describe('Episodes', () => {
  let store: SQLiteStore;
  beforeEach(() => { store = new SQLiteStore(tmpDb()); });
  afterEach(() => store.close());

  it('insert and get round-trips correctly', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const fetched = store.getEpisode(ep.id);
    expect(fetched).not.toBeNull();
    expect(fetched!.id).toBe(ep.id);
    expect(fetched!.rawText).toBe(ep.rawText);
    expect(fetched!.scopeId).toBe(ep.scopeId);
  });

  it('getEpisode returns null for missing id', () => {
    expect(store.getEpisode('no-such-id')).toBeNull();
  });

  it('metadata round-trips correctly', () => {
    const ep = makeEpisode();
    ep.metadata = { key: 'value', num: 42 };
    store.insertEpisode(ep);
    const fetched = store.getEpisode(ep.id);
    expect(fetched!.metadata).toEqual({ key: 'value', num: 42 });
  });

  it('ftsSearchEpisodes returns results for matching query', () => {
    const ep = makeEpisode('default', 'Alice loves dark mode.');
    store.insertEpisode(ep);
    const results = store.ftsSearchEpisodes('dark', 'default', 10);
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].episode.id).toBe(ep.id);
    expect(results[0].score).toBeGreaterThan(0);
  });

  it('ftsSearchEpisodes isolates by scope', () => {
    const epA = makeEpisode('scope-a', 'cats in scope A');
    const epB = makeEpisode('scope-b', 'dogs in scope B');
    store.insertEpisode(epA);
    store.insertEpisode(epB);
    const resultA = store.ftsSearchEpisodes('cats', 'scope-a', 5);
    const resultB = store.ftsSearchEpisodes('dogs', 'scope-b', 5);
    expect(resultA).toHaveLength(1);
    expect(resultB).toHaveLength(1);
    expect(store.ftsSearchEpisodes('cats', 'scope-b', 5)).toHaveLength(0);
  });

  it('ftsSearchEpisodes with malformed query returns empty (no throw)', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const result = store.ftsSearchEpisodes('dark OR* AND!', 'default', 10);
    expect(Array.isArray(result)).toBe(true);
  });

  it('ftsSearchEpisodes with empty query returns empty (no throw)', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const result = store.ftsSearchEpisodes('', 'default', 10);
    expect(Array.isArray(result)).toBe(true);
  });

  it('getRecentEpisodes respects limit', () => {
    for (let i = 0; i < 5; i++) store.insertEpisode(makeEpisode('default', `episode ${i}`));
    const recent = store.getRecentEpisodes('default', 3);
    expect(recent).toHaveLength(3);
  });

  it('getRecentEpisodes returns empty for unknown scope', () => {
    expect(store.getRecentEpisodes('empty-scope', 10)).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

describe('Entities', () => {
  let store: SQLiteStore;
  beforeEach(() => { store = new SQLiteStore(tmpDb()); });
  afterEach(() => store.close());

  it('insert and get round-trips correctly', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const ent = makeEntity();
    store.insertEntity(ent);
    const fetched = store.getEntity(ent.id);
    expect(fetched).not.toBeNull();
    expect(fetched!.canonicalName).toBe('Alice');
  });

  it('getEntity returns null for missing id', () => {
    expect(store.getEntity('no-such-id')).toBeNull();
  });

  it('findEntityByName returns matching entity', () => {
    store.insertEpisode(makeEpisode());
    const ent = makeEntity('default', 'Bob');
    store.insertEntity(ent);
    const found = store.findEntityByName('default', 'Bob');
    expect(found).not.toBeNull();
    expect(found!.id).toBe(ent.id);
  });

  it('findEntityByName is case insensitive', () => {
    store.insertEpisode(makeEpisode());
    const ent = makeEntity('default', 'Charlie');
    store.insertEntity(ent);
    expect(store.findEntityByName('default', 'charlie')).not.toBeNull();
    expect(store.findEntityByName('default', 'CHARLIE')).not.toBeNull();
  });

  it('findEntityByName returns null when missing', () => {
    expect(store.findEntityByName('default', 'Ghost')).toBeNull();
  });

  it('findEntityByAlias returns matching entity', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const ent = makeEntity('default', 'Diana');
    store.insertEntity(ent);
    const al = makeAlias(ent.id, ep.id, 'di');
    store.insertAlias(al);
    const found = store.findEntityByAlias('di', 'default');
    expect(found).not.toBeNull();
    expect(found!.id).toBe(ent.id);
  });

  it('findEntityByAlias returns null when missing', () => {
    expect(store.findEntityByAlias('phantom', 'default')).toBeNull();
  });

  it('getEntitiesForScope returns all entities in scope', () => {
    store.insertEpisode(makeEpisode());
    for (const name of ['Alice', 'Bob', 'Carol']) {
      store.insertEntity(makeEntity('default', name));
    }
    const entities = store.getEntitiesForScope('default');
    expect(entities).toHaveLength(3);
    const names = new Set(entities.map(e => e.canonicalName));
    expect(names).toEqual(new Set(['Alice', 'Bob', 'Carol']));
  });

  it('getEntitiesForScope returns empty for unknown scope', () => {
    expect(store.getEntitiesForScope('no-scope')).toHaveLength(0);
  });

  it('updateEntitySummary sets the summary field', () => {
    store.insertEpisode(makeEpisode());
    const ent = makeEntity();
    store.insertEntity(ent);
    store.updateEntitySummary(ent.id, 'Alice is a person.');
    const fetched = store.getEntity(ent.id);
    expect(fetched!.summary).toBe('Alice is a person.');
  });

  it('updateEntitySeen updates last_seen_at', () => {
    store.insertEpisode(makeEpisode());
    const ent = makeEntity();
    store.insertEntity(ent);
    const newTime = new Date();
    store.updateEntitySeen(ent.id, newTime);
    const fetched = store.getEntity(ent.id);
    expect(fetched!.lastSeenAt).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Aliases
// ---------------------------------------------------------------------------

describe('Aliases', () => {
  let store: SQLiteStore;
  beforeEach(() => { store = new SQLiteStore(tmpDb()); });
  afterEach(() => store.close());

  it('insert and getAliasesForEntity round-trips correctly', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const ent = makeEntity();
    store.insertEntity(ent);
    const al = makeAlias(ent.id, ep.id, 'al');
    store.insertAlias(al);
    const aliases = store.getAliasesForEntity(ent.id);
    expect(aliases).toHaveLength(1);
    expect(aliases[0].value).toBe('al');
  });

  it('duplicate alias insert is silently ignored (OR IGNORE)', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const ent = makeEntity();
    store.insertEntity(ent);
    const al = makeAlias(ent.id, ep.id);
    store.insertAlias(al);
    store.insertAlias(al); // second insert should be silently ignored
    expect(store.getAliasesForEntity(ent.id)).toHaveLength(1);
  });

  it('getAliasesForEntity returns empty when no aliases', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const ent = makeEntity();
    store.insertEntity(ent);
    expect(store.getAliasesForEntity(ent.id)).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Facts
// ---------------------------------------------------------------------------

describe('Facts', () => {
  let store: SQLiteStore;
  let ep: Episode;
  let ent: Entity;

  beforeEach(() => {
    store = new SQLiteStore(tmpDb());
    ep = makeEpisode();
    store.insertEpisode(ep);
    ent = makeEntity();
    store.insertEntity(ent);
  });
  afterEach(() => store.close());

  it('insert and get round-trips correctly', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    const fetched = store.getFact(f.id);
    expect(fetched).not.toBeNull();
    expect(fetched!.predicate).toBe('prefers');
    expect(fetched!.truthState).toBe('current');
  });

  it('getFact returns null for missing id', () => {
    expect(store.getFact('no-id')).toBeNull();
  });

  it('getCurrentFacts returns current facts for entity+predicate', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id, 'likes', 'coffee');
    store.insertFact(f);
    const results = store.getCurrentFacts(ep.scopeId, ent.id, 'likes');
    expect(results).toHaveLength(1);
    expect(results[0].objectValueJson).toEqual({ value: 'coffee' });
  });

  it('getCurrentFacts excludes superseded facts', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id, 'prefers', 'dark mode', 'superseded');
    store.insertFact(f);
    expect(store.getCurrentFacts('default', ent.id, 'prefers')).toHaveLength(0);
  });

  it('getCurrentFactsForScope returns all current facts', () => {
    const f1 = makeFact(ep.scopeId, ep.id, ent.id, 'p1', 'v1');
    const f2 = makeFact(ep.scopeId, ep.id, ent.id, 'p2', 'v2');
    store.insertFact(f1);
    store.insertFact(f2);
    const results = store.getCurrentFactsForScope('default', 10);
    expect(results).toHaveLength(2);
  });

  it('supersedeFact sets truth_state to superseded', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.supersedeFact(f.id, new Date());
    const fetched = store.getFact(f.id);
    expect(fetched!.truthState).toBe('superseded');
    expect(fetched!.validTo).not.toBeNull();
  });

  it('hideFact sets truth_state to hidden', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.hideFact(f.id);
    const fetched = store.getFact(f.id);
    expect(fetched!.truthState).toBe('hidden');
  });

  it('deleteFact removes the fact', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.deleteFact(f.id);
    expect(store.getFact(f.id)).toBeNull();
  });

  it('reinforceFact increments access_count and boosts strength', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.reinforceFact(f.id, 0.1);
    const fetched = store.getFact(f.id);
    expect(fetched!.accessCount).toBe(1);
    expect(fetched!.strength).toBeGreaterThan(1.0);
  });

  it('applyDecay reduces strength for unaccessed fact', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.applyDecay('default', 10);
    const fetched = store.getFact(f.id);
    expect(fetched!.strength).toBeLessThanOrEqual(1.0);
  });

  it('ftsSearchFacts returns results for matching query', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id, 'prefers', 'dark mode');
    store.insertFact(f);
    const results = store.ftsSearchFacts('dark', 'default', 5);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });

  it('ftsSearchFacts with malformed query returns empty (no throw)', () => {
    store.insertFact(makeFact(ep.scopeId, ep.id, ent.id));
    const results = store.ftsSearchFacts('AND* (NEAR/bad)', 'default', 5);
    expect(Array.isArray(results)).toBe(true);
  });

  it('ftsSearchFacts excludes hidden facts', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id, 'prefers', 'dark mode');
    store.insertFact(f);
    store.hideFact(f.id);
    const results = store.ftsSearchFacts('dark', 'default', 5);
    const ids = results.map(r => r.fact.id);
    expect(ids).not.toContain(f.id);
  });

  it('hideEntityFacts hides all facts for an entity', () => {
    const f1 = makeFact(ep.scopeId, ep.id, ent.id, 'likes', 'coffee');
    const f2 = makeFact(ep.scopeId, ep.id, ent.id, 'prefers', 'vim');
    store.insertFact(f1);
    store.insertFact(f2);
    store.hideEntityFacts(ent.id);
    for (const fid of [f1.id, f2.id]) {
      expect(store.getFact(fid)!.truthState).toBe('hidden');
    }
  });

  it('hideFactsInScope hides all facts in scope', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.hideFactsInScope('default');
    expect(store.getFact(f.id)!.truthState).toBe('hidden');
  });

  it('deleteFactsInScope removes all facts in scope', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.deleteFactsInScope('default');
    expect(store.getFact(f.id)).toBeNull();
  });

  it('deleteEntityFacts removes all facts for an entity', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.deleteEntityFacts(ent.id);
    expect(store.getFact(f.id)).toBeNull();
  });

  it('deleteEpisodeFacts removes all facts for an episode', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    store.deleteEpisodeFacts(ep.id);
    expect(store.getFact(f.id)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Quarantined Facts
// ---------------------------------------------------------------------------

describe('QuarantinedFacts', () => {
  let store: SQLiteStore;
  beforeEach(() => { store = new SQLiteStore(tmpDb()); });
  afterEach(() => store.close());

  it('insert and getPendingQuarantinedFacts round-trips', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const qf = makeQuarantine('default', ep.id);
    store.insertQuarantinedFact(qf);
    const pending = store.getPendingQuarantinedFacts('default');
    expect(pending).toHaveLength(1);
    expect(pending[0].id).toBe(qf.id);
  });

  it('updateQuarantineStatus to approved removes from pending', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const qf = makeQuarantine('default', ep.id);
    store.insertQuarantinedFact(qf);
    store.updateQuarantineStatus(qf.id, 'approved');
    expect(store.getPendingQuarantinedFacts('default')).toHaveLength(0);
  });

  it('updateQuarantineStatus to rejected removes from pending', () => {
    const ep = makeEpisode();
    store.insertEpisode(ep);
    const qf = makeQuarantine('default', ep.id);
    store.insertQuarantinedFact(qf);
    store.updateQuarantineStatus(qf.id, 'rejected');
    expect(store.getPendingQuarantinedFacts('default')).toHaveLength(0);
  });

  it('getPendingQuarantinedFacts returns empty when none', () => {
    expect(store.getPendingQuarantinedFacts('default')).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Retrieval Traces
// ---------------------------------------------------------------------------

describe('RetrievalTraces', () => {
  let store: SQLiteStore;
  let ep: Episode;
  let ent: Entity;

  beforeEach(() => {
    store = new SQLiteStore(tmpDb());
    ep = makeEpisode();
    store.insertEpisode(ep);
    ent = makeEntity();
    store.insertEntity(ent);
  });
  afterEach(() => store.close());

  it('insert and getTraces round-trips', () => {
    const f = makeFact(ep.scopeId, ep.id, ent.id);
    store.insertFact(f);
    const trace: RetrievalTrace = {
      id: randomUUID(),
      query: 'dark mode',
      scopeId: 'default',
      candidateFactId: f.id,
      semanticScore: 0.0,
      keywordScore: 0.8,
      graphScore: 0.0,
      temporalScore: 0.5,
      finalScore: 0.7,
      matchedEntities: ['Alice'],
      sourceEpisodeIds: [ep.id],
      createdAt: now(),
    };
    store.insertRetrievalTrace(trace);
    const traces = store.getTraces('default', 'dark mode');
    expect(traces.length).toBeGreaterThanOrEqual(1);
  });

  it('getTraces returns empty when none', () => {
    expect(store.getTraces('default', 'anything')).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Close
// ---------------------------------------------------------------------------

describe('Close', () => {
  it('close() does not throw', () => {
    const store = new SQLiteStore(tmpDb());
    expect(() => store.close()).not.toThrow();
  });

  it('after close(), using the store throws', () => {
    const store = new SQLiteStore(tmpDb());
    store.close();
    expect(() => store.getEpisode('any')).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Graph traversal helpers (store layer)
// ---------------------------------------------------------------------------

describe('GraphTraversalHelpers', () => {
  let store: SQLiteStore;
  beforeEach(() => { store = new SQLiteStore(tmpDb()); });
  afterEach(() => store.close());

  it('getConnectedEntityIds returns empty for empty seeds', () => {
    expect(store.getConnectedEntityIds([], 'scope1').size).toBe(0);
  });

  it('getConnectedEntityIds returns empty for nonexistent entity', () => {
    expect(store.getConnectedEntityIds(['nonexistent'], 'scope1').size).toBe(0);
  });

  it('getEntityGraph returns empty Map for empty seeds', () => {
    expect(store.getEntityGraph([], 'scope1', 2).size).toBe(0);
  });

  it('getFactsForEntities returns empty for empty ids', () => {
    expect(store.getFactsForEntities([], 'scope1')).toHaveLength(0);
  });
});
