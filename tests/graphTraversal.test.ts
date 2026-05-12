/**
 * Graph traversal tests — covers getConnectedEntityIds, getEntityGraph,
 * getFactsForEntities, and end-to-end retrieval with graph scoring.
 * Node.js port of the graph-related sections of Python's test_embedding_and_graph.py.
 */
import * as os from 'node:os';
import * as path from 'node:path';
import {
  Engram,
  BaseExtractor,
  SQLiteStore,
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
    `engram-graph-${Date.now()}-${Math.random().toString(36).slice(2)}.db`,
  );
}

function tmpEngram(overrides?: Partial<EngramConfig>): Engram {
  return new Engram({ dbPath: tmpDb(), ...overrides });
}

/** Extracts exactly two entities linked by one fact. */
class TwoEntityExtractor extends BaseExtractor {
  constructor(
    private entityA: string,
    private entityB: string,
    private predicate: string,
    private confidence = 0.95,
  ) {
    super();
  }

  extract(_data: string | Record<string, unknown>): ExtractionResult {
    const entities: ExtractedEntity[] = [
      { name: this.entityA, type: 'person', confidence: this.confidence },
      { name: this.entityB, type: 'person', confidence: this.confidence },
    ];
    const facts: ExtractedFact[] = [
      {
        subject: this.entityA,
        predicate: this.predicate,
        object: this.entityB,
        factType: 'assertion',
        confidence: this.confidence,
      },
    ];
    return { entities, facts, confidence: this.confidence };
  }
}

/** Extracts a single entity + fact. */
class SingleFactExtractor extends BaseExtractor {
  constructor(
    private entity: string,
    private predicate: string,
    private obj: string,
    private confidence = 0.95,
  ) {
    super();
  }

  extract(_data: string | Record<string, unknown>): ExtractionResult {
    return {
      entities: [{ name: this.entity, type: 'person', confidence: this.confidence }],
      facts: [{
        subject: this.entity,
        predicate: this.predicate,
        object: this.obj,
        factType: 'assertion',
        confidence: this.confidence,
      }],
      confidence: this.confidence,
    };
  }
}

// ---------------------------------------------------------------------------
// SQLiteStore graph helpers (unit-level)
// ---------------------------------------------------------------------------

describe('SQLiteStore graph helpers', () => {
  let store: SQLiteStore;

  beforeEach(() => { store = new SQLiteStore(tmpDb()); });
  afterEach(() => store.close());

  it('getConnectedEntityIds returns empty Set for empty seeds', () => {
    expect(store.getConnectedEntityIds([], 'scope1').size).toBe(0);
  });

  it('getConnectedEntityIds returns empty Set for nonexistent entity', () => {
    expect(store.getConnectedEntityIds(['nonexistent-id'], 'scope1').size).toBe(0);
  });

  it('getEntityGraph returns empty Map for empty seeds', () => {
    expect(store.getEntityGraph([], 'scope1', 2).size).toBe(0);
  });

  it('getFactsForEntities returns empty array for empty ids', () => {
    expect(store.getFactsForEntities([], 'scope1')).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Graph traversal via Engram
// ---------------------------------------------------------------------------

describe('GraphTraversal', () => {
  it('one-hop neighbour is found when storing A→B link', () => {
    const mem = tmpEngram();
    mem.setExtractor(new TwoEntityExtractor('Alice', 'Bob', 'knows'));
    try {
      mem.store('Alice knows Bob.');
      const result = mem.retrieve('Bob');
      const predicates = result.facts.map(f => f.predicate);
      expect(predicates).toContain('knows');
    } finally {
      mem.close();
    }
  });

  it('seed entity facts are included in results', () => {
    const mem = tmpEngram();
    mem.setExtractor(new SingleFactExtractor('Carol', 'role', 'CEO'));
    try {
      mem.store('Carol is CEO.');
      const result = mem.retrieve('Carol role');
      const predicates = result.facts.map(f => f.predicate);
      expect(predicates).toContain('role');
    } finally {
      mem.close();
    }
  });

  it('getConnectedEntityIds is bidirectional', () => {
    const mem = tmpEngram();
    mem.setExtractor(new TwoEntityExtractor('Alice', 'Bob', 'reports_to'));
    try {
      mem.store('Alice reports to Bob.');

      const alice = mem._store.findEntityByName('default', 'Alice');
      const bob   = mem._store.findEntityByName('default', 'Bob');
      expect(alice).not.toBeNull();
      expect(bob).not.toBeNull();

      // Alice → Bob (outgoing)
      const fromAlice = mem._store.getConnectedEntityIds([alice!.id], 'default');
      expect(fromAlice).toContain(bob!.id);

      // Bob ← Alice (incoming)
      const fromBob = mem._store.getConnectedEntityIds([bob!.id], 'default');
      expect(fromBob).toContain(alice!.id);
    } finally {
      mem.close();
    }
  });

  it('seed entities are excluded from neighbor result', () => {
    const mem = tmpEngram();
    mem.setExtractor(new TwoEntityExtractor('Alice', 'Bob', 'works_with'));
    try {
      mem.store('Alice works with Bob.');
      const alice = mem._store.findEntityByName('default', 'Alice');
      expect(alice).not.toBeNull();
      const neighbors = mem._store.getConnectedEntityIds([alice!.id], 'default');
      expect(neighbors).not.toContain(alice!.id);
    } finally {
      mem.close();
    }
  });

  it('retrieval traces are written after retrieve()', () => {
    const mem = tmpEngram();
    mem.setExtractor(new TwoEntityExtractor('Alice', 'Bob', 'mentors'));
    try {
      mem.store('Alice mentors Bob.');
      mem.retrieve('Bob mentors');
      const traces = mem._store.getTraces('default', 'Bob mentors');
      expect(traces.length).toBeGreaterThanOrEqual(1);
    } finally {
      mem.close();
    }
  });

  it('graph scores in traces are numbers', () => {
    const mem = tmpEngram();
    mem.setExtractor(new TwoEntityExtractor('Alice', 'Bob', 'manages'));
    try {
      mem.store('Alice manages Bob.');
      const extra = new SingleFactExtractor('Alice', 'favourite_food', 'sushi');
      mem.setExtractor(extra);
      mem.store('Alice likes sushi.');

      mem.retrieve('Alice manages Bob');
      const traces = mem._store.getTraces('default', 'Alice manages Bob');
      expect(traces.length).toBeGreaterThanOrEqual(1);
      for (const trace of traces) {
        expect(typeof trace.graphScore).toBe('number');
        expect(trace.graphScore).toBeGreaterThanOrEqual(0.0);
      }
    } finally {
      mem.close();
    }
  });

  it('getEntityGraph BFS assigns correct hop distances', () => {
    const mem = tmpEngram();
    // Create A→B→C chain
    mem.setExtractor(new TwoEntityExtractor('A', 'B', 'links'));
    mem.store('A links B.');
    mem.setExtractor(new TwoEntityExtractor('B', 'C', 'links'));
    mem.store('B links C.');

    try {
      const aEnt = mem._store.findEntityByName('default', 'A');
      const bEnt = mem._store.findEntityByName('default', 'B');
      const cEnt = mem._store.findEntityByName('default', 'C');
      expect(aEnt).not.toBeNull();
      expect(bEnt).not.toBeNull();
      expect(cEnt).not.toBeNull();

      const graph = mem._store.getEntityGraph([aEnt!.id], 'default', 2);
      expect(graph.get(aEnt!.id)).toBe(0);  // seed = distance 0
      expect(graph.get(bEnt!.id)).toBe(1);  // 1-hop
      expect(graph.get(cEnt!.id)).toBe(2);  // 2-hop
    } finally {
      mem.close();
    }
  });

  it('getEntityGraph respects maxHops limit', () => {
    const mem = tmpEngram();
    mem.setExtractor(new TwoEntityExtractor('A', 'B', 'links'));
    mem.store('A links B.');
    mem.setExtractor(new TwoEntityExtractor('B', 'C', 'links'));
    mem.store('B links C.');

    try {
      const aEnt = mem._store.findEntityByName('default', 'A');
      const cEnt = mem._store.findEntityByName('default', 'C');

      // maxHops=1 should NOT reach C
      const graph1 = mem._store.getEntityGraph([aEnt!.id], 'default', 1);
      expect(graph1.has(cEnt!.id)).toBe(false);

      // maxHops=2 should reach C
      const graph2 = mem._store.getEntityGraph([aEnt!.id], 'default', 2);
      expect(graph2.has(cEnt!.id)).toBe(true);
    } finally {
      mem.close();
    }
  });

  it('final_score in traces is > 0 when keyword matches', () => {
    const mem = tmpEngram();
    mem.setExtractor(new SingleFactExtractor('Dave', 'plays', 'guitar'));
    try {
      mem.store('Dave plays guitar.');
      mem.retrieve('Dave guitar');
      const traces = mem._store.getTraces('default', 'Dave guitar');
      expect(traces.length).toBeGreaterThanOrEqual(1);
      expect(traces.some(t => t.finalScore > 0.0)).toBe(true);
    } finally {
      mem.close();
    }
  });

  it('getFactsForEntities returns facts for provided entity ids', () => {
    const mem = tmpEngram();
    mem.setExtractor(new SingleFactExtractor('Eve', 'speaks', 'French'));
    try {
      mem.store('Eve speaks French.');
      const eve = mem._store.findEntityByName('default', 'Eve');
      expect(eve).not.toBeNull();
      const facts = mem._store.getFactsForEntities([eve!.id], 'default');
      expect(facts.length).toBeGreaterThanOrEqual(1);
      expect(facts.some(f => f.predicate === 'speaks')).toBe(true);
    } finally {
      mem.close();
    }
  });
});
