import { DatabaseSync } from 'node:sqlite';
import type { Alias, Entity, Episode, Fact, QuarantinedFact, RetrievalTrace } from '../models/index.js';

const SCHEMA_VERSION = 1;

/** Remove FTS5 special characters to prevent OperationalError on malformed queries. */
function sanitizeFts(query: string): string {
  const sanitized = query.replace(/[^\w\s]/gu, ' ').trim();
  return sanitized.length > 0 ? sanitized : '""';
}

const SCHEMA = `
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS episodes (
  id          TEXT PRIMARY KEY,
  scope_id    TEXT NOT NULL,
  source      TEXT NOT NULL,
  raw_text    TEXT NOT NULL,
  metadata    TEXT NOT NULL DEFAULT '{}',
  created_at  TEXT NOT NULL,
  checksum    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS episode_scope_time_idx ON episodes (scope_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes USING fts5(
  episode_id,
  raw_text,
  scope_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS entities (
  id              TEXT PRIMARY KEY,
  scope_id        TEXT NOT NULL,
  type            TEXT NOT NULL DEFAULT 'unknown',
  canonical_name  TEXT NOT NULL,
  summary         TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  first_seen_at   TEXT NOT NULL,
  last_seen_at    TEXT NOT NULL,
  salience        REAL NOT NULL DEFAULT 0.5,
  strength        REAL NOT NULL DEFAULT 1.0,
  confidence      REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS entity_scope_idx ON entities (scope_id);
CREATE INDEX IF NOT EXISTS entity_name_idx  ON entities (scope_id, canonical_name);

CREATE TABLE IF NOT EXISTS aliases (
  id                TEXT PRIMARY KEY,
  entity_id         TEXT NOT NULL REFERENCES entities(id),
  value             TEXT NOT NULL,
  normalized_value  TEXT NOT NULL,
  embedding_id      TEXT,
  source_episode_id TEXT NOT NULL REFERENCES episodes(id),
  confidence        REAL NOT NULL DEFAULT 1.0,
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alias_norm_idx   ON aliases (normalized_value);
CREATE INDEX IF NOT EXISTS alias_entity_idx ON aliases (entity_id);

CREATE TABLE IF NOT EXISTS facts (
  id                TEXT PRIMARY KEY,
  scope_id          TEXT NOT NULL,
  subject_entity_id TEXT NOT NULL REFERENCES entities(id),
  predicate         TEXT NOT NULL,
  object_entity_id  TEXT,
  object_value_json TEXT,
  fact_type         TEXT NOT NULL DEFAULT 'assertion',
  valid_from        TEXT NOT NULL,
  valid_to          TEXT,
  truth_state       TEXT NOT NULL DEFAULT 'current',
  source_episode_id TEXT NOT NULL REFERENCES episodes(id),
  confidence        REAL NOT NULL DEFAULT 1.0,
  salience          REAL NOT NULL DEFAULT 0.5,
  strength          REAL NOT NULL DEFAULT 1.0,
  access_count      INTEGER NOT NULL DEFAULT 0,
  last_accessed_at  TEXT,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fact_subject_pred_idx ON facts (subject_entity_id, predicate);
CREATE INDEX IF NOT EXISTS fact_validity_idx     ON facts (valid_from, valid_to);
CREATE INDEX IF NOT EXISTS fact_truth_state_idx  ON facts (truth_state);
CREATE INDEX IF NOT EXISTS fact_scope_idx        ON facts (scope_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(
  fact_id,
  predicate,
  object_text,
  scope_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS quarantined_facts (
  id                            TEXT PRIMARY KEY,
  scope_id                      TEXT NOT NULL,
  source_episode_id             TEXT NOT NULL REFERENCES episodes(id),
  extracted_subject             TEXT NOT NULL,
  extracted_predicate           TEXT NOT NULL,
  extracted_object              TEXT NOT NULL,
  candidate_supersedes_fact_id  TEXT,
  extractor_confidence          REAL NOT NULL,
  resolution_confidence         REAL NOT NULL,
  reason                        TEXT NOT NULL,
  status                        TEXT NOT NULL DEFAULT 'pending',
  created_at                    TEXT NOT NULL,
  reviewed_at                   TEXT
);
CREATE INDEX IF NOT EXISTS quarantine_scope_status_idx ON quarantined_facts (scope_id, status);

CREATE TABLE IF NOT EXISTS retrieval_traces (
  id                TEXT PRIMARY KEY,
  query             TEXT NOT NULL,
  scope_id          TEXT NOT NULL,
  candidate_fact_id TEXT,
  semantic_score    REAL NOT NULL DEFAULT 0.0,
  keyword_score     REAL NOT NULL DEFAULT 0.0,
  graph_score       REAL NOT NULL DEFAULT 0.0,
  temporal_score    REAL NOT NULL DEFAULT 0.0,
  final_score       REAL NOT NULL DEFAULT 0.0,
  matched_entities  TEXT NOT NULL DEFAULT '[]',
  source_episode_ids TEXT NOT NULL DEFAULT '[]',
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trace_scope_idx ON retrieval_traces (scope_id, created_at);

CREATE TABLE IF NOT EXISTS embeddings (
  id          TEXT PRIMARY KEY,
  ref_type    TEXT NOT NULL,
  ref_id      TEXT NOT NULL,
  vector      BLOB NOT NULL,
  model       TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS embedding_ref_idx ON embeddings (ref_type, ref_id);
`;

function ts(d: Date | null | undefined): string | null {
  return d ? d.toISOString() : null;
}

function dt(s: string | null | undefined): Date | null {
  return s ? new Date(s) : null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Row = Record<string, any>;

export class SQLiteStore {
  private db: DatabaseSync;

  constructor(dbPath: string = 'engram.db') {
    this.db = new DatabaseSync(dbPath);
    this.db.exec(SCHEMA);
    this.db.exec(`PRAGMA user_version = ${SCHEMA_VERSION}`);
  }

  close(): void {
    this.db.close();
  }

  private transaction(fn: () => void): void {
    this.db.exec('BEGIN');
    try {
      fn();
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  // ------------------------------------------------------------------
  // Episodes
  // ------------------------------------------------------------------

  insertEpisode(ep: Episode): void {
    this.db.prepare(
      'INSERT INTO episodes VALUES (?,?,?,?,?,?,?)',
    ).run(ep.id, ep.scopeId, ep.source, ep.rawText,
          JSON.stringify(ep.metadata), ts(ep.createdAt), ep.checksum);
    this.db.prepare(
      'INSERT INTO fts_episodes VALUES (?,?,?)',
    ).run(ep.id, ep.rawText, ep.scopeId);
  }

  getEpisode(id: string): Episode | null {
    const row = this.db.prepare('SELECT * FROM episodes WHERE id=?').get(id) as Row | undefined;
    return row ? this.rowToEpisode(row) : null;
  }

  ftsSearchEpisodes(query: string, scopeId: string, limit: number): Array<{ episode: Episode; score: number }> {
    const safeQuery = sanitizeFts(query);
    try {
      const rows = this.db.prepare(`
        SELECT e.*, -fe.rank AS kw_score
        FROM fts_episodes fe
        JOIN episodes e ON e.id = fe.episode_id
        WHERE fts_episodes MATCH ? AND fe.scope_id = ?
        ORDER BY fe.rank
        LIMIT ?
      `).all(safeQuery, scopeId, limit) as Row[];
      return rows.map(r => ({ episode: this.rowToEpisode(r), score: r.kw_score as number }));
    } catch {
      return [];
    }
  }

  getRecentEpisodes(scopeId: string, limit: number): Episode[] {
    const rows = this.db.prepare(
      'SELECT * FROM episodes WHERE scope_id=? ORDER BY created_at DESC LIMIT ?',
    ).all(scopeId, limit) as Row[];
    return rows.map(r => this.rowToEpisode(r));
  }

  private rowToEpisode(row: Row): Episode {
    return {
      id: row.id as string,
      scopeId: row.scope_id as string,
      source: row.source as string,
      rawText: row.raw_text as string,
      metadata: JSON.parse(row.metadata as string) as Record<string, unknown>,
      createdAt: new Date(row.created_at as string),
      checksum: row.checksum as string,
    };
  }

  // ------------------------------------------------------------------
  // Entities
  // ------------------------------------------------------------------

  insertEntity(entity: Entity): void {
    this.db.prepare(
      'INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
    ).run(entity.id, entity.scopeId, entity.type, entity.canonicalName,
          entity.summary, ts(entity.createdAt), ts(entity.updatedAt),
          ts(entity.firstSeenAt), ts(entity.lastSeenAt),
          entity.salience, entity.strength, entity.confidence);
  }

  getEntity(id: string): Entity | null {
    const row = this.db.prepare('SELECT * FROM entities WHERE id=?').get(id) as Row | undefined;
    return row ? this.rowToEntity(row) : null;
  }

  findEntityByName(scopeId: string, name: string): Entity | null {
    const row = this.db.prepare(
      'SELECT * FROM entities WHERE scope_id=? AND canonical_name=? COLLATE NOCASE',
    ).get(scopeId, name) as Row | undefined;
    return row ? this.rowToEntity(row) : null;
  }

  findEntityByAlias(normalizedValue: string, scopeId: string): Entity | null {
    const row = this.db.prepare(`
      SELECT e.* FROM aliases a
      JOIN entities e ON e.id = a.entity_id
      WHERE a.normalized_value=? AND e.scope_id=?
      LIMIT 1
    `).get(normalizedValue, scopeId) as Row | undefined;
    return row ? this.rowToEntity(row) : null;
  }

  updateEntitySeen(id: string, when: Date): void {
    this.db.prepare(
      'UPDATE entities SET last_seen_at=?, updated_at=? WHERE id=?',
    ).run(ts(when), ts(when), id);
  }

  getEntitiesForScope(scopeId: string): Entity[] {
    const rows = this.db.prepare(
      'SELECT * FROM entities WHERE scope_id=? ORDER BY salience DESC',
    ).all(scopeId) as Row[];
    return rows.map(r => this.rowToEntity(r));
  }

  private rowToEntity(row: Row): Entity {
    return {
      id: row.id as string,
      scopeId: row.scope_id as string,
      type: row.type as string,
      canonicalName: row.canonical_name as string,
      summary: row.summary as string | null,
      createdAt: new Date(row.created_at as string),
      updatedAt: new Date(row.updated_at as string),
      firstSeenAt: new Date(row.first_seen_at as string),
      lastSeenAt: new Date(row.last_seen_at as string),
      salience: row.salience as number,
      strength: row.strength as number,
      confidence: row.confidence as number,
    };
  }

  // ------------------------------------------------------------------
  // Aliases
  // ------------------------------------------------------------------

  insertAlias(alias: Alias): void {
    this.db.prepare(
      'INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?,?,?,?)',
    ).run(alias.id, alias.entityId, alias.value, alias.normalizedValue,
          alias.embeddingId, alias.sourceEpisodeId, alias.confidence, ts(alias.createdAt));
  }

  // ------------------------------------------------------------------
  // Facts
  // ------------------------------------------------------------------

  insertFact(fact: Fact): void {
    const objText = fact.objectValueJson
      ? JSON.stringify(fact.objectValueJson)
      : fact.objectEntityId ?? '';

    this.db.prepare(
      'INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
    ).run(fact.id, fact.scopeId, fact.subjectEntityId, fact.predicate,
          fact.objectEntityId,
          fact.objectValueJson ? JSON.stringify(fact.objectValueJson) : null,
          fact.factType, ts(fact.validFrom), ts(fact.validTo),
          fact.truthState, fact.sourceEpisodeId,
          fact.confidence, fact.salience, fact.strength,
          fact.accessCount, ts(fact.lastAccessedAt),
          ts(fact.createdAt), ts(fact.updatedAt));
    this.db.prepare(
      'INSERT INTO fts_facts VALUES (?,?,?,?)',
    ).run(fact.id, fact.predicate, objText, fact.scopeId);
  }

  getFact(id: string): Fact | null {
    const row = this.db.prepare('SELECT * FROM facts WHERE id=?').get(id) as Row | undefined;
    return row ? this.rowToFact(row) : null;
  }

  getCurrentFacts(scopeId: string, subjectEntityId: string, predicate: string): Fact[] {
    const rows = this.db.prepare(`
      SELECT * FROM facts
      WHERE scope_id=? AND subject_entity_id=? AND predicate=?
        AND truth_state='current' AND valid_to IS NULL
    `).all(scopeId, subjectEntityId, predicate) as Row[];
    return rows.map(r => this.rowToFact(r));
  }

  supersedeFact(id: string, validTo: Date): void {
    const now = ts(new Date());
    this.db.prepare(
      "UPDATE facts SET valid_to=?, truth_state='superseded', updated_at=? WHERE id=?",
    ).run(ts(validTo), now, id);
  }

  hideFact(id: string): void {
    const now = ts(new Date());
    this.db.prepare(
      "UPDATE facts SET truth_state='hidden', updated_at=? WHERE id=?",
    ).run(now, id);
  }

  deleteFact(id: string): void {
    this.db.prepare('DELETE FROM fts_facts WHERE fact_id=?').run(id);
    this.db.prepare('DELETE FROM facts WHERE id=?').run(id);
  }

  ftsSearchFacts(
    query: string,
    scopeId: string,
    limit: number,
    asOf?: Date | null,
  ): Array<{ fact: Fact; score: number }> {
    const safeQuery = sanitizeFts(query);
    try {
      const rows = this.db.prepare(`
        SELECT f.*, -ff.rank AS kw_score
        FROM fts_facts ff
        JOIN facts f ON f.id = ff.fact_id
        WHERE fts_facts MATCH ? AND ff.scope_id = ?
          AND f.truth_state != 'hidden'
        ORDER BY ff.rank
        LIMIT ?
      `).all(safeQuery, scopeId, limit) as Row[];

      return rows
        .map(r => ({ fact: this.rowToFact(r), score: r.kw_score as number }))
        .filter(({ fact }) => {
          if (!asOf) return true;
          if (fact.validFrom > asOf) return false;
          if (fact.validTo && fact.validTo <= asOf) return false;
          return true;
        });
    } catch {
      return [];
    }
  }

  getCurrentFactsForScope(scopeId: string, limit: number): Fact[] {
    const rows = this.db.prepare(`
      SELECT * FROM facts
      WHERE scope_id=? AND truth_state='current'
      ORDER BY salience DESC, strength DESC
      LIMIT ?
    `).all(scopeId, limit) as Row[];
    return rows.map(r => this.rowToFact(r));
  }

  reinforceFact(id: string, boost: number): void {
    const now = ts(new Date());
    this.db.prepare(`
      UPDATE facts
      SET access_count = access_count + 1,
          last_accessed_at = ?,
          strength = MIN(strength + ?, 5.0),
          updated_at = ?
      WHERE id=?
    `).run(now, boost, now, id);
  }

  applyDecay(scopeId: string, stableThreshold: number): void {
    const now = new Date();
    const rows = this.db.prepare(
      'SELECT id, strength, access_count, updated_at FROM facts WHERE scope_id=?',
    ).all(scopeId) as Row[];

    const update = this.db.prepare(
      'UPDATE facts SET strength=?, updated_at=? WHERE id=?',
    );
    this.transaction(() => {
      for (const r of rows) {
        const ageHours = Math.max(
          1,
          (now.getTime() - new Date(r.updated_at as string).getTime()) / 3_600_000,
        );
        let newStrength: number;
        if ((r.access_count as number) < stableThreshold) {
          newStrength = Math.max(0.01, (r.strength as number) / Math.sqrt(ageHours / 24 + 1));
        } else {
          newStrength = Math.max(0.01, (r.strength as number) - 0.001 * ageHours);
        }
        update.run(newStrength, ts(now), r.id as string);
      }
    });
  }

  hideEntityFacts(entityId: string): void {
    const now = ts(new Date());
    this.db.prepare(
      "UPDATE facts SET truth_state='hidden', updated_at=? WHERE subject_entity_id=?",
    ).run(now, entityId);
  }

  deleteEntityFacts(entityId: string): void {
    const rows = this.db.prepare(
      'SELECT id FROM facts WHERE subject_entity_id=?',
    ).all(entityId) as Row[];
    for (const r of rows) this.deleteFact(r.id as string);
  }

  deleteEpisodeFacts(episodeId: string): void {
    const rows = this.db.prepare(
      'SELECT id FROM facts WHERE source_episode_id=?',
    ).all(episodeId) as Row[];
    for (const r of rows) this.deleteFact(r.id as string);
  }

  // ------------------------------------------------------------------
  // Scope-level operations
  // ------------------------------------------------------------------

  hideFactsInScope(scopeId: string): void {
    const now = ts(new Date());
    this.db.prepare(
      "UPDATE facts SET truth_state='hidden', updated_at=? WHERE scope_id=?",
    ).run(now, scopeId);
  }

  deleteFactsInScope(scopeId: string): void {
    const rows = this.db.prepare(
      'SELECT id FROM facts WHERE scope_id=?',
    ).all(scopeId) as Row[];
    for (const r of rows) this.deleteFact(r.id as string);
  }

  private rowToFact(row: Row): Fact {
    let objVal: Record<string, unknown> | null = null;
    if (row.object_value_json) {
      try { objVal = JSON.parse(row.object_value_json as string) as Record<string, unknown>; }
      catch { objVal = { value: row.object_value_json }; }
    }
    return {
      id: row.id as string,
      scopeId: row.scope_id as string,
      subjectEntityId: row.subject_entity_id as string,
      predicate: row.predicate as string,
      objectEntityId: row.object_entity_id as string | null,
      objectValueJson: objVal,
      factType: row.fact_type as string,
      validFrom: new Date(row.valid_from as string),
      validTo: dt(row.valid_to as string | null),
      truthState: row.truth_state as string,
      sourceEpisodeId: row.source_episode_id as string,
      confidence: row.confidence as number,
      salience: row.salience as number,
      strength: row.strength as number,
      accessCount: row.access_count as number,
      lastAccessedAt: dt(row.last_accessed_at as string | null),
      createdAt: new Date(row.created_at as string),
      updatedAt: new Date(row.updated_at as string),
    };
  }

  // ------------------------------------------------------------------
  // Quarantined Facts
  // ------------------------------------------------------------------

  insertQuarantinedFact(qf: QuarantinedFact): void {
    this.db.prepare(
      'INSERT INTO quarantined_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
    ).run(qf.id, qf.scopeId, qf.sourceEpisodeId,
          qf.extractedSubject, qf.extractedPredicate, qf.extractedObject,
          qf.candidateSuperseedsFactId,
          qf.extractorConfidence, qf.resolutionConfidence,
          qf.reason, qf.status, ts(qf.createdAt), ts(qf.reviewedAt));
  }

  getPendingQuarantinedFacts(scopeId: string): QuarantinedFact[] {
    const rows = this.db.prepare(
      "SELECT * FROM quarantined_facts WHERE scope_id=? AND status='pending'",
    ).all(scopeId) as Row[];
    return rows.map(r => this.rowToQF(r));
  }

  updateQuarantineStatus(id: string, status: string): void {
    const now = ts(new Date());
    this.db.prepare(
      'UPDATE quarantined_facts SET status=?, reviewed_at=? WHERE id=?',
    ).run(status, now, id);
  }

  private rowToQF(row: Row): QuarantinedFact {
    return {
      id: row.id as string,
      scopeId: row.scope_id as string,
      sourceEpisodeId: row.source_episode_id as string,
      extractedSubject: row.extracted_subject as string,
      extractedPredicate: row.extracted_predicate as string,
      extractedObject: row.extracted_object as string,
      candidateSuperseedsFactId: row.candidate_supersedes_fact_id as string | null,
      extractorConfidence: row.extractor_confidence as number,
      resolutionConfidence: row.resolution_confidence as number,
      reason: row.reason as string,
      status: row.status as string,
      createdAt: new Date(row.created_at as string),
      reviewedAt: dt(row.reviewed_at as string | null),
    };
  }

  // ------------------------------------------------------------------
  // Retrieval Traces
  // ------------------------------------------------------------------

  insertRetrievalTrace(trace: RetrievalTrace): void {
    this.db.prepare(
      'INSERT INTO retrieval_traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
    ).run(trace.id, trace.query, trace.scopeId, trace.candidateFactId,
          trace.semanticScore, trace.keywordScore,
          trace.graphScore, trace.temporalScore, trace.finalScore,
          JSON.stringify(trace.matchedEntities),
          JSON.stringify(trace.sourceEpisodeIds),
          ts(trace.createdAt));
  }

  getTraces(scopeId: string, query: string): RetrievalTrace[] {
    const rows = this.db.prepare(`
      SELECT * FROM retrieval_traces
      WHERE scope_id=? AND query=?
      ORDER BY created_at DESC LIMIT 50
    `).all(scopeId, query) as Row[];
    return rows.map(r => ({
      id: r.id as string,
      query: r.query as string,
      scopeId: r.scope_id as string,
      candidateFactId: r.candidate_fact_id as string | null,
      semanticScore: r.semantic_score as number,
      keywordScore: r.keyword_score as number,
      graphScore: r.graph_score as number,
      temporalScore: r.temporal_score as number,
      finalScore: r.final_score as number,
      matchedEntities: JSON.parse(r.matched_entities as string) as string[],
      sourceEpisodeIds: JSON.parse(r.source_episode_ids as string) as string[],
      createdAt: new Date(r.created_at as string),
    }));
  }
}
