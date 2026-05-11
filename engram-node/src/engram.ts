import { randomUUID } from 'node:crypto';
import { defaultConfig, type EngramConfig } from './config.js';
import { NullExtractor, type BaseExtractor } from './extraction/base.js';
import { IngestionPipeline } from './ingestion/pipeline.js';
import { RetrievalPipeline } from './retrieval/pipeline.js';
import { SQLiteStore } from './storage/sqliteStore.js';
import { BackgroundWorkers } from './workers/background.js';
import type { Fact } from './models/index.js';
import type {
  ContextResult,
  ExplainResult,
  ForgetResult,
  RetrievalResult,
  StoreResult,
} from './results.js';

export interface ForgetSelector {
  factId?: string;
  entityId?: string;
  episodeId?: string;
  scopeId?: string;
}

export class Engram {
  /** @internal */ readonly _store: SQLiteStore;
  private _extractor: BaseExtractor;
  private _ingestion: IngestionPipeline;
  private _retrieval: RetrievalPipeline;
  private _workers: BackgroundWorkers;
  private _config: EngramConfig;

  constructor(config?: Partial<EngramConfig>) {
    this._config = defaultConfig(config);
    this._store = new SQLiteStore(this._config.dbPath);
    this._extractor = new NullExtractor();
    this._ingestion = new IngestionPipeline(this._store, this._extractor, this._config);
    this._retrieval = new RetrievalPipeline(this._store, this._config);
    this._workers = new BackgroundWorkers(this._store, this._config);
  }

  setExtractor(extractor: BaseExtractor): void {
    this._extractor = extractor;
    this._ingestion = new IngestionPipeline(this._store, extractor, this._config);
  }

  get workers(): BackgroundWorkers {
    return this._workers;
  }

  store(
    data: string | Record<string, unknown>,
    metadata: Record<string, unknown> = {},
    scope?: string,
  ): StoreResult {
    return this._ingestion.ingest(data, metadata, scope);
  }

  retrieve(
    query: string,
    topK?: number,
    scope?: string,
    asOf?: Date | null,
  ): RetrievalResult {
    return this._retrieval.retrieve(query, topK, scope, asOf);
  }

  getContext(
    query: string,
    topK?: number,
    scope?: string,
    asOf?: Date | null,
  ): ContextResult {
    return this._retrieval.getContext(query, topK, scope, asOf);
  }

  updateFact(
    factId: string,
    newValue: string,
    validFrom?: Date,
  ): Fact {
    const existing = this._store.getFact(factId);
    if (!existing) throw new Error(`Fact not found: ${factId}`);

    const now = validFrom ?? new Date();
    this._store.supersedeFact(factId, now);

    const newFact: Fact = {
      id: randomUUID(),
      scopeId: existing.scopeId,
      subjectEntityId: existing.subjectEntityId,
      predicate: existing.predicate,
      objectEntityId: null,
      objectValueJson: { value: newValue },
      factType: existing.factType,
      validFrom: now,
      validTo: null,
      truthState: 'current',
      sourceEpisodeId: existing.sourceEpisodeId,
      confidence: existing.confidence,
      salience: existing.salience,
      strength: 1.0,
      accessCount: 0,
      lastAccessedAt: null,
      createdAt: now,
      updatedAt: now,
    };
    this._store.insertFact(newFact);
    return newFact;
  }

  forget(selector: ForgetSelector, hard = false): ForgetResult {
    if (!selector.factId && !selector.entityId && !selector.episodeId && !selector.scopeId) {
      throw new Error('forget() requires at least one selector key: factId, entityId, episodeId, or scopeId');
    }

    let affectedFacts = 0;
    let affectedEntities = 0;
    const affectedAliases = 0;

    if (selector.factId) {
      const fact = this._store.getFact(selector.factId);
      if (fact) {
        hard ? this._store.deleteFact(selector.factId) : this._store.hideFact(selector.factId);
        affectedFacts++;
      }
    }

    if (selector.entityId) {
      if (hard) {
        this._store.deleteEntityFacts(selector.entityId);
      } else {
        this._store.hideEntityFacts(selector.entityId);
      }
      affectedEntities++;
      affectedFacts++;
    }

    if (selector.episodeId) {
      if (hard) {
        this._store.deleteEpisodeFacts(selector.episodeId);
      }
      affectedFacts++;
    }

    if (selector.scopeId) {
      if (hard) {
        this._store.deleteFactsInScope(selector.scopeId);
      } else {
        this._store.hideFactsInScope(selector.scopeId);
      }
      affectedFacts++;
    }

    return { affectedFacts, affectedEntities, affectedAliases };
  }

  explain(query: string, _resultId?: string, scope?: string): ExplainResult {
    const scopeId = scope ?? this._config.defaultScope;
    const traces = this._store.getTraces(scopeId, query);
    const quarantinedFacts = this._store.getPendingQuarantinedFacts(scopeId);

    const qCount = quarantinedFacts.length;
    const summary = qCount > 0
      ? `Query matched ${traces.length} trace(s). Quarantine pool has ${qCount} pending fact(s).`
      : `Query matched ${traces.length} trace(s). No quarantined facts.`;

    return { traces, quarantinedFacts, summary };
  }

  close(): void {
    this._store.close();
  }

  [Symbol.dispose](): void {
    this.close();
  }
}
