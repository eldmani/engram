// Main class
export { Engram } from './engram.js';
export type { ForgetSelector } from './engram.js';

// Config
export { defaultConfig } from './config.js';
export type { EngramConfig } from './config.js';

// Extraction
export { BaseExtractor, NullExtractor } from './extraction/base.js';
export type { ExtractedEntity, ExtractedFact, ExtractionResult } from './extraction/base.js';

// Models
export type { Alias, Entity, Episode, Fact, QuarantinedFact, RetrievalTrace } from './models/index.js';

// Results
export type {
  StoreResult,
  RetrievalResult,
  ContextResult,
  ForgetResult,
  ExplainResult,
  FactView,
  EntityView,
  TimelineEvent,
  EpisodeView,
  EvidenceView,
} from './results.js';

// Storage (advanced usage)
export { SQLiteStore } from './storage/sqliteStore.js';

// Workers
export { BackgroundWorkers } from './workers/background.js';
