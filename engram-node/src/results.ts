import type { Entity, Episode, Fact, QuarantinedFact, RetrievalTrace } from './models/index.js';

export interface StoreResult {
  episodeId: string;
  entityIds: string[];
  factIds: string[];
  quarantinedCount: number;
}

export interface FactView {
  factId: string;
  subject: string;
  predicate: string;
  object: string;
  validFrom: Date;
  validTo: Date | null;
  truthState: string;
  confidence: number;
  salience: number;
  score: number;
}

export interface EntityView {
  entityId: string;
  canonicalName: string;
  type: string;
  summary: string | null;
  salience: number;
}

export interface TimelineEvent {
  timestamp: Date;
  description: string;
  factId?: string;
  episodeId?: string;
}

export interface EpisodeView {
  episodeId: string;
  rawText: string;
  createdAt: Date;
  source: string;
  score: number;
}

export interface EvidenceView {
  episodeIds: string[];
  description: string;
  factId?: string;
}

export interface RetrievalResult {
  facts: Fact[];
  entities: Entity[];
  episodes: Episode[];
  scores: Record<string, number>;
  traceIds: string[];
}

export interface ContextResult {
  factsBlock: FactView[];
  entitiesBlock: EntityView[];
  timelineBlock: TimelineEvent[];
  episodesBlock: EpisodeView[];
  evidenceBlock: EvidenceView[];
  confidence: number;
  formatted: string;
}

export interface ForgetResult {
  affectedFacts: number;
  affectedEntities: number;
  affectedAliases: number;
}

export interface ExplainResult {
  traces: RetrievalTrace[];
  quarantinedFacts: QuarantinedFact[];
  summary: string;
}
