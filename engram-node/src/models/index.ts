export interface Episode {
  id: string;
  scopeId: string;
  source: string;
  rawText: string;
  metadata: Record<string, unknown>;
  createdAt: Date;
  checksum: string;
}

export interface Entity {
  id: string;
  scopeId: string;
  type: string;
  canonicalName: string;
  summary: string | null;
  createdAt: Date;
  updatedAt: Date;
  firstSeenAt: Date;
  lastSeenAt: Date;
  salience: number;
  strength: number;
  confidence: number;
}

export interface Alias {
  id: string;
  entityId: string;
  value: string;
  normalizedValue: string;
  embeddingId: string | null;
  sourceEpisodeId: string;
  confidence: number;
  createdAt: Date;
}

export interface Fact {
  id: string;
  scopeId: string;
  subjectEntityId: string;
  predicate: string;
  objectEntityId: string | null;
  objectValueJson: Record<string, unknown> | null;
  factType: string;
  validFrom: Date;
  validTo: Date | null;
  /** current | superseded | disputed | hidden */
  truthState: string;
  sourceEpisodeId: string;
  confidence: number;
  salience: number;
  strength: number;
  accessCount: number;
  lastAccessedAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface QuarantinedFact {
  id: string;
  scopeId: string;
  sourceEpisodeId: string;
  extractedSubject: string;
  extractedPredicate: string;
  extractedObject: string;
  candidateSuperseedsFactId: string | null;
  extractorConfidence: number;
  resolutionConfidence: number;
  reason: string;
  /** pending | approved | rejected */
  status: string;
  createdAt: Date;
  reviewedAt: Date | null;
}

export interface RetrievalTrace {
  id: string;
  query: string;
  scopeId: string;
  candidateFactId: string | null;
  semanticScore: number;
  keywordScore: number;
  graphScore: number;
  temporalScore: number;
  finalScore: number;
  matchedEntities: string[];
  sourceEpisodeIds: string[];
  createdAt: Date;
}
