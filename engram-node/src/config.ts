export interface EngramConfig {
  dbPath: string;
  defaultScope: string;

  // Confidence thresholds
  extractorConfidenceMin: number;
  resolutionConfidenceMin: number;
  supersessionConfidenceMin: number;

  // Retrieval
  defaultTopK: number;

  // Hybrid scoring weights
  // NOTE: weightSemantic is reserved for a future vector embedder.
  // Until setEmbedder() is called, semantic score is always 0.0 and
  // keyword scoring carries the signal for that weight's share.
  weightSemantic: number;
  weightKeyword: number;
  weightGraph: number;
  weightTemporal: number;
  weightSalience: number;
  weightStrength: number;

  // Memory dynamics
  reinforcementBoost: number;
  stableAccessThreshold: number;
  decayIntervalHours: number;
  heatPromotionThreshold: number;
}

export function defaultConfig(overrides?: Partial<EngramConfig>): EngramConfig {
  return {
    dbPath: 'engram.db',
    defaultScope: 'default',
    extractorConfidenceMin: 0.7,
    resolutionConfidenceMin: 0.75,
    supersessionConfidenceMin: 0.85,
    defaultTopK: 10,
    weightSemantic: 0.40,
    weightKeyword: 0.30,
    weightGraph: 0.10,
    weightTemporal: 0.10,
    weightSalience: 0.05,
    weightStrength: 0.05,
    reinforcementBoost: 0.05,
    stableAccessThreshold: 5,
    decayIntervalHours: 24.0,
    heatPromotionThreshold: 10.0,
    ...overrides,
  };
}
