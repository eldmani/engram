export interface ExtractedEntity {
  name: string;
  type: string;
  confidence: number;
}

export interface ExtractedFact {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  factType?: string;
}

export interface ExtractionResult {
  entities: ExtractedEntity[];
  facts: ExtractedFact[];
  confidence: number;
}

export abstract class BaseExtractor {
  abstract extract(data: string | Record<string, unknown>): ExtractionResult;
}

export class NullExtractor extends BaseExtractor {
  extract(_data: string | Record<string, unknown>): ExtractionResult {
    return { entities: [], facts: [], confidence: 1.0 };
  }
}
