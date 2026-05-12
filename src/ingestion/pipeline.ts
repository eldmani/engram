import { createHash, randomUUID } from 'node:crypto';
import type { EngramConfig } from '../config.js';
import type { BaseExtractor } from '../extraction/base.js';
import type { Alias, Entity, Episode, Fact, QuarantinedFact } from '../models/index.js';
import type { SQLiteStore } from '../storage/sqliteStore.js';
import type { StoreResult } from '../results.js';

function normalize(s: string): string {
  return s.toLowerCase().trim().replace(/\s+/g, ' ');
}

export class IngestionPipeline {
  constructor(
    private store: SQLiteStore,
    private extractor: BaseExtractor,
    private config: EngramConfig,
  ) {}

  ingest(
    data: string | Record<string, unknown>,
    metadata: Record<string, unknown> = {},
    scope?: string,
  ): StoreResult {
    const scopeId = scope ?? this.config.defaultScope;
    const now = new Date();

    // Step 1: Normalize raw text
    const rawText = typeof data === 'string' ? data : JSON.stringify(data);
    const checksum = createHash('sha256').update(rawText).digest('hex');

    // Step 2: Create episode
    const episodeId = randomUUID();
    const episode: Episode = {
      id: episodeId,
      scopeId,
      source: (metadata['source'] as string | undefined) ?? 'user',
      rawText,
      metadata,
      createdAt: now,
      checksum,
    };
    this.store.insertEpisode(episode);

    // Step 3: Extract
    const extraction = this.extractor.extract(data);

    if (!extraction.entities.length && !extraction.facts.length) {
      return { episodeId, entityIds: [], factIds: [], quarantinedCount: 0 };
    }

    const entityIds: string[] = [];
    const factIds: string[] = [];
    let quarantinedCount = 0;

    // Step 4: Resolve entities
    const entityMap = new Map<string, Entity>(); // normalized name → entity
    for (const ext of extraction.entities) {
      const normName = normalize(ext.name);
      let entity = this.store.findEntityByAlias(normName, scopeId)
        ?? this.store.findEntityByName(scopeId, ext.name);

      if (!entity) {
        entity = {
          id: randomUUID(),
          scopeId,
          type: ext.type,
          canonicalName: ext.name,
          summary: null,
          createdAt: now,
          updatedAt: now,
          firstSeenAt: now,
          lastSeenAt: now,
          salience: 0.5,
          strength: 1.0,
          confidence: ext.confidence,
        };
        this.store.insertEntity(entity);

        const alias: Alias = {
          id: randomUUID(),
          entityId: entity.id,
          value: ext.name,
          normalizedValue: normName,
          embeddingId: null,
          sourceEpisodeId: episodeId,
          confidence: ext.confidence,
          createdAt: now,
        };
        this.store.insertAlias(alias);
      } else {
        this.store.updateEntitySeen(entity.id, now);
      }

      entityMap.set(normName, entity);
      if (!entityIds.includes(entity.id)) entityIds.push(entity.id);
    }

    // Steps 5-10: Process facts
    for (const ef of extraction.facts) {
      const normSubject = normalize(ef.subject);
      const subjectEntity = entityMap.get(normSubject)
        ?? this.store.findEntityByAlias(normSubject, scopeId)
        ?? this.store.findEntityByName(scopeId, ef.subject);

      if (!subjectEntity) continue;

      // Step 5: Check existing facts (supersession detection)
      const existing = this.store.getCurrentFacts(scopeId, subjectEntity.id, ef.predicate);
      const isSupersession = existing.length > 0;

      // Step 6: Salience
      const importancePriors: Record<string, number> = {
        prefers: 0.2, likes: 0.15, works_at: 0.3, lives_in: 0.2,
      };
      const novelty = isSupersession ? 0.0 : 0.5;
      const importance = importancePriors[ef.predicate] ?? 0.0;
      const salience = 0.5 * (1 + novelty) * (1 + importance);

      // Step 7: Confidence gate
      const extConf = extraction.confidence * ef.confidence;
      const threshold = isSupersession
        ? this.config.supersessionConfidenceMin
        : this.config.extractorConfidenceMin;

      if (extConf < threshold) {
        // Quarantine
        const qf: QuarantinedFact = {
          id: randomUUID(),
          scopeId,
          sourceEpisodeId: episodeId,
          extractedSubject: ef.subject,
          extractedPredicate: ef.predicate,
          extractedObject: ef.object,
          candidateSuperseedsFactId: existing[0]?.id ?? null,
          extractorConfidence: ef.confidence,
          resolutionConfidence: extraction.confidence,
          reason: `Combined confidence ${extConf.toFixed(3)} < threshold ${threshold}`,
          status: 'pending',
          createdAt: now,
          reviewedAt: null,
        };
        this.store.insertQuarantinedFact(qf);
        quarantinedCount++;
        continue;
      }

      // Step 8: Supersede existing
      if (isSupersession) {
        for (const old of existing) {
          this.store.supersedeFact(old.id, now);
        }
      }

      // Resolve object entity for graph traversal
      const normObject = normalize(ef.object);
      const objectEntity = entityMap.get(normObject)
        ?? this.store.findEntityByAlias(normObject, scopeId)
        ?? this.store.findEntityByName(scopeId, ef.object);

      // Step 9: Write fact
      const fact: Fact = {
        id: randomUUID(),
        scopeId,
        subjectEntityId: subjectEntity.id,
        predicate: ef.predicate,
        objectEntityId: objectEntity?.id ?? null,
        objectValueJson: { value: ef.object },
        factType: ef.factType ?? 'assertion',
        validFrom: now,
        validTo: null,
        truthState: 'current',
        sourceEpisodeId: episodeId,
        confidence: extConf,
        salience,
        strength: 1.0,
        accessCount: 0,
        lastAccessedAt: null,
        createdAt: now,
        updatedAt: now,
      };
      this.store.insertFact(fact);
      factIds.push(fact.id);
    }

    return { episodeId, entityIds, factIds, quarantinedCount };
  }
}
