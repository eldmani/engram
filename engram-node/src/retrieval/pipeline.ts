import { randomUUID } from 'node:crypto';
import type { EngramConfig } from '../config.js';
import type { SQLiteStore } from '../storage/sqliteStore.js';
import type {
  ContextResult,
  EntityView,
  EpisodeView,
  EvidenceView,
  FactView,
  RetrievalResult,
  TimelineEvent,
} from '../results.js';
import type { Fact } from '../models/index.js';

function temporalScore(fact: Fact): number {
  const ageDays = (Date.now() - fact.createdAt.getTime()) / 86_400_000;
  return 1.0 / (1.0 + ageDays);
}

export class RetrievalPipeline {
  constructor(
    private store: SQLiteStore,
    private config: EngramConfig,
  ) {}

  retrieve(
    query: string,
    topK?: number,
    scope?: string,
    asOf?: Date | null,
  ): RetrievalResult {
    const scopeId = scope ?? this.config.defaultScope;
    const k = topK ?? this.config.defaultTopK;

    // FTS on episodes
    const epResults = this.store.ftsSearchEpisodes(query, scopeId, k * 2);

    // FTS on facts
    const ftsFacts = this.store.ftsSearchFacts(query, scopeId, k * 2, asOf);

    // Current facts for scope
    const currentFacts = this.store.getCurrentFactsForScope(scopeId, k * 2);

    // Merge and score
    const scoreMap = new Map<string, number>();
    const factMap = new Map<string, Fact>();

    for (const { fact, score } of ftsFacts) {
      factMap.set(fact.id, fact);
      scoreMap.set(fact.id, score * this.config.weightKeyword);
    }

    for (const fact of currentFacts) {
      if (!factMap.has(fact.id)) {
        factMap.set(fact.id, fact);
        scoreMap.set(fact.id, 0.0);
      }
      const prev = scoreMap.get(fact.id) ?? 0;
      const tScore = temporalScore(fact) * this.config.weightTemporal;
      const sScore = fact.salience * this.config.weightSalience;
      const strScore = fact.strength * this.config.weightStrength;
      scoreMap.set(fact.id, prev + tScore + sScore + strScore);
    }

    // Sort
    const ranked = Array.from(factMap.values())
      .sort((a, b) => (scoreMap.get(b.id) ?? 0) - (scoreMap.get(a.id) ?? 0))
      .slice(0, k);

    const scores: Record<string, number> = {};
    for (const f of ranked) scores[f.id] = scoreMap.get(f.id) ?? 0;

    // Collect entities
    const entityIds = new Set(ranked.map(f => f.subjectEntityId));
    const entities = Array.from(entityIds)
      .map(id => this.store.getEntity(id))
      .filter((e): e is NonNullable<typeof e> => e !== null);

    // Episodes
    const epScoreMap = new Map<string, number>(
      epResults.map(({ episode, score }) => [episode.id, score]),
    );
    const episodeIds = new Set(ranked.map(f => f.sourceEpisodeId));
    for (const { episode } of epResults) episodeIds.add(episode.id);
    const episodes = Array.from(episodeIds)
      .map(id => this.store.getEpisode(id))
      .filter((e): e is NonNullable<typeof e> => e !== null);

    // Persist traces + reinforce
    const now = new Date();
    const traceIds: string[] = [];
    for (const fact of ranked) {
      const traceId = randomUUID();
      this.store.insertRetrievalTrace({
        id: traceId,
        query,
        scopeId,
        candidateFactId: fact.id,
        semanticScore: 0.0,
        keywordScore: ftsFacts.find(r => r.fact.id === fact.id)?.score ?? 0,
        graphScore: 0.0,
        temporalScore: temporalScore(fact),
        finalScore: scoreMap.get(fact.id) ?? 0,
        matchedEntities: [fact.subjectEntityId],
        sourceEpisodeIds: [fact.sourceEpisodeId],
        createdAt: now,
      });
      traceIds.push(traceId);
      this.store.reinforceFact(fact.id, this.config.reinforcementBoost);
    }

    return { facts: ranked, entities, episodes, scores, traceIds };
  }

  getContext(
    query: string,
    topK?: number,
    scope?: string,
    asOf?: Date | null,
  ): ContextResult {
    const result = this.retrieve(query, topK, scope, asOf);
    const { facts, entities, episodes, scores } = result;

    const factsBlock: FactView[] = facts.map(f => {
      const obj = f.objectValueJson
        ? String((f.objectValueJson as { value?: unknown }).value ?? JSON.stringify(f.objectValueJson))
        : f.objectEntityId ?? '';
      const subjectEntity = entities.find(e => e.id === f.subjectEntityId);
      return {
        factId: f.id,
        subject: subjectEntity?.canonicalName ?? f.subjectEntityId,
        predicate: f.predicate,
        object: obj,
        validFrom: f.validFrom,
        validTo: f.validTo,
        truthState: f.truthState,
        confidence: f.confidence,
        salience: f.salience,
        score: scores[f.id] ?? 0,
      };
    });

    const entitiesBlock: EntityView[] = entities
      .filter(e => e.summary)
      .map(e => ({
        entityId: e.id,
        canonicalName: e.canonicalName,
        type: e.type,
        summary: e.summary,
        salience: e.salience,
      }));

    const timelineBlock: TimelineEvent[] = facts
      .map(f => {
        const subjectEntity = entities.find(e => e.id === f.subjectEntityId);
        const obj = f.objectValueJson
          ? String((f.objectValueJson as { value?: unknown }).value ?? '')
          : '';
        return {
          timestamp: f.validFrom,
          description: `${subjectEntity?.canonicalName ?? ''} ${f.predicate} ${obj}`.trim(),
          factId: f.id,
          episodeId: f.sourceEpisodeId,
        } satisfies TimelineEvent;
      })
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    const episodesBlock: EpisodeView[] = episodes.map(ep => ({
      episodeId: ep.id,
      rawText: ep.rawText,
      createdAt: ep.createdAt,
      source: ep.source,
      score: epScoreFor(ep.id, result),
    }));

    const evidenceBlock: EvidenceView[] = facts.map(f => ({
      episodeIds: [f.sourceEpisodeId],
      description: `Fact ${f.id} sourced from episode ${f.sourceEpisodeId}`,
      factId: f.id,
    }));

    const confidence = factsBlock.length
      ? factsBlock.reduce((s, f) => s + f.confidence, 0) / factsBlock.length
      : 0.0;

    const formatted = this.format(factsBlock, entitiesBlock, timelineBlock, episodesBlock, evidenceBlock);

    return {
      factsBlock,
      entitiesBlock,
      timelineBlock,
      episodesBlock,
      evidenceBlock,
      confidence,
      formatted,
    };
  }

  private format(
    facts: FactView[],
    entities: EntityView[],
    timeline: TimelineEvent[],
    episodes: EpisodeView[],
    evidence: EvidenceView[],
  ): string {
    const lines: string[] = [];

    lines.push('[FACTS]');
    if (facts.length) {
      for (const f of facts) {
        lines.push(`  - ${f.subject} ${f.predicate} ${f.object} (confidence=${f.confidence.toFixed(2)})`);
      }
    } else {
      lines.push('  (none)');
    }

    if (entities.length) {
      lines.push('[ENTITIES]');
      for (const e of entities) {
        lines.push(`  - ${e.canonicalName} [${e.type}]: ${e.summary}`);
      }
    }

    if (timeline.length > 1) {
      lines.push('[TIMELINE]');
      for (const t of timeline) {
        lines.push(`  - ${t.timestamp.toISOString()}: ${t.description}`);
      }
    }

    lines.push('[EPISODES]');
    if (episodes.length) {
      for (const ep of episodes) {
        const snippet = ep.rawText.length > 120 ? ep.rawText.slice(0, 120) + '…' : ep.rawText;
        lines.push(`  [${ep.episodeId.slice(0, 8)}] ${snippet}`);
      }
    } else {
      lines.push('  (none)');
    }

    if (evidence.length) {
      lines.push('[EVIDENCE]');
      for (const ev of evidence) {
        lines.push(`  - ${ev.description}`);
      }
    }

    return lines.join('\n');
  }
}

function epScoreFor(episodeId: string, result: RetrievalResult): number {
  const factWithEp = result.facts.find(f => f.sourceEpisodeId === episodeId);
  return factWithEp ? (result.scores[factWithEp.id] ?? 0) : 0;
}
