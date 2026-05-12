import type { Fact } from '../models/index.js';
import type { SQLiteStore } from '../storage/sqliteStore.js';
import type { EngramConfig } from '../config.js';

export class BackgroundWorkers {
  constructor(
    private store: SQLiteStore,
    private config: EngramConfig,
  ) {}

  reinforce(factIds: string[]): void {
    for (const id of factIds) {
      this.store.reinforceFact(id, this.config.reinforcementBoost);
    }
  }

  runDecay(scopeId: string): void {
    this.store.applyDecay(scopeId, this.config.stableAccessThreshold);
  }

  computeHeat(fact: Fact): number {
    const ageHours = Math.max(1, (Date.now() - fact.createdAt.getTime()) / 3_600_000);
    return (fact.accessCount * fact.strength) / Math.log2(2 + ageHours);
  }

  runHeatPromotion(scopeId: string): string[] {
    const facts = this.store.getCurrentFactsForScope(scopeId, 1000);
    return facts
      .filter(f => this.computeHeat(f) >= this.config.heatPromotionThreshold)
      .map(f => f.id);
  }

  runReconsolidation(scopeId: string): void {
    const pending = this.store.getPendingQuarantinedFacts(scopeId);
    for (const qf of pending) {
      // Look up the entity by the quarantined fact's extracted subject name.
      const entity = this.store.findEntityByName(scopeId, qf.extractedSubject);
      if (entity === null) continue;

      // If there is already a current fact for the same subject+predicate,
      // the correct knowledge is already committed — reject the quarantine.
      const existing = this.store.getCurrentFacts(scopeId, entity.id, qf.extractedPredicate);
      if (existing.length > 0) {
        this.store.updateQuarantineStatus(qf.id, 'rejected');
      }
    }
  }

  runCleanup(scopeId: string): void {
    this.runDecay(scopeId);
    // Future: alias pruning, orphan entity cleanup, etc.
  }
}
