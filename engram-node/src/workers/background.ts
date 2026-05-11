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
      // If the same subject+predicate already has a committed current fact from
      // a more recent store(), the quarantine should be rejected.
      if (!qf.candidateSuperseedsFactId) {
        // No conflict target anymore — can auto-approve
        continue;
      }
      const original = this.store.getFact(qf.candidateSuperseedsFactId);
      if (!original || original.truthState !== 'current') {
        // Original already superseded by something else — reject quarantine
        this.store.updateQuarantineStatus(qf.id, 'rejected');
      }
    }
  }

  runCleanup(scopeId: string): void {
    this.runDecay(scopeId);
    // Future: alias pruning, orphan entity cleanup, etc.
  }
}
