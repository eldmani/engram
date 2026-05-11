from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from uuid import uuid4

from ..config import EngramConfig
from ..models import Fact
from ..storage.sqlite_store import SQLiteStore

_log = logging.getLogger(__name__)


class BackgroundWorkers:
    """
    Background workers for memory dynamics.

    All methods are safe to call manually, in a thread, or via a scheduler
    (e.g., APScheduler, Celery, or a simple threading.Timer loop).
    """

    def __init__(self, store: SQLiteStore, config: EngramConfig) -> None:
        self._store = store
        self._cfg = config

    # ------------------------------------------------------------------
    # Reinforcement — called after retrieval
    # ------------------------------------------------------------------

    def reinforce(self, fact_ids: list[str]) -> None:
        """Boost strength on every accessed fact."""
        for fid in fact_ids:
            self._store.reinforce_fact(fid, self._cfg.reinforcement_boost)

    # ------------------------------------------------------------------
    # Strength Decay
    # ------------------------------------------------------------------

    def run_decay(self, scope_id: str) -> None:
        """
        Apply power-law decay to infrequently accessed facts and slow linear
        decay to stable facts. Does NOT modify truth_state or valid_to.
        """
        self._store.apply_decay(scope_id, self._cfg.stable_access_threshold)

    # ------------------------------------------------------------------
    # Heat Promotion
    # ------------------------------------------------------------------

    def compute_heat(self, fact: Fact) -> float:
        """
        heat = (access_count * strength) / log(2 + age_hours)
        Higher heat → candidate for promotion to persistent fast paths.
        """
        if fact.created_at is None:
            return 0.0
        age_hours = max(1.0, (datetime.now(timezone.utc) - fact.created_at).total_seconds() / 3600)
        return (fact.access_count * fact.strength) / math.log(2 + age_hours)

    def run_heat_promotion(self, scope_id: str) -> list[str]:
        """
        Return IDs of facts above heat_promotion_threshold.
        Callers can use this list to populate caches or denormalized views.
        """
        facts = self._store.get_current_facts_for_scope(scope_id, limit=500)
        hot = [
            f.id for f in facts
            if self.compute_heat(f) >= self._cfg.heat_promotion_threshold
        ]
        return hot

    # ------------------------------------------------------------------
    # Reconsolidation
    # ------------------------------------------------------------------

    def run_reconsolidation(self, scope_id: str) -> None:
        """
        Re-evaluate pending quarantined facts.

        If the same predicate has since been confirmed by additional episodes
        (i.e. a fact with same subject/predicate now exists at high confidence),
        approve and commit the quarantined candidate.
        """
        pending = self._store.get_pending_quarantined_facts(scope_id)
        for qf in pending:
            # Look up the entity by extracted_subject name
            entity = self._store.find_entity_by_name(scope_id, qf.extracted_subject)
            if entity is None:
                continue

            # Check if a current fact now exists for same subject+predicate
            existing = self._store.get_current_facts(
                scope_id, entity.id, qf.extracted_predicate
            )
            if existing:
                # Same predicate confirmed by a later episode — reject quarantine
                # (the correct fact is already committed via a subsequent store())
                self._store.update_quarantine_status(qf.id, "rejected")
            # Future enhancement: if confidence increased, promote to approved

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def run_cleanup(self, scope_id: str) -> None:
        """
        Prune orphan structures and low-value data.
        Extend as needed for production workloads.
        """
        # Decay first so low-strength facts are identified
        self.run_decay(scope_id)
        # Future: prune orphan aliases, compact stale traces, etc.
