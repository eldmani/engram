from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class Episode:
    id: str
    scope_id: str
    source: str
    raw_text: str
    metadata: Dict[str, Any]
    created_at: datetime
    checksum: str

    @staticmethod
    def compute_checksum(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
