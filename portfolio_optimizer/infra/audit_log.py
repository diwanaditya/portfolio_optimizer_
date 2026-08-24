"""
Tamper-Evident Audit Log.

Every trading decision (order placed, risk check passed/failed, rebalance
triggered, manual override) should be recorded in a log that can prove,
after the fact, whether it was altered. This module implements a
hash-chained append-only log (each entry's hash includes the previous
entry's hash, so altering or deleting any past entry breaks every
subsequent hash and is immediately detectable) — the same core idea
behind blockchain integrity, applied to a simple local audit trail rather
than a distributed ledger.

SCOPE HONESTY: this proves TAMPER-EVIDENCE (you can detect if the log was
altered), not tamper-PROOFNESS (an attacker with write access to the log
file could still rewrite the whole chain from a point forward, computing
new valid hashes as they go, unless entries are also signed and pushed to
an external, append-only store — e.g. syncing chained hashes to a
separate system, a WORM-storage bucket, or a real blockchain). For actual
regulatory audit-log requirements, pair this with off-box replication of
at least the hash chain.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class AuditEntry:
    sequence: int
    timestamp: str
    event_type: str
    payload: dict
    previous_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "sequence": self.sequence, "timestamp": self.timestamp,
            "event_type": self.event_type, "payload": self.payload,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TamperEvidentAuditLog:
    GENESIS_HASH = "0" * 64

    def __init__(self):
        self.entries: list = []

    def record(self, event_type: str, payload: dict) -> AuditEntry:
        prev_hash = self.entries[-1].entry_hash if self.entries else self.GENESIS_HASH
        entry = AuditEntry(
            sequence=len(self.entries), timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type, payload=payload, previous_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> tuple[bool, int | None]:
        """Returns (is_valid, first_broken_index). Recomputes every
        entry's hash and checks the chain linkage; if anything in the log
        was altered after the fact, this will detect exactly where.
        """
        expected_prev = self.GENESIS_HASH
        for i, entry in enumerate(self.entries):
            if entry.previous_hash != expected_prev:
                return False, i
            if entry.compute_hash() != entry.entry_hash:
                return False, i
            expected_prev = entry.entry_hash
        return True, None

    def export_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries], indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TamperEvidentAuditLog":
        log = cls()
        data = json.loads(json_str)
        for item in data:
            entry = AuditEntry(**item)
            log.entries.append(entry)
        return log

    def entries_by_type(self, event_type: str) -> list:
        return [e for e in self.entries if e.event_type == event_type]
