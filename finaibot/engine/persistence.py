"""Local, atomic JSON persistence for deterministic M1 backtest runs.

Persisted records carry a canonical SHA-256 integrity digest.  The digest is
intended to detect accidental or incomplete changes to a checkpoint; it is
not an authenticity guarantee when an actor can rewrite both the payload and
the digest.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from typing import Any


CHECKPOINT_VERSION = 2
INTEGRITY_DIGEST_FIELD = "integrity_digest"


class RunPersistenceError(RuntimeError):
    """Raised when a run record cannot be safely written, loaded, or resumed."""


class BacktestInterrupted(RuntimeError):
    """Raised by the deterministic test failpoint after a committed checkpoint."""

    def __init__(self, run_id: str, path: Path):
        super().__init__(f"backtest interrupted after checkpoint for run {run_id}")
        self.run_id = run_id
        self.path = path


@dataclass(frozen=True)
class RunPersistenceConfig:
    """Configure optional local run-record persistence.

    ``directory`` is deliberately excluded from reproducibility fingerprints;
    it is a machine-local storage location, not a trading input.  The
    ``interrupt_after_bars`` field is a deterministic test/operations failpoint
    and is also excluded from trading-input fingerprints.
    """

    directory: str | Path | None = None
    checkpoint_every_bars: int = 1
    interrupt_after_bars: int | None = None

    def __post_init__(self) -> None:
        if self.directory is not None:
            if isinstance(self.directory, bool) or not str(self.directory).strip():
                raise ValueError("directory must be a non-empty path")
            object.__setattr__(self, "directory", str(self.directory))
        if isinstance(self.checkpoint_every_bars, bool) or not isinstance(self.checkpoint_every_bars, int) or self.checkpoint_every_bars < 1:
            raise ValueError("checkpoint_every_bars must be a positive integer")
        if self.interrupt_after_bars is not None:
            if isinstance(self.interrupt_after_bars, bool) or not isinstance(self.interrupt_after_bars, int) or self.interrupt_after_bars < 1:
                raise ValueError("interrupt_after_bars must be a positive integer")
            if self.directory is None:
                raise ValueError("interrupt_after_bars requires a persistence directory")

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def run_path(self, run_id: str) -> Path:
        if not self.enabled:
            raise RunPersistenceError("run persistence is not configured")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or any(separator in run_id for separator in ("/", "\\")):
            raise RunPersistenceError("run_id is not a safe local record identifier")
        return Path(self.directory) / f"{run_id}.json"


def write_run_record(config: RunPersistenceConfig, run_id: str, record: dict[str, Any]) -> Path:
    """Atomically commit one JSON record, preserving the previous record on failure."""
    path = config.run_path(run_id)
    temporary_path: Path | None = None
    try:
        committed_record = dict(record)
        committed_record[INTEGRITY_DIGEST_FIELD] = canonical_record_digest(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(committed_record, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RunPersistenceError(f"could not atomically write run record {path}") from exc
    return path


def load_run_record(config: RunPersistenceConfig, run_id: str) -> dict[str, Any]:
    """Load and structurally validate a JSON run record without object deserialization."""
    path = config.run_path(run_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunPersistenceError(f"could not load run record {path}") from exc
    if not isinstance(record, dict):
        raise RunPersistenceError("run record must contain a JSON object")
    stored_digest = record.get(INTEGRITY_DIGEST_FIELD)
    if not isinstance(stored_digest, str) or len(stored_digest) != 64 or any(character not in "0123456789abcdef" for character in stored_digest):
        raise RunPersistenceError("run record is missing its integrity digest")
    try:
        expected_digest = canonical_record_digest(record)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunPersistenceError("run record contains values that cannot be integrity-checked") from exc
    if stored_digest != expected_digest:
        raise RunPersistenceError("run record integrity digest mismatch")
    if isinstance(record.get("checkpoint_version"), bool) or record.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise RunPersistenceError("unsupported or missing checkpoint version")
    if not isinstance(record.get("run_id"), str) or record.get("run_id") != run_id:
        raise RunPersistenceError("run record run_id does not match the requested run")
    if not isinstance(record.get("status"), str) or record.get("status") not in {"RUNNING", "INTERRUPTED", "COMPLETED", "REJECTED"}:
        raise RunPersistenceError("run record has an unsupported status")
    required = ("created_at", "strategy", "strategy_hash", "dataset_hash", "initial_capital", "configuration", "configuration_hash", "input_hash", "engine_version", "software", "progress", "state")
    missing = [key for key in required if key not in record]
    if missing:
        raise RunPersistenceError(f"run record is incomplete; missing: {', '.join(missing)}")
    if not isinstance(record["created_at"], str) or not _is_utc_timestamp(record["created_at"]):
        raise RunPersistenceError("run record created_at must be a timezone-aware UTC timestamp")
    for key in ("strategy_hash", "dataset_hash", "configuration_hash", "input_hash", "engine_version"):
        if not isinstance(record[key], str) or not record[key]:
            raise RunPersistenceError(f"run record {key} must be a non-empty string")
    if not isinstance(record["strategy"], Mapping) or not isinstance(record["configuration"], Mapping) or not isinstance(record["software"], Mapping):
        raise RunPersistenceError("run record strategy, configuration, and software must be objects")
    if isinstance(record["initial_capital"], bool):
        raise RunPersistenceError("run record initial_capital must be finite and numeric")
    try:
        initial_capital = float(record["initial_capital"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunPersistenceError("run record initial_capital must be finite and numeric") from exc
    if not isfinite(initial_capital) or initial_capital <= 0:
        raise RunPersistenceError("run record initial_capital must be finite and positive")
    if not isinstance(record["progress"], Mapping) or not isinstance(record["state"], Mapping):
        raise RunPersistenceError("run record progress and state must be objects")
    return record


def _is_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def canonical_record_digest(record: Mapping[str, Any]) -> str:
    """Return the stable digest of a record, excluding its digest field."""
    if not isinstance(record, Mapping):
        raise TypeError("run record must be a mapping")
    payload = {
        str(key): value
        for key, value in record.items()
        if key != INTEGRITY_DIGEST_FIELD
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
