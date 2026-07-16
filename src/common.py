"""Shared, dependency-light experiment utilities.

The generation jobs are intentionally append-only.  A process may receive
SIGUSR1 five minutes before Slurm's wall clock and exit after its current batch;
on resubmission, completed record keys are skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def object_fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def prompt_hash(prompt: str) -> str:
    return sha256_text(prompt)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {resolved}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {resolved}:{line_number}")
            yield value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Append and fsync records so a preempted job loses at most the active batch."""
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(dict(record)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def atomic_write_json(path: str | Path, value: Any) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable_revision(
    configured_revision: str | None,
    artifact_id: str,
    revision_file: str | Path,
) -> str:
    """Return a 40-hex Hub commit, rejecting mutable labels such as ``main``."""
    revision = configured_revision
    revision_path = resolve_path(revision_file)
    if not revision and revision_path.exists():
        revisions = json.loads(revision_path.read_text(encoding="utf-8"))
        entry = revisions.get(artifact_id, {})
        revision = entry.get("revision") if isinstance(entry, dict) else entry
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError(
            f"{artifact_id} is not pinned to a 40-character Hub commit. "
            "Run src/stage_assets.py first or set an immutable revision in the config."
        )
    return revision


KEY_FIELDS = ("dataset", "item_id", "form_id", "mode", "seed", "method")


def record_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    missing = [field for field in KEY_FIELDS if field not in record]
    if missing:
        raise KeyError(f"Result record is missing key fields: {missing}")
    return tuple(str(record[field]) for field in KEY_FIELDS)


def completed_keys(path: str | Path) -> tuple[set[tuple[str, ...]], list[tuple[str, ...]]]:
    seen: set[tuple[str, ...]] = set()
    duplicates: list[tuple[str, ...]] = []
    for record in iter_jsonl(path):
        key = record_key(record)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return seen, duplicates


@dataclass
class StopController:
    requested: bool = False
    received_signal: int | None = None

    def install(self) -> None:
        for sig in (signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)):
            signal.signal(sig, self._handle)

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.received_signal = signum
        print(f"[signal] received {signum}; stopping after the active batch", flush=True)


def write_progress(
    path: str | Path,
    *,
    run_id: str,
    config_fingerprint: str,
    completed: int,
    expected: int,
    stop_controller: StopController | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "completed_records": completed,
        "expected_records": expected,
        "updated_at": utc_now(),
    }
    if stop_controller:
        payload["stop_requested"] = stop_controller.requested
        payload["received_signal"] = stop_controller.received_signal
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)


def batched(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
