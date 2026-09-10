from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def sortable_id(prefix: str = "") -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}{stamp}_{uuid.uuid4().hex[:12]}"


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=pretty,
    )


def atomic_write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    atomic_write_text(path, json_dumps(value, pretty=True) + "\n", mode=mode)


_append_locks: dict[str, threading.Lock] = {}
_append_locks_guard = threading.Lock()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _append_locks_guard:
        lock = _append_locks.setdefault(key, threading.Lock())
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json_dumps(value) + "\n")
            handle.flush()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_identifier(value: str, *, label: str = "identifier") -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(
            f"Invalid {label}: use 1-128 letters, numbers, dots, dashes, or underscores"
        )
    return value


def descriptive_slug(value: str, *, label: str = "name") -> str:
    original = value.strip()
    slug = re.sub(r"[^a-z0-9]+", "-", original.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:120].rstrip("-")
    if len(slug) < 8 or slug in {
        "context",
        "memory",
        "summary",
        "checkpoint",
        "offload",
        "notes",
        "misc",
        "temp",
    }:
        raise ValueError(
            f"{label} must be descriptive enough to reveal what should be loaded"
        )
    return slug


def resolve_inside(root: Path, supplied: str) -> Path:
    relative = Path(supplied)
    if relative.is_absolute():
        candidate = relative.expanduser().resolve()
    else:
        candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must remain inside {root}: {supplied}") from exc
    return candidate


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Small cross-process advisory lock on Unix; thread-safe fallback elsewhere."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def from_code_file(cls, file: str | Path) -> "Paths":
        # artificium-code/artificium/<module>.py -> Artificium-revolution root
        return cls(Path(file).resolve().parents[2])

    @property
    def code(self) -> Path:
        return self.root / "artificium-code"

    @property
    def prompts(self) -> Path:
        return self.code / "prompts"

    @property
    def prompt_manifest(self) -> Path:
        return self.prompts / "manifest.toml"

    @property
    def config(self) -> Path:
        return self.code / "config.json"

    @property
    def secrets(self) -> Path:
        return self.code / ".secrets.json"

    @property
    def mind(self) -> Path:
        return self.root / "mind"

    @property
    def self_file(self) -> Path:
        return self.mind / "self.txt"

    @property
    def meta_memory(self) -> Path:
        return self.mind / "meta_memory.md"

    @property
    def memory(self) -> Path:
        return self.mind / "memory"

    @property
    def working_context(self) -> Path:
        return self.runtime / "context.jsonl"

    @property
    def streams(self) -> Path:
        return self.runtime / "attention"

    @property
    def interactions(self) -> Path:
        return self.mind / "interactions"

    @property
    def created_tools(self) -> Path:
        return self.mind / "tools"

    @property
    def space(self) -> Path:
        """Agent-owned workspace for projects and artifacts that are not memories."""

        return self.mind / "space"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def lifetime_log(self) -> Path:
        return self.logs / "lifetime.jsonl"

    @property
    def life_loop_log(self) -> Path:
        return self.logs / "life-loop.jsonl"

    @property
    def model_log(self) -> Path:
        return self.logs / "model"

    @property
    def outputs(self) -> Path:
        return self.logs / "outputs"

    @property
    def context_archive(self) -> Path:
        return self.logs / "context"

    @property
    def feature_logs(self) -> Path:
        return self.logs / "features"

    @property
    def feature_summary(self) -> Path:
        return self.feature_logs / "summary.json"

    @property
    def records_lock(self) -> Path:
        return self.runtime / "records.lock"

    def feature_log(self, feature: str) -> Path:
        safe = safe_identifier(feature, label="feature log name")
        return self.feature_logs / f"{safe}.jsonl"

    @property
    def runtime(self) -> Path:
        return self.logs / "runtime"

    @property
    def runtime_state(self) -> Path:
        return self.runtime / "state.json"

    @property
    def initialization_state(self) -> Path:
        return self.runtime / "initialization.json"

    @property
    def pressure_state(self) -> Path:
        return self.runtime / "context_pressure.json"

    @property
    def visual_context(self) -> Path:
        return self.runtime / "visual-context.json"

    @property
    def sleep_state(self) -> Path:
        return self.runtime / "sleep.json"

    @property
    def self_history(self) -> Path:
        return self.runtime / "self-history"

    @property
    def process_lock(self) -> Path:
        return self.runtime / "artificium.lock"

    @property
    def interaction_lock(self) -> Path:
        return self.runtime / "interactions.lock"

    @property
    def scheduler_root(self) -> Path:
        return self.runtime / "scheduler"

    @property
    def scheduler_tasks(self) -> Path:
        return self.scheduler_root / "tasks"

    @property
    def scheduler_lock(self) -> Path:
        return self.scheduler_root / "scheduler.lock"

    @property
    def receipts(self) -> Path:
        return self.runtime / "interaction_receipts"

    @property
    def notification_root(self) -> Path:
        return self.runtime / "notifications"

    @property
    def notifications_new(self) -> Path:
        return self.notification_root / "new"

    @property
    def notifications_processing(self) -> Path:
        return self.notification_root / "processing"

    @property
    def notifications_delivered(self) -> Path:
        return self.notification_root / "delivered"

    def ensure_layout(self) -> None:
        directories = [
            self.code,
            self.mind,
            self.memory,
            self.streams,
            self.interactions,
            self.created_tools,
            self.space,
            self.logs,
            self.model_log,
            self.outputs,
            self.context_archive,
            self.feature_logs,
            self.runtime,
            self.self_history,
            self.receipts,
            self.scheduler_tasks,
            self.notifications_new,
            self.notifications_processing,
            self.notifications_delivered,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for jsonl in (self.working_context, self.lifetime_log, self.life_loop_log):
            if not jsonl.exists():
                atomic_write_text(jsonl, "")
