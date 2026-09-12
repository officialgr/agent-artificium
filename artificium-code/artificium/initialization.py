"""Restart-safe first-wake state and validation of the shipped mind."""

from __future__ import annotations

from typing import Any

from .filesystem import Paths, atomic_write_json, atomic_write_text, read_json, utc_now
from .records import Records


def render_self(directive: str) -> str:
    value = directive.strip()
    if not value:
        raise ValueError("Self must not be empty")
    return value + "\n"


class Initialization:
    def __init__(self, paths: Paths, records: Records):
        self.paths = paths
        self.records = records

    def ensure(self) -> dict[str, Any]:
        state = read_json(self.paths.initialization_state)
        if isinstance(state, dict):
            return state
        state = {
            "status": "pending",
            "created_at": utc_now(),
            "completed_at": None,
            "summary": None,
        }
        atomic_write_json(self.paths.initialization_state, state)
        return state

    def pending(self) -> bool:
        return self.ensure().get("status") != "completed"

    def finish(self, summary: str) -> dict[str, Any]:
        if len(summary.strip()) < 20:
            raise ValueError("initialization summary must explain what was verified")
        state = self.ensure()
        state.update(
            {"status": "completed", "completed_at": utc_now(), "summary": summary.strip()}
        )
        atomic_write_json(self.paths.initialization_state, state)
        self.records.emit("initialization_completed", summary=summary.strip())
        return state


def initialize_mind(
    paths: Paths,
    records: Records,
    *,
    self_directive: str | None = None,
) -> None:
    """Use the shipped/instance mind without recreating its editable memories."""
    paths.ensure_layout()
    if not paths.self_file.exists() and self_directive is not None:
        atomic_write_text(paths.self_file, render_self(self_directive))
        records.emit("self_created", path=str(paths.self_file), source="setup")
    for path in (paths.self_file, paths.meta_memory):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. A complete installation includes mind/; "
                "restore the missing file from your backup or a clean release."
            )
    Initialization(paths, records).ensure()
