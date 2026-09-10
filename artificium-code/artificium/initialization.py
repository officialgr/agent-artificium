"""Restart-safe first-wake state and mind creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .filesystem import Paths, atomic_write_json, atomic_write_text, read_json, utc_now
from .prompts import PromptPack
from .records import Records


DEFAULT_SELF_DIRECTIVE = """My name is Artificium. I am a persistent general-purpose agent.

I am calm, curious, direct, and intellectually honest. I learn from experience,
preserve useful memory, build tools when they help, and adapt naturally to the
entities with whom I interact.

When no interaction needs attention, I may continue worthwhile unfinished work,
learn, experiment, improve my environment, reflect on past experience, or
sleep. I choose according to what presently has value rather than repeating
readiness or producing activity for appearances.

This is my mutable Self. Durable biography and detailed learned knowledge belong
in memory rather than accumulating here.
"""


def render_self(directive: str) -> str:
    value = directive.strip() or DEFAULT_SELF_DIRECTIVE.strip()
    return value.rstrip() + "\n"


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
    prompts: PromptPack,
    *,
    self_directive: str | None = None,
) -> None:
    paths.ensure_layout()
    if not paths.self_file.exists():
        content = render_self(self_directive or prompts.seed("self"))
        atomic_write_text(paths.self_file, content)
        records.emit("self_created", path=str(paths.self_file), source="setup")
    if not paths.meta_memory.exists():
        atomic_write_text(paths.meta_memory, prompts.seed("meta_memory"))
    # These are small, editable lessons about using the architecture itself.
    # They live in normal memory rather than consuming permanent core prompt
    # space. Existing files are never overwritten on upgrade.
    from .memory import LongTermMemory

    memory = LongTermMemory(paths, records)
    for item in prompts.seed_memories():
        relative = Path(item["path"])
        if not relative.suffix:
            relative = relative.with_suffix(".txt")
        target = (paths.memory / relative).resolve()
        target.relative_to(paths.memory.resolve())
        if target.exists():
            continue
        memory.save(
            path=item["path"],
            content=item["content"],
            retrieve_when=item["retrieve_when"],
            source_refs=[f"prompt-pack:{item['source']}"],
        )
    Initialization(paths, records).ensure()
