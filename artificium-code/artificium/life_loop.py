from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .filesystem import json_dumps, sortable_id


TAG_PATTERN = re.compile(
    r"<(?P<tag>think|tool|tool_call)\b(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
TOOL_START_PATTERN = re.compile(r"<(?:tool_call|tool)\b", re.IGNORECASE)
NAME_PATTERN = re.compile(
    r"\bname\s*=\s*(?:\"(?P<double>[^\"]+)\"|'(?P<single>[^']+)'|(?P<bare>[^\s>]+))",
    re.IGNORECASE,
)


def _strip_fence(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.I)
    return match.group(1).strip() if match else value


@dataclass
class ToolIntent:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class LifeLoopOutput:
    raw: str
    thoughts: list[str]
    tools: list[ToolIntent]
    visible: str


def parse_life_loop_output(content: str) -> LifeLoopOutput:
    thoughts: list[str] = []
    tools: list[ToolIntent] = []
    spans: list[tuple[int, int]] = []
    complete_spans: list[tuple[int, int]] = []
    for match in TAG_PATTERN.finditer(content):
        spans.append(match.span())
        complete_spans.append(match.span())
        tag = match.group("tag").lower()
        body = match.group("body").strip()
        if tag == "think":
            if body:
                thoughts.append(body)
            continue
        raw_arguments = _strip_fence(body)
        parse_error: str | None = None
        name = ""
        arguments: dict[str, Any] = {}
        try:
            decoded = json.loads(raw_arguments or "{}")
            if not isinstance(decoded, dict):
                raise TypeError("tool call must be a JSON object")
            if tag == "tool_call":
                if "tool" in decoded:
                    name = str(decoded.get("tool") or "").strip()
                    arguments = {
                        str(key): value for key, value in decoded.items() if key != "tool"
                    }
                else:
                    # Revolution 1.4 and earlier used a nested name/arguments object.
                    # Keep it readable so old working contexts can recover while new
                    # promptgramming teaches the smaller flat request.
                    name = str(decoded.get("name") or "").strip()
                    supplied_arguments = decoded.get("arguments", {})
                    if not isinstance(supplied_arguments, dict):
                        raise TypeError("tool call `arguments` must be a JSON object")
                    arguments = supplied_arguments
            else:
                # Legacy syntax remains readable so an existing context or a less capable
                # model cannot strand itself while learning the canonical protocol.
                name_match = NAME_PATTERN.search(match.group("attrs"))
                if name_match:
                    name = next(
                        value
                        for value in name_match.group("double", "single", "bare")
                        if value is not None
                    ).strip()
                arguments = decoded
            if not name:
                parse_error = "tool call is missing a non-empty name"
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = f"invalid tool-call JSON: {exc}"
        tools.append(
            ToolIntent(
                id=sortable_id("tool_"),
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
                parse_error=parse_error,
            )
        )

    # A missing or mismatched closing tag previously became ordinary visible text,
    # providing no correction and leaving a bad few-shot example in working context.
    # Detect every unmatched tool start and turn it into an explicit parse failure.
    unmatched_starts = [
        match
        for match in TOOL_START_PATTERN.finditer(content)
        if not any(start <= match.start() < end for start, end in complete_spans)
    ]
    for index, match in enumerate(unmatched_starts):
        end = unmatched_starts[index + 1].start() if index + 1 < len(unmatched_starts) else len(content)
        raw = content[match.start() : end].strip()
        open_end = raw.find(">")
        opening = raw[: open_end + 1] if open_end >= 0 else raw[:200]
        name_match = NAME_PATTERN.search(opening)
        name = ""
        if name_match:
            name = next(
                value
                for value in name_match.group("double", "single", "bare")
                if value is not None
            ).strip()
        if not name:
            json_name = re.search(
                r'["\'](?:tool|name)["\']\s*:\s*["\']([^"\']+)', raw
            )
            if json_name:
                name = json_name.group(1).strip()
        spans.append((match.start(), end))
        tools.append(
            ToolIntent(
                id=sortable_id("tool_"),
                name=name,
                arguments={},
                raw_arguments=raw,
                parse_error=(
                    "incomplete or mismatched tool tag; close a flat request with "
                    "`</tool_call>`"
                ),
            )
        )

    visible_parts: list[str] = []
    cursor = 0
    merged_spans: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1] = (merged_spans[-1][0], max(end, merged_spans[-1][1]))
        else:
            merged_spans.append((start, end))
    for start, end in merged_spans:
        if start > cursor:
            visible_parts.append(content[cursor:start])
        cursor = end
    visible_parts.append(content[cursor:])
    visible = "\n".join(part.strip() for part in visible_parts if part.strip()).strip()
    return LifeLoopOutput(content, thoughts, tools, visible)


def render_normalized_life_loop_output(
    parsed: LifeLoopOutput, *, include_tools: bool = True
) -> str:
    """Return clean context evidence without preserving arbitrary tool formatting."""

    parts = [f"<think>{thought}</think>" for thought in parsed.thoughts]
    if parsed.visible:
        parts.append(parsed.visible)
    if include_tools:
        for intent in parsed.tools:
            if intent.parse_error or not intent.name:
                continue
            payload = {"tool": intent.name, **intent.arguments}
            parts.append(f"<tool_call>{json_dumps(payload)}</tool_call>")
    if not parts:
        return "[No valid life-loop output was retained from this inference.]"
    return "\n".join(parts)


def render_observation(
    *,
    kind: str,
    content: Any,
    tool_name: str | None = None,
    tool_id: str | None = None,
) -> str:
    attributes = [f'kind="{kind}"']
    if tool_name:
        attributes.append(f'tool="{tool_name}"')
    if tool_id:
        attributes.append(f'id="{tool_id}"')
    body = content if isinstance(content, str) else json_dumps(content, pretty=True)
    return (
        f"<life-loop-observation {' '.join(attributes)}>\n"
        f"{body.rstrip()}\n"
        "</life-loop-observation>"
    )


def concise_tool_catalog(specs: list[dict[str, Any]]) -> str:
    lines = []
    for spec in specs:
        signature = ", ".join(spec.get("arguments", []))
        lines.append(
            f"- `{spec['name']}({signature})` — {str(spec['description']).strip()}"
        )
    return "\n".join(lines)
