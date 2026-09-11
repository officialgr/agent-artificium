"""A bounded, tool-free summarizer. It never runs an Artificium life-loop."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .config import Config
from .context_budget import TokenCount, budget_request, measure
from .engine import EngineError, make_engine
from .filesystem import json_dumps, sortable_id
from .memory import TokenEstimator
from .prompts import PromptPack
from .records import Records


MAX_RECOVERY_ATTEMPTS = 3


def helper_config(config: Config, attempt: int) -> Config:
    if config.adapter == "custom_json":
        raise EngineError("Emergency summarization requires a built-in API adapter; custom JSON history was retained.", kind="context")
    # Keep routing and the model's template, but no tools, conversation IDs,
    # continuation IDs, response schemas, or other life-loop request options.
    options = {key: copy.deepcopy(config.request_options[key]) for key in (
        "provider", "service_tier", "chat_template", "chat_template_kwargs",
    ) if key in config.request_options}
    effort = None
    if attempt == 1:
        if config.adapter in {"llamacpp", "vllm"}:
            options.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        elif config.adapter == "gemini":
            effort = "none" if config.model.removeprefix("models/").startswith("gemini-2.5") else "low"
        elif config.adapter == "ollama":
            effort = "low" if "gpt-oss" in config.model.lower() else "none"
        elif config.adapter in {"anthropic", "openai_responses", "openrouter"}:
            effort = "none"
    return replace(
        config, request_options=options, reasoning_effort=effort,
        reasoning_budget_tokens=None, reasoning_mode=None,
        max_output_tokens=min(2048, config.context_window_tokens // 16),
        stop_sequences=[], emergency_offload=False,
    )


def source_excerpt(history: list[dict[str, Any]], byte_limit: int) -> tuple[str, bool]:
    # No image loading. Image paths and tool results remain inert source data.
    raw = json_dumps(history).encode("utf-8")
    if len(raw) <= byte_limit:
        return raw.decode("utf-8"), False
    start = byte_limit // 3
    text = (raw[:start].decode("utf-8", errors="ignore")
            + "\n[Middle of record omitted; consult the original request log.]\n"
            + raw[-(byte_limit - start):].decode("utf-8", errors="ignore"))
    return text, True


def summarize(*, config: Config, api_key: str | None, history: list[dict[str, Any]],
              original_log: str, attempt: int, prompts: PromptPack,
              records: Records) -> tuple[str, bool]:
    helper = helper_config(config, attempt)
    engine = make_engine(helper, api_key)
    engine.request_attempts = 1
    excerpt, omitted = source_excerpt(history, min(262_144, config.context_window_tokens // (2 ** attempt)))
    messages = [
        {"role": "system", "content": prompts.runtime("emergency_summary")},
        {"role": "user", "content": json_dumps({
            "original_request_log": original_log, "excerpt_omitted_material": omitted,
            "execution_record": excerpt,
        })},
    ]
    request_id = sortable_id("emergency_summary_")
    parameters: dict[str, Any] = {}
    try:
        prepared = engine.prepare(messages)
        count = measure(engine, prepared, messages, TokenEstimator(config.chars_per_token))
        if count.source == "estimate":
            # Be deliberately conservative for the small text-only helper.
            count = TokenCount(max(count.tokens, len(json_dumps(messages).encode("utf-8"))), "estimate")
        prepared = budget_request(prepared, helper, count)
        parameters = prepared.safe_summary()
        parameters["token_count"] = {"tokens": count.tokens, "source": count.source}
        reply = engine.complete_prepared(prepared)
        records.model_exchange(request_id=request_id, messages=messages,
                               request_parameters=parameters, response=vars(reply))
        if reply.finish_reason in {"length", "incomplete", "max_tokens", "MAX_TOKENS"} or len(reply.content.strip()) < 40:
            raise EngineError("The emergency summary was empty, too short, or truncated; original working memory is retained.", kind="context", reply=reply)
        # Bound even a provider which ignores its requested output cap.
        if len(reply.content.encode("utf-8")) > helper.max_output_tokens * 16:
            raise EngineError("The emergency summary exceeded its output bound.", kind="context", reply=reply)
        return reply.content.strip(), omitted
    except EngineError as exc:
        records.model_exchange(request_id=request_id, messages=messages,
                               request_parameters=parameters, error=str(exc),
                               response=vars(exc.reply) if exc.reply else None)
        raise
