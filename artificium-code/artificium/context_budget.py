"""Request budgets share one serving window; working memory is only a target."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from .config import Config
from .engine import Engine, EngineError, PreparedRequest
from .memory import TokenEstimator


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    source: str


def measure(engine: Engine, prepared: PreparedRequest,
            messages: list[dict[str, Any]], estimator: TokenEstimator) -> TokenCount:
    count = engine.count_input_tokens(prepared)
    if count is not None:
        return TokenCount(count, "provider")
    return TokenCount(estimator.messages(messages), "estimate")


def minimum_generation_room(config: Config) -> int:
    """Minimum room to proceed, never a maximum length for the response."""
    return (config.reasoning_budget_tokens or 0) + 256


def margin(config: Config, count: TokenCount) -> int:
    # Even native counts can differ slightly from a provider's final accounting.
    return max(256, config.context_window_tokens // (100 if count.source == "provider" else 20))


def available_output(config: Config, count: TokenCount) -> int:
    return config.context_window_tokens - count.tokens - margin(config, count)


def limit_output(prepared: PreparedRequest, limit: int, config: Config) -> PreparedRequest:
    """Bound generation including reasoning, retaining any smaller explicit cap.

    Custom contracts opt in by mapping $artificium.max_output_tokens. Never
    guess the meaning of an arbitrary custom field.
    """
    body = copy.deepcopy(prepared.payload)

    def cap(target: dict[str, Any], key: str) -> None:
        old = target.get(key)
        target[key] = min(old, limit) if isinstance(old, int) and old > 0 else limit

    adapter = prepared.adapter
    if adapter in {"llamacpp", "vllm", "openrouter", "openai_compatible"}:
        keys = [key for key in ("max_tokens", "max_completion_tokens") if key in body]
        for key in keys or ["max_tokens"]:
            cap(body, key)
    elif adapter == "openai_responses":
        cap(body, "max_output_tokens")
    elif adapter == "anthropic":
        cap(body, "max_tokens")
        thinking = body.get("thinking") or {}
        if thinking.get("type") == "enabled" and thinking.get("budget_tokens", 0) >= body["max_tokens"]:
            raise EngineError("Remaining context cannot fit the configured thinking budget and an answer.", kind="context")
    elif adapter == "gemini":
        cap(body.setdefault("generationConfig", {}), "maxOutputTokens")
    elif adapter == "ollama":
        cap(body.setdefault("options", {}), "num_predict")
    elif adapter == "custom_json":
        def mapped(template: Any, target: Any) -> None:
            if isinstance(template, dict) and isinstance(target, dict):
                for key, value in template.items():
                    if value == "$artificium.max_output_tokens":
                        cap(target, key)
                    elif key in target:
                        mapped(value, target[key])
        mapped(config.custom_contract.get("body"), body)
    return replace(prepared, payload=body)


def budget_request(prepared: PreparedRequest, config: Config, count: TokenCount) -> PreparedRequest:
    available = available_output(config, count)
    if available < minimum_generation_room(config):
        raise EngineError(
            f"Input ({count.tokens} tokens, {count.source}) leaves insufficient generation room "
            f"inside the {config.context_window_tokens}-token serving context.",
            kind="context", hint="Offload working memory or enable emergency offloading; history is retained.",
        )
    # No harness default output cap. Use the remaining serving context, subject
    # to an operator's explicit limit and the model's reported output maximum.
    # limit_output also retains smaller values in expert request options.
    limit = min(available, config.max_output_tokens or available)
    model_limit = config.model_capabilities.get("max_output_tokens")
    if isinstance(model_limit, int) and not isinstance(model_limit, bool) and model_limit > 0:
        limit = min(limit, model_limit)
    return limit_output(prepared, limit, config)


def context_exhausted(error: EngineError, config: Config, count: TokenCount | None) -> bool:
    if error.kind == "context":
        return True
    reply = error.reply
    if not reply or reply.finish_reason not in {"length", "incomplete", "max_tokens", "MAX_TOKENS"}:
        return False
    usage = reply.usage
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", usage.get("promptTokenCount", usage.get("prompt_eval_count"))))
    generated = usage.get("completion_tokens", usage.get("output_tokens", usage.get("eval_count")))
    if generated is None and isinstance(usage.get("candidatesTokenCount"), int):
        generated = usage["candidatesTokenCount"] + (usage.get("thoughtsTokenCount") or 0)
    if isinstance(prompt, int) and isinstance(generated, int):
        if prompt + generated >= config.context_window_tokens - max(256, config.context_window_tokens // 100):
            return True
    # A context-sized cap can stop output before the safety margin is used.
    # Require evidence that generation actually used that room. An empty answer
    # cut off by a smaller user/provider limit is not evidence of context overflow.
    return (count is not None and isinstance(generated, int)
            and generated >= max(1, available_output(config, count)))
