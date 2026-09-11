from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.cli import _setup_options, build_parser
from artificium.config import Config, ConfigStore
from artificium.context_budget import (TokenCount, available_output, budget_request,
                                      context_exhausted, measure)
from artificium.engine import Engine, EngineError, EngineReply, make_engine
from artificium.filesystem import Paths, read_json, read_jsonl
from artificium.memory import TokenEstimator
from artificium.records import Console
from artificium.recovery import helper_config, source_excerpt, summarize
from artificium.runtime import Artificium
from artificium.setup import SetupOptions, SetupWizard


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = "Completed the first calculation and saved results in mind/space/result.txt. The next step is to check the remaining conjecture; do not rerun the completed action."


class SequenceEngine(Engine):
    def __init__(self, *replies, count=None):
        self.replies = list(replies)
        self.count = count
        self.requests = []

    def count_input_tokens(self, prepared):
        return self.count

    def complete(self, messages):
        self.requests.append(messages)
        result = self.replies.pop(0) if self.replies else EngineReply("<think>Continue.</think>")
        if isinstance(result, Exception):
            raise result
        return result


class CountingCase(unittest.TestCase):
    def config(self, provider="llamacpp", **values):
        return Config(provider=provider, model="local-model", **values)

    def test_native_count_routes_receive_the_rendered_request_without_generation(self):
        for provider, suffix, response in (
            ("llamacpp", "/chat/completions/input_tokens", {"input_tokens": 101376}),
            ("vllm", "/tokenize", {"count": 101376}),
            ("openai", "/responses/input_tokens", {"input_tokens": 101376}),
            ("anthropic", "/messages/count_tokens", {"input_tokens": 101376}),
            ("gemini", ":countTokens", {"totalTokens": 101376}),
        ):
            with self.subTest(provider=provider):
                engine = make_engine(self.config(provider), "secret")
                prepared = engine.prepare([{"role": "system", "content": "system text"},
                                           {"role": "user", "content": "literal <__media__> and source data"}])
                before = json.dumps(prepared.payload)
                with mock.patch("artificium.engine.request_json", return_value=response) as request:
                    count = measure(engine, prepared, [], TokenEstimator())
                self.assertEqual(count, TokenCount(101376, "provider"))
                args, kwargs = request.call_args
                self.assertTrue(args[0].endswith(suffix))
                self.assertIn("source data", json.dumps(kwargs["payload"]))
                self.assertEqual(kwargs["attempts"], 1)
                self.assertEqual(json.dumps(prepared.payload), before)
                if provider == "llamacpp":
                    self.assertEqual(kwargs["payload"], prepared.payload)
                    self.assertNotIn("<__media__>", json.dumps(kwargs["payload"]))

    def test_legacy_llama_counts_template_but_never_counts_image_placeholders_as_exact(self):
        engine = make_engine(self.config(), None)
        messages = [{"role": "user", "content": "Count this"}]
        prepared = engine.prepare(messages)
        with mock.patch("artificium.engine.request_json", side_effect=[
            EngineError("missing", status=404), {"prompt": "<start>Count this"}, {"tokens": [1, 2, 3]},
        ]) as request:
            self.assertEqual(engine.count_input_tokens(prepared), 3)
            self.assertTrue(request.call_args_list[1].args[0].endswith("/apply-template"))
            self.assertEqual(request.call_args.kwargs["payload"]["content"], "<start>Count this")
        media = dataclasses.replace(prepared, payload={"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}]})
        with mock.patch("artificium.engine.request_json") as request:
            self.assertIsNone(engine.count_input_tokens(media))
            request.assert_not_called()

    def test_missing_count_api_and_unknown_provider_keep_the_old_estimator(self):
        messages = [{"role": "user", "content": "a large amount of text " * 100}]
        engine = make_engine(self.config("vllm"), None)
        with mock.patch("artificium.engine.request_json", side_effect=EngineError("missing", status=404)) as request:
            for _ in range(2):
                self.assertEqual(measure(engine, engine.prepare(messages), messages, TokenEstimator()),
                                 TokenCount(TokenEstimator().messages(messages), "estimate"))
            self.assertEqual(request.call_count, 1)
        for provider in ("ollama", "openrouter"):
            engine = make_engine(self.config(provider), "secret")
            with mock.patch("artificium.engine.request_json") as request:
                self.assertEqual(measure(engine, engine.prepare(messages), messages, TokenEstimator()).source, "estimate")
                request.assert_not_called()

    def test_input_rejection_from_count_endpoint_is_preserved(self):
        engine = make_engine(self.config(), None)
        with mock.patch("artificium.engine.request_json", side_effect=EngineError("too many tokens", kind="context")):
            with self.assertRaises(EngineError):
                engine.count_input_tokens(engine.prepare([{"role": "user", "content": "source"}]))

    def test_materialized_image_count_and_generation_use_the_same_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"original image bytes")
            config = self.config()
            engine = make_engine(config, None)
            messages = [{"role": "user", "content": [{"type": "artificium_image", "path": str(path), "mime": "image/png"}]}]
            prepared = engine.prepare(messages)
            with mock.patch("artificium.engine.request_json", return_value={"input_tokens": 1500}) as count_call:
                count = measure(engine, prepared, messages, TokenEstimator())
            path.write_bytes(b"changed after preflight")
            response = {"choices": [{"message": {"content": "received"}, "finish_reason": "stop"}]}
            with mock.patch.object(engine, "_post", return_value=response) as inference:
                engine.complete_prepared(budget_request(prepared, config, count))
            self.assertEqual(inference.call_args.args[0].payload["messages"], count_call.call_args.kwargs["payload"]["messages"])

    def test_generation_caps_share_room_with_reasoning_and_do_not_mutate_config(self):
        for provider, location in (
            ("llamacpp", ("max_tokens",)), ("vllm", ("max_tokens",)),
            ("openrouter", ("max_tokens",)), ("openai", ("max_output_tokens",)),
            ("anthropic", ("max_tokens",)), ("gemini", ("generationConfig", "maxOutputTokens")),
            ("ollama", ("options", "num_predict")),
        ):
            with self.subTest(provider=provider):
                config = self.config(provider, context_window_tokens=102400, max_output_tokens=20000)
                engine = make_engine(config, "key")
                prepared = engine.prepare([{"role": "user", "content": "test"}])
                limited = budget_request(prepared, config, TokenCount(80000, "provider"))
                value = limited.payload
                for key in location:
                    value = value[key]
                self.assertEqual(value, 20000)
                limited = budget_request(prepared, config, TokenCount(98000, "provider"))
                value = limited.payload
                for key in location:
                    value = value[key]
                self.assertLess(value + 98000, config.context_window_tokens)
                self.assertEqual(config.max_output_tokens, 20000)
                with self.assertRaises(EngineError):
                    budget_request(prepared, config, TokenCount(103087, "provider"))

    def test_unset_output_limit_uses_remaining_context_instead_of_a_fixed_default(self):
        for provider, location in (
            ("llamacpp", ("max_tokens",)), ("vllm", ("max_tokens",)),
            ("openrouter", ("max_tokens",)), ("openai", ("max_output_tokens",)),
            ("anthropic", ("max_tokens",)), ("gemini", ("generationConfig", "maxOutputTokens")),
            ("ollama", ("options", "num_predict")),
        ):
            for context, expected in ((100000, 39000), (1_000_000, 930000)):
                with self.subTest(provider=provider, context=context):
                    config = self.config(provider, context_window_tokens=context,
                                         working_memory_tokens=100000)
                    prepared = make_engine(config, "key").prepare([{"role": "user", "content": "test"}])
                    limited = budget_request(prepared, config, TokenCount(60000, "provider"))
                    value = limited.payload
                    for key in location:
                        value = value[key]
                    self.assertEqual(value, expected)
                    self.assertIsNone(config.max_output_tokens)

    def test_reported_model_limits_and_explicit_choices_are_preserved(self):
        for provider, location in (
            ("llamacpp", ("max_tokens",)), ("openrouter", ("max_tokens",)),
            ("openai", ("max_output_tokens",)), ("anthropic", ("max_tokens",)),
            ("gemini", ("generationConfig", "maxOutputTokens")),
        ):
            for chosen, reported, expected in (
                (None, 65536, 65536), (100000, 131072, 100000),
                (4096, 65536, 4096), (100000, 65536, 65536),
            ):
                with self.subTest(provider=provider, chosen=chosen, reported=reported):
                    config = self.config(provider, context_window_tokens=1_000_000,
                                         max_output_tokens=chosen,
                                         model_capabilities={"max_output_tokens": reported})
                    prepared = make_engine(config, "key").prepare([{"role": "user", "content": "test"}])
                    value = budget_request(prepared, config, TokenCount(60000, "provider")).payload
                    for key in location:
                        value = value[key]
                    self.assertEqual(value, expected)
                    self.assertEqual(config.max_output_tokens, chosen)

    def test_expert_output_limit_and_normal_reasoning_are_not_replaced(self):
        for chosen in (4096, 100000):
            config = self.config(context_window_tokens=1_000_000, reasoning_effort="medium",
                                 request_options={"max_completion_tokens": chosen})
            prepared = make_engine(config, None).prepare([{"role": "user", "content": "test"}])
            body = budget_request(prepared, config, TokenCount(60000, "provider")).payload
            self.assertEqual(body["max_completion_tokens"], chosen)
            self.assertNotIn("max_tokens", body)
            self.assertEqual(body["reasoning_effort"], "medium")
            self.assertEqual(config.reasoning_effort, "medium")

    def test_large_reasoning_budget_is_allowed_when_it_fits(self):
        config = self.config("anthropic", context_window_tokens=200000,
                             reasoning_budget_tokens=50000,
                             model_capabilities={"max_output_tokens": 131072})
        prepared = make_engine(config, "key").prepare([{"role": "user", "content": "test"}])
        body = budget_request(prepared, config, TokenCount(60000, "provider")).payload
        self.assertEqual(body["thinking"]["budget_tokens"], 50000)
        self.assertEqual(body["max_tokens"], 131072)
        with self.assertRaises(EngineError):
            budget_request(prepared, config, TokenCount(160000, "provider"))

    def test_empty_reply_retains_usage_and_context_failure_is_distinct_from_output_limit(self):
        config = self.config(context_window_tokens=102400)
        engine = make_engine(config, None)
        response = {"choices": [{"message": {"content": "", "reasoning_content": "unfinished"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 100382, "completion_tokens": 2018}}
        with mock.patch.object(engine, "_post", return_value=response):
            with self.assertRaises(EngineError) as caught:
                engine.complete([{"role": "user", "content": "test"}])
        self.assertEqual(caught.exception.reply.raw, response)
        self.assertTrue(context_exhausted(caught.exception, config, TokenCount(81000, "estimate")))
        caught.exception.reply.usage = {"prompt_tokens": 1000, "completion_tokens": 1024}
        self.assertFalse(context_exhausted(caught.exception, config, TokenCount(1000, "provider")))
        # Estimates leave a larger margin, but using all remaining output room
        # is still a context-related failure. A smaller output cap is not.
        count = TokenCount(80000, "estimate")
        caught.exception.reply.usage = {"prompt_tokens": 80000, "completion_tokens": available_output(config, count)}
        self.assertTrue(context_exhausted(caught.exception, config, count))
        caught.exception.reply.usage = {"prompt_tokens": 80000, "completion_tokens": 1024}
        self.assertFalse(context_exhausted(caught.exception, config, count))
        caught.exception.reply.usage = {}
        self.assertFalse(context_exhausted(caught.exception, config, count))

    def test_working_memory_tracks_or_decouples_without_increasing_server_context(self):
        config = self.config(context_window_tokens=1_000_000)
        self.assertEqual(config.working_memory_limit, 1_000_000)
        config = dataclasses.replace(config, working_memory_tokens=100000)
        self.assertEqual(config.chunk_tokens("broad"), 50000)
        self.assertEqual(config.carry_tokens("broad"), 12000)
        saved = Config.from_dict(config.grouped_dict())
        self.assertEqual(saved.working_memory_limit, 100000)
        self.assertNotIn("working_memory_tokens", saved.grouped_dict()["model"])
        self.assertEqual(dataclasses.replace(config, working_memory_tokens=None).working_memory_limit, 1_000_000)
        with self.assertRaises(ValueError):
            dataclasses.replace(config, working_memory_tokens=1_000_001)


class RecoveryCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        self.paths.ensure_layout()
        shutil.copytree(ROOT / "artificium-code/prompts", self.paths.prompts)
        shutil.copyfile(ROOT / "mind/tools/scheduler.py", self.paths.created_tools / "scheduler.py")
        self.config = Config(provider="llamacpp", model="test-model", context_window_tokens=102400,
                             max_life_loop_rounds=2, emergency_offload=True)
        ConfigStore(self.paths).save(self.config)
        self.network = mock.patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def agent(self, engine=None):
        return Artificium(self.paths, engine=engine or SequenceEngine(), console=Console(quiet=True))

    def test_helper_is_isolated_same_api_no_tools_and_bounded(self):
        agent = self.agent()
        config = dataclasses.replace(self.config, reasoning_effort="medium", temperature=0.3,
                                     request_options={"tools": [{"type": "web_search"}], "previous_response_id": "old",
                                                      "provider": {"order": ["provider-a"]}})
        captured = []
        def post(engine_self, prepared):
            captured.append((engine_self.config, prepared))
            return {"choices": [{"message": {"content": SUMMARY}, "finish_reason": "stop"}]}
        with mock.patch("artificium.engine.JSONEngine._post", post), \
             mock.patch("artificium.engine.JSONEngine.count_input_tokens", return_value=None):
            summary, omitted = summarize(config=config, api_key="the-key",
                history=[{"role": "user", "content": "Research record " * 40000}],
                original_log="original.json", attempt=1, prompts=agent.prompts, records=agent.records)
        helper, request = captured[0]
        self.assertEqual((helper.model, helper.base_url, helper.endpoint, helper.context_window_tokens),
                         (config.model, config.base_url, config.endpoint, config.context_window_tokens))
        self.assertEqual(helper.temperature, 0.3)
        self.assertEqual(request.headers["Authorization"], "Bearer the-key")
        self.assertEqual(request.payload["provider"]["order"], ["provider-a"])
        self.assertNotIn("tools", request.payload)
        self.assertNotIn("previous_response_id", request.payload)
        self.assertNotIn("Artificium", request.payload["messages"][0]["content"])
        self.assertNotIn(agent.paths.self_file.read_text().strip(), json.dumps(request.payload))
        self.assertEqual(request.payload["chat_template_kwargs"]["enable_thinking"], False)
        self.assertLessEqual(request.payload["max_tokens"], 2048)
        self.assertTrue(omitted)
        self.assertEqual(summary, SUMMARY)
        self.assertEqual(config.reasoning_effort, "medium")

    def test_context_failure_recovers_without_replaying_actions_or_losing_messages(self):
        engine = SequenceEngine(EngineError("request (103087 tokens) exceeds context", status=400, kind="context"),
                                EngineReply("<think>Resume from the checkpoint.</think>"))
        agent = self.agent(engine)
        agent.working.append({"role": "assistant", "content": "A completed action wrote result.txt; preserve that evidence."}, origin="test")
        previous = self.paths.working_context.read_bytes()
        self_before = self.paths.self_file.read_bytes()
        artifact = self.paths.space / "result.txt"
        artifact.write_text("completed once")
        notification = agent.notifications.create(type="interaction", source="user", summary="A new user message")
        def during_summary(**kwargs):
            for index in range(49):
                agent.notifications.create(type="interaction", source="user", summary=f"Message while recovering {index}")
            return SUMMARY, True
        with mock.patch("artificium.runtime.summarize", side_effect=during_summary) as helper:
            agent.run_turn()
        self.assertEqual(helper.call_count, 1)
        self.assertEqual(artifact.read_text(), "completed once")
        self.assertEqual(self.paths.self_file.read_bytes(), self_before)
        archives = list(self.paths.context_archive.glob("*.jsonl"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), previous)
        resumed = json.dumps(engine.requests[-1])
        self.assertIn("EMERGENCY WORKING-MEMORY OFFLOAD", resumed)
        self.assertIn("A new user message", resumed)
        self.assertIn("checkpoint", resumed.lower())
        self.assertFalse(agent._emergency_state_path.exists())
        self.assertFalse(list(self.paths.notifications_new.glob(f"{notification.id}.json")))
        self.assertFalse(list(self.paths.notifications_processing.glob(f"{notification.id}.json")))
        preserved = [read_json(path) for directory in (self.paths.notifications_new, self.paths.notifications_delivered)
                     for path in directory.glob("*.json")]
        user_messages = [item for item in preserved if item.get("source") == "user"]
        self.assertEqual(len(user_messages), 50)
        self.assertEqual(len({item["summary"] for item in user_messages}), 50)

    def test_truncated_helper_outputs_are_logged_and_never_committed(self):
        agent = self.agent(SequenceEngine(EngineError("context overflow", kind="context")))
        agent.working.append({"role": "user", "content": "Keep all of this"}, origin="test")
        before = self.paths.working_context.read_bytes()
        raw = {"choices": [{"message": {"content": SUMMARY}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 2000, "completion_tokens": 2048}}
        with mock.patch("artificium.engine.JSONEngine._post", return_value=raw) as inference, \
             mock.patch("artificium.engine.JSONEngine.count_input_tokens", return_value=2000):
            with self.assertRaisesRegex(EngineError, "after 3 attempts"):
                agent.run_turn()
        self.assertEqual(inference.call_count, 3)
        self.assertEqual(self.paths.working_context.read_bytes(), before)
        logs = list(self.paths.model_log.glob("emergency_summary_*.json"))
        self.assertEqual(len(logs), 3)
        self.assertEqual(read_json(logs[0])["response"]["raw"], raw)

    def test_disabled_recovery_preserves_history_and_releases_claims(self):
        ConfigStore(self.paths).save(dataclasses.replace(self.config, emergency_offload=False))
        agent = self.agent(SequenceEngine(EngineError("too many tokens", kind="context")))
        agent.working.append({"role": "user", "content": "Keep this exact history"}, origin="test")
        before = self.paths.working_context.read_bytes()
        agent.notifications.create(type="interaction", source="user", summary="Keep pending")
        with mock.patch("artificium.runtime.summarize") as helper, self.assertRaises(EngineError):
            agent.run_turn()
        helper.assert_not_called()
        self.assertEqual(self.paths.working_context.read_bytes(), before)
        self.assertEqual(len(list(self.paths.notifications_new.glob("*.json"))), 1)

    def test_three_failed_attempts_survive_restart_and_do_not_replace_context(self):
        agent = self.agent(SequenceEngine(EngineError("context overflow", kind="context")))
        agent.working.append({"role": "user", "content": "Original history"}, origin="test")
        before = self.paths.working_context.read_bytes()
        agent.notifications.create(type="interaction", source="user", summary="Message retained")
        with mock.patch("artificium.runtime.summarize", side_effect=EngineError("helper failed")) as helper:
            with self.assertRaisesRegex(EngineError, "after 3 attempts"):
                agent.run_turn()
            self.assertEqual(helper.call_count, 3)
            restarted = self.agent(SequenceEngine(EngineError("context overflow", kind="context")))
            with self.assertRaisesRegex(EngineError, "after 3 attempts"):
                restarted.run_turn()
            self.assertEqual(helper.call_count, 3)
        self.assertEqual(self.paths.working_context.read_bytes(), before)
        self.assertFalse(list(self.paths.context_archive.glob("*.jsonl")))
        self.assertEqual(len(list(self.paths.notifications_new.glob("*.json"))), 1)

    def test_native_count_triggers_offload_when_character_estimate_is_low(self):
        ConfigStore(self.paths).save(dataclasses.replace(self.config, mandatory_offload=True, max_life_loop_rounds=1))
        engine = SequenceEngine(count=85000)
        agent = self.agent(engine)
        agent.run_turn()
        self.assertIn("MANDATORY WORKING-MEMORY OFFLOADING", json.dumps(engine.requests[0]))
        context = [item for item in read_jsonl(self.paths.life_loop_log) if item["kind"] == "context_usage"][0]
        self.assertEqual((context["estimated_tokens"], context["token_count_source"]), (85000, "provider"))

    def test_oversized_pinned_prompt_cannot_be_fixed_by_erasing_working_history(self):
        agent = self.agent(SequenceEngine(count=103087))
        agent.working.append({"role": "user", "content": "Keep original"}, origin="test")
        before = self.paths.working_context.read_bytes()
        with mock.patch("artificium.runtime.summarize", return_value=(SUMMARY, False)) as helper:
            with self.assertRaisesRegex(EngineError, "after 3 attempts"):
                agent.run_turn()
        self.assertEqual(helper.call_count, 3)
        self.assertEqual(self.paths.working_context.read_bytes(), before)

    def test_harness_cli_can_decouple_and_resync_without_probing_or_editing_self(self):
        agent = self.agent()
        self_before = self.paths.self_file.read_bytes()
        parser = build_parser()
        args = parser.parse_args(["configure", "harness", "--working-memory-tokens", "50000", "--emergency-offload", "on", "--yes"])
        updated = SetupWizard(self.paths).reconfigure(_setup_options(args))
        self.assertEqual((updated.working_memory_limit, updated.context_window_tokens), (50000, 102400))
        self.assertTrue(updated.emergency_offload)
        updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope="harness", working_memory_tokens="same"))
        self.assertIsNone(updated.working_memory_tokens)
        self.assertEqual(self.paths.self_file.read_bytes(), self_before)


if __name__ == "__main__":
    unittest.main()
