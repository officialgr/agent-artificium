from __future__ import annotations

import io
import json
import shlex
import shutil
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from artificium.cli import _setup_options, build_parser
from artificium.config import Config, ConfigStore
from artificium.engine import Engine, EngineError, EngineReply, make_engine, request_json
from artificium.filesystem import Paths
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.setup import ModelDiscovery, SetupOptions, SetupWizard
from artificium.setup_ui import edit_request_timeout, model_editor


ROOT = Path(__file__).resolve().parents[2]


class TimeoutConfigCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        self.paths.ensure_layout()
        shutil.copytree(ROOT / "artificium-code/prompts", self.paths.prompts)
        self.wizard = SetupWizard(self.paths)
        for name, value in (("_probe", ModelDiscovery()), ("_details", {})):
            patch = mock.patch.object(self.wizard, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_default_off_and_explicit_numeric_settings_round_trip(self):
        store = ConfigStore(self.paths)
        for value, expected in ((None, None), ("off", None), ("OFF", None),
                                (600, 600), ("1800", 1800), (86400, 86400)):
            with self.subTest(value=value):
                config = Config(provider="llamacpp", model="test", request_timeout_seconds=value)
                store.save(config)
                self.assertEqual(store.load().request_timeout_seconds, expected)
                saved = json.loads(self.paths.config.read_text())
                self.assertEqual(saved["model"]["request_timeout_seconds"], expected)
                self.assertEqual(config.shell_timeout_seconds, 120)
                self.assertEqual(config.max_shell_timeout_seconds, 3600)
        self.assertIsNone(Config(provider="llamacpp", model="test").request_timeout_seconds)
        self.assertEqual(Config.from_dict({"provider": "llamacpp", "model": "test", "request_timeout_seconds": 600}).request_timeout_seconds, 600)

    def test_invalid_timeouts_cannot_disable_waiting_accidentally(self):
        for value in (0, -1, 86401, "nan", "inf", "never", "", True, False):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Request timeout"):
                Config(provider="llamacpp", model="test", request_timeout_seconds=value)

    def test_cli_can_disable_or_set_timeout_and_unrelated_edits_preserve_it(self):
        current = Config(provider="llamacpp", model="test", request_timeout_seconds=600)
        for arguments, expected in (([], 600), (["--temperature", "0.3"], 600),
                                    (["--request-timeout", "off"], None),
                                    (["--request-timeout", "1800"], 1800)):
            with self.subTest(arguments=arguments):
                args = build_parser().parse_args(["configure", "model", *arguments, "--yes"])
                options = _setup_options(args)
                config, _ = self.wizard._build(options, current)
                self.assertEqual(config.request_timeout_seconds, expected)
                self.assertEqual(current.request_timeout_seconds, 600)
        options = _setup_options(build_parser().parse_args(["configure", "harness", "--request-timeout", "off"]))
        with self.assertRaisesRegex(ValueError, "configure model"):
            self.wizard._validate_scope(options)

    def test_interactive_editor_displays_off_and_changes_both_directions(self):
        current = Config(provider="llamacpp", model="test")
        options = SetupOptions(scope="model")
        output = io.StringIO()
        with redirect_stdout(output), mock.patch("builtins.input", side_effect=["4", "1800", "4", "off", "done"]):
            model_editor(self.wizard, options, current)
        self.assertIn("off (wait indefinitely)", output.getvalue())
        self.assertIn("1800 seconds", output.getvalue())
        config, _ = self.wizard._build(options, current)
        self.assertIsNone(config.request_timeout_seconds)

    def test_timeout_editor_rejects_invalid_values_and_keeps_pending_choice(self):
        options = SetupOptions(request_timeout_seconds="off")
        with redirect_stdout(io.StringIO()), mock.patch("builtins.input", side_effect=["0", "nan", ""]):
            edit_request_timeout(options, Config(provider="llamacpp", model="test", request_timeout_seconds=600))
        self.assertEqual(options.request_timeout_seconds, "off")


class TimeoutTransportCase(unittest.TestCase):
    def test_inference_transports_pass_off_or_the_numeric_timeout_to_http(self):
        for provider in ("llamacpp", "vllm", "ollama", "openrouter", "openai", "anthropic", "gemini", "custom"):
            for timeout in (None, 1800):
                with self.subTest(provider=provider, timeout=timeout):
                    engine = make_engine(Config(provider=provider, model="test", base_url="http://localhost/v1", request_timeout_seconds=timeout), "test-key")
                    prepared = engine.prepare([{"role": "user", "content": "hello"}])
                    response = mock.MagicMock()
                    response.__enter__.return_value.read.return_value = b'{"ok": true}'
                    with mock.patch("urllib.request.urlopen", return_value=response) as opened:
                        self.assertEqual(engine._post(prepared), {"ok": True})
                    self.assertIs(opened.call_args.kwargs["timeout"], engine.config.request_timeout_seconds)

    def test_counting_checks_keep_short_timeouts_including_legacy_llama(self):
        for provider in ("llamacpp", "vllm", "openai", "anthropic", "gemini"):
            for timeout, expected in ((None, 10), (3, 3)):
                with self.subTest(provider=provider, timeout=timeout):
                    engine = make_engine(Config(provider=provider, model="test", request_timeout_seconds=timeout), "test-key")
                    prepared = engine.prepare([{"role": "user", "content": "hello"}])
                    with mock.patch("artificium.engine.request_json", return_value={"input_tokens": 17, "count": 17, "totalTokens": 17}) as request:
                        self.assertEqual(engine.count_input_tokens(prepared), 17)
                    self.assertEqual(request.call_args.kwargs["timeout"], expected)
        engine = make_engine(Config(provider="llamacpp", model="test"), None)
        with mock.patch("artificium.engine.request_json", side_effect=[
            EngineError("missing", status=404), {"prompt": "hello"}, {"tokens": [1, 2]},
        ]) as request:
            self.assertEqual(engine.count_input_tokens(engine.prepare([{"role": "user", "content": "hello"}])), 2)
        self.assertEqual([call.kwargs["timeout"] for call in request.call_args_list], [10, 10, 10])

    def test_connection_and_os_timeout_errors_still_surface_with_timeout_off(self):
        for failure, kind in ((ConnectionResetError("peer reset"), "network"),
                              (urllib.error.URLError(ConnectionRefusedError("refused")), "network"),
                              (TimeoutError("network timeout"), "timeout"),
                              (urllib.error.URLError(TimeoutError("connect timeout")), "timeout")):
            with self.subTest(failure=failure), mock.patch("urllib.request.urlopen", side_effect=failure) as opened:
                with self.assertRaises(EngineError) as caught:
                    request_json("http://localhost/model", payload={}, timeout=None, attempts=3)
                self.assertEqual(caught.exception.kind, kind)
                self.assertEqual(opened.call_count, 1)

    def test_rejected_requests_still_require_correction_with_timeout_off(self):
        failure = urllib.error.HTTPError("http://localhost/model", 401, "unauthorized", {},
                                         io.BytesIO(b'{"error":{"message":"Invalid API key"}}'))
        with mock.patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(EngineError) as caught:
                request_json("http://localhost/model", timeout=None)
        self.assertEqual(caught.exception.status, 401)
        self.assertTrue(caught.exception.requires_operator_action)


class ShellTimeoutCase(unittest.TestCase):
    def test_real_shell_timeout_reaches_the_next_model_request_with_guidance(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        paths = Paths(Path(temporary.name))
        paths.ensure_layout()
        shutil.copytree(ROOT / "artificium-code/prompts", paths.prompts)
        shutil.copytree(ROOT / "mind", paths.mind, dirs_exist_ok=True)
        ConfigStore(paths).save(Config(provider="llamacpp", model="test", max_life_loop_rounds=2))
        # exec replaces the test shell, so no descendant is left running.
        command = f"exec {shlex.quote(sys.executable)} -c 'import time; time.sleep(10)'"
        call = {"tool": "run_shell", "command": command, "timeout_seconds": 0.05}

        class RecordingEngine(Engine):
            def __init__(self):
                self.requests = []

            def complete(self, messages):
                self.requests.append(json.dumps(messages))
                return EngineReply("<tool_call>" + json.dumps(call) + "</tool_call>"
                                   if len(self.requests) == 1 else "<think>I can continue after the timeout.</think>")

        engine = RecordingEngine()
        agent = Artificium(paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="shell-timeout-test")
        self.assertEqual(len(engine.requests), 2)
        self.assertIn("TimeoutExpired", engine.requests[1])
        self.assertIn("SHELL TIMEOUT", engine.requests[1])
        self.assertIn("child processes", engine.requests[1])
        self.assertIn("launch it in the background", engine.requests[1])
        self.assertIn("recorded PID", engine.requests[1])
        self.assertIn("TOOL EXECUTION ERROR", engine.requests[1])


if __name__ == "__main__":
    unittest.main()
