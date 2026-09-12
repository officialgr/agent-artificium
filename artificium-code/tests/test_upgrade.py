from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.cli import main
from artificium.filesystem import Paths
from artificium.initialization import initialize_mind
from artificium.records import Records
from artificium.upgrade import _files, _replace, upgrade


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("git"), "Git is required for upgrade integration tests")
class UpgradeCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "upstream"
        self.source.mkdir()
        for name in ("artificium-code", "scripts", "mind"):
            shutil.copytree(ROOT / name, self.source / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("artificium.py", "README.md", ".gitignore"):
            shutil.copy2(ROOT / name, self.source / name)
        self.git("init", "-q", "-b", "main")
        self.commit()
        self.paths = Paths(self.directory / "instance")
        shutil.copytree(self.source, self.paths.root, ignore=shutil.ignore_patterns(".git"))
        self.paths.ensure_layout()
        self.protected = {
            "mind/self.txt": b"My learned purpose.\n",
            "mind/meta_memory.md": b"My learned memory map.\n",
            "mind/memory/harness/tool-building-and-workspace.txt": b"My learned operating strategy.\n",
            "mind/memory/project/new-discovery.txt": b"A new result.\n",
            "mind/tools/scheduler.py": (self.paths.created_tools / "scheduler.py").read_bytes(),
            "mind/tools/custom.py": b"# agent-built tool\n",
            "mind/space/result.txt": b"result\n",
            "mind/interactions/chat/events/event.json": b"{}\n",
            "artificium-code/config.json": b'{"temperature":0.3}\n',
            "artificium-code/.secrets.json": b'{"api_key":"private"}\n',
            "artificium-code/custom-contract.json": b'{"private":"contract"}\n',
            "artificium-code/prompts/.env.local": b"PRIVATE=value\n",
            ".env": b"PRIVATE=value\n",
            "logs/context/working.jsonl": b'{"content":"my current context"}\n',
        }
        for name, content in self.protected.items():
            path = self.paths.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (self.paths.code / "artificium/obsolete.py").write_text("OLD = True\n")
        (self.source / "artificium-code/artificium/new_feature.py").write_text("NEW = True\n")
        (self.source / "README.md").write_text("Updated documentation.\n")
        self.commit()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], stderr=subprocess.PIPE, text=True)

    def commit(self):
        self.git("add", "--all")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Fixture revision")

    def run_upgrade(self, **options):
        return upgrade(self.paths, repository=str(self.source), report=lambda _: None, **options)

    def assert_preserved(self):
        for name, content in self.protected.items():
            self.assertEqual((self.paths.root / name).read_bytes(), content, name)

    def snapshot(self):
        return {name: (path.read_bytes(), path.stat().st_mode & 0o777) for name, path in _files(self.paths.root).items()}

    def test_upgrade_changes_code_preserves_state_and_removes_old_seed_tree(self):
        old_seed = self.paths.prompts / "mind-seed/memory/harness/old-guide.txt"
        old_seed.parent.mkdir(parents=True)
        old_seed.write_text("Old duplicate.\n")
        result = self.run_upgrade()
        self.assertEqual((self.paths.code / "artificium/new_feature.py").read_text(), "NEW = True\n")
        self.assertFalse((self.paths.code / "artificium/obsolete.py").exists())
        self.assertFalse((self.paths.prompts / "mind-seed").exists())
        self.assert_preserved()
        backup = Path(result["backup"])
        self.assertEqual((backup / "files/artificium-code/artificium/obsolete.py").read_text(), "OLD = True\n")
        self.assertEqual(json.loads((backup / "upgrade.json").read_text())["status"], "completed")
        self.assertEqual(self.run_upgrade()["changed_files"], 0)

    def test_upstream_settings_are_never_installed(self):
        (self.source / "artificium-code/config.json").write_text('{"temperature":1}')
        self.git("add", "-f", "artificium-code/config.json")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Bad upstream settings")
        self.run_upgrade()
        self.assert_preserved()

    def test_check_is_read_only_even_while_agent_is_running(self):
        before = self.snapshot()
        with mock.patch("artificium.upgrade.process_state", return_value={"alive": True}):
            self.assertGreater(self.run_upgrade(check=True)["changed_files"], 0)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.paths.logs / "upgrades").exists())
        self.assert_preserved()

    def test_apply_refuses_a_running_agent_before_downloading(self):
        with mock.patch("artificium.upgrade.process_state", return_value={"alive": True}), mock.patch("artificium.upgrade.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "Stop Artificium"):
                self.run_upgrade()
            run.assert_not_called()

    def test_download_failure_does_not_change_code_or_state(self):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "Could not download"):
            self.run_upgrade(ref="missing-branch")
        self.assertEqual(self.snapshot(), before)
        self.assert_preserved()

    def test_invalid_source_is_rejected_before_replacing_files(self):
        (self.source / "artificium-code/artificium/new_feature.py").write_text("def broken(\n")
        self.commit()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Invalid Python"):
            self.run_upgrade()
        self.assertEqual(self.snapshot(), before)
        self.assert_preserved()

    def test_failed_replacement_restores_previous_code(self):
        (self.source / "artificium-code/artificium/z_finish.py").write_text("LAST = True\n")
        self.commit()
        before = self.snapshot()
        count = 0
        def replace(source, destination):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError("Simulated write failure")
            _replace(source, destination)
        with mock.patch("artificium.upgrade._replace", side_effect=replace):
            with self.assertRaisesRegex(RuntimeError, "code was restored"):
                self.run_upgrade()
        self.assertEqual(self.snapshot(), before)
        self.assert_preserved()

    def test_new_launcher_can_upgrade_an_old_installation(self):
        (self.paths.root / "artificium.py").write_text("raise SystemExit('Old launcher has no upgrade command')\n")
        (self.paths.code / "artificium/upgrade.py").unlink()
        result = subprocess.run([sys.executable, str(ROOT / "artificium.py"),
                                 "--root", str(self.paths.root), "upgrade", "--repo", str(self.source)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Code backup:", result.stdout)
        self.assertTrue((self.paths.code / "artificium/upgrade.py").is_file())
        self.assert_preserved()

    def test_symlinked_destination_is_not_followed(self):
        outside = self.directory / "outside.py"
        outside.write_text("PRIVATE = True\n")
        (self.paths.code / "artificium/new_feature.py").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_upgrade()
        self.assertEqual(outside.read_text(), "PRIVATE = True\n")
        self.assert_preserved()

    def test_deleted_memory_is_not_recreated_at_upgrade_or_startup(self):
        deleted = self.paths.memory / "harness/infinite-attention.txt"
        deleted.unlink()
        self.run_upgrade()
        initialize_mind(self.paths, Records(self.paths))
        self.assertFalse(deleted.exists())
        self.assert_preserved()

    def test_cli_can_upgrade_a_zip_install_without_configuration(self):
        self.paths.config.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--root", str(self.paths.root), "upgrade", "--repo", str(self.source)])
        self.assertEqual(result, 0)
        self.assertFalse(self.paths.config.exists())
        self.assertTrue((self.paths.code / "artificium/new_feature.py").is_file())

    def test_missing_mind_is_reported_without_inventing_a_replacement(self):
        self.paths.meta_memory.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "complete installation includes mind"):
            initialize_mind(self.paths, Records(self.paths))
        self.assertFalse(self.paths.meta_memory.exists())
