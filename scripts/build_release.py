#!/usr/bin/env python3
"""Build a reproducible release from committed source, never from a live mind."""
from __future__ import annotations

import argparse
import io
import stat
import subprocess
import tomllib
import zipfile
from pathlib import Path


def release_files(root: Path, ref: str = "HEAD") -> dict[str, bytes]:
    if not ref or ref.startswith("-"):
        raise ValueError("Provide a Git revision, not an option")
    try:
        top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], stderr=subprocess.PIPE, text=True)
        if Path(top.strip()).resolve() != root.resolve():
            raise ValueError("Build releases from an Artificium Git checkout; commit intended changes first.")
        data = subprocess.check_output(["git", "-C", str(root), "archive", "--format=zip", ref], stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Build releases from an Artificium Git checkout; commit intended changes first.") from error
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        for item in source.infolist():
            path = Path(item.filename)
            if item.is_dir() or path.name in {"config.json", ".secrets.json"} or path.name.startswith(".env"):
                continue
            allowed = (item.filename in {"README.md", ".gitignore", "artificium.py", "mind/self.txt", "mind/meta_memory.md", "mind/tools/scheduler.py"}
                       or path.parts[0] in {"artificium-code", "scripts", ".github"}
                       or item.filename.startswith(("mind/memory/harness/", "mind/memory/tools/")))
            if not allowed or (path.suffix not in {".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml"} and item.filename != ".gitignore"):
                continue
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Symlinked release input: {item.filename}")
            files[item.filename] = source.read(item)
    return files


def build(root: Path, destination: Path, ref: str = "HEAD") -> Path:
    files = release_files(root, ref)
    version = tomllib.loads(files['artificium-code/pyproject.toml'].decode())['project']['version']
    prefix = f'Artificium-revolution-v{version}/'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, content in sorted(files.items()):
            info = zipfile.ZipInfo(prefix + relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
        for name in ('logs/.gitkeep', 'mind/interactions/.gitkeep', 'mind/space/.gitkeep'):
            archive.writestr(zipfile.ZipInfo(prefix + name, date_time=(2026, 1, 1, 0, 0, 0)), b'')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise ValueError('Release ZIP integrity check failed')
    return destination


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root.parent / f'{root.name}-source.zip')
    parser.add_argument('--ref', default='HEAD', help='Committed revision to package (default: HEAD)')
    args = parser.parse_args()
    print(build(root, args.output.resolve(), args.ref))
