#!/usr/bin/env python3
"""Build a clean source ZIP without exporting a running instance's private mind."""
from __future__ import annotations

import argparse
import tomllib
import zipfile
from pathlib import Path


def release_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in ('README.md', '.gitignore', 'artificium.py'):
        files[name] = root / name
    for folder in ('artificium-code', 'scripts', '.github'):
        for path in (root / folder).rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root)
            if any(part in {'__pycache__', '.pytest_cache', 'build', 'dist'} or part.endswith('.egg-info') for part in rel.parts):
                continue
            if path.suffix in {'.pyc', '.pyo'} or path.name in {'config.json', '.secrets.json'} or path.name.startswith('.env'):
                continue
            # Public source formats only. Unknown files must be deliberately added.
            if path.suffix not in {'.py', '.md', '.txt', '.toml', '.json', '.yml', '.yaml'}:
                continue
            files[rel.as_posix()] = path
    prompts = root / 'artificium-code/prompts'
    manifest = tomllib.loads((prompts / 'manifest.toml').read_text())
    files['mind/self.txt'] = prompts / manifest['mind_seed']['self']
    files['mind/meta_memory.md'] = prompts / manifest['mind_seed']['meta_memory']
    for item in manifest['seed_memory']:
        files['mind/memory/' + item['path']] = prompts / item['source']
    files['mind/tools/scheduler.py'] = root / 'mind/tools/scheduler.py'
    for path in files.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or symlinked release input: {path}')
    return files


def build(root: Path, destination: Path) -> Path:
    version = tomllib.loads((root / 'artificium-code/pyproject.toml').read_text())['project']['version']
    prefix = f'Artificium-revolution-v{version}/'
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = release_files(root)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, source in sorted(files.items()):
            info = zipfile.ZipInfo(prefix + relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
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
    print(build(root, parser.parse_args().output.resolve()))
