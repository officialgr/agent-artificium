#!/usr/bin/env python3
"""Repository-root launcher and import-compatible source-tree shim."""

from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent / "artificium-code"
PACKAGE_ROOT = CODE_ROOT / "artificium"


if __name__ == "__main__":
    sys.path.insert(0, str(CODE_ROOT))
    from artificium.cli import main

    raise SystemExit(main())

__path__ = [str(PACKAGE_ROOT)]
VERSION = "1.9.2-revolution"

from artificium.interactions import ArtificiumClient  # noqa: E402
from artificium.runtime import Artificium  # noqa: E402

__all__ = ["Artificium", "ArtificiumClient", "VERSION"]
