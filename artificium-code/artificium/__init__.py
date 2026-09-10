"""Artificium-revolution — continual learning and Infinite Attention."""

VERSION = "1.9.3-revolution"

from .interactions import ArtificiumClient
from .runtime import Artificium

__all__ = ["Artificium", "ArtificiumClient", "VERSION"]
