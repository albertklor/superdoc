"""Compatibility shim for LayoutLMv3 config.

Use the official Hugging Face config defaults/behavior directly.
"""

from transformers import LayoutLMv3Config as HFLayoutLMv3Config


class LayoutLMv3Config(HFLayoutLMv3Config):
    """Alias of `transformers.LayoutLMv3Config` for package compatibility."""


__all__ = ["LayoutLMv3Config"]
