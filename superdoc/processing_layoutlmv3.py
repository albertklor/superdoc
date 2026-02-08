"""Compatibility shim for LayoutLMv3 processor.

Use the official Hugging Face processor directly.
"""

from transformers import LayoutLMv3Processor

__all__ = ["LayoutLMv3Processor"]
