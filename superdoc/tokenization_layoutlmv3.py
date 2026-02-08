"""Compatibility shim for LayoutLMv3 tokenizer.

Use the official Hugging Face fast tokenizer implementation directly.
"""

from transformers import LayoutLMv3TokenizerFast as HFLayoutLMv3TokenizerFast

# Keep the historical local API where both names resolve to the fast tokenizer.
LayoutLMv3Tokenizer = HFLayoutLMv3TokenizerFast
LayoutLMv3TokenizerFast = HFLayoutLMv3TokenizerFast

__all__ = ["LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast"]
