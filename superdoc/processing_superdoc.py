"""Processor class for SuperDoc."""

from typing import Optional, Union

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


class SuperDocProcessor(ProcessorMixin):
    r"""
    Constructs a SuperDoc processor which combines a SuperDoc image processor and a SuperDoc tokenizer into a
    single processor.

    [`SuperDocProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`SuperDocImageProcessor`] to resize and normalize document images. The words and bounding boxes
    must be provided by the user. These are then provided to [`SuperDocTokenizer`] or [`SuperDocTokenizerFast`],
    which turns the words and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
    Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
    classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`SuperDocImageProcessor`, *optional*):
            An instance of [`SuperDocImageProcessor`]. The image processor is a required input.
        tokenizer (`SuperDocTokenizer` or `SuperDocTokenizerFast`, *optional*):
            An instance of [`SuperDocTokenizer`] or [`SuperDocTokenizerFast`]. The tokenizer is a required input.
    """

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, list[PreTokenizedInput]]] = None,
        boxes: Union[list[list[int]], list[list[list[int]]]] = None,
        word_labels: Optional[Union[list[int], list[list[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method first forwards the `images` argument to [`~SuperDocImageProcessor.__call__`] to get resized and
        normalized `pixel_values`. It then passes the words (`text`/`text_pair`) and `boxes` to
        [`~SuperDocTokenizer.__call__`] and returns the output together with the pixel values.

        Please refer to the docstring of the above two methods for more information.

        Args:
            images: The document images to process.
            text: The words/text to tokenize.
            text_pair: Optional second sequence for sequence pairs.
            boxes: The bounding boxes corresponding to the words (normalized to 0-1000 scale).
            word_labels: Optional word-level labels for token classification tasks.
        """
        if boxes is None:
            raise ValueError("You must provide bounding boxes for the words.")

        # First, apply the image processor
        features = self.image_processor(images=images, return_tensors=return_tensors)

        # Second, apply the tokenizer
        encoded_inputs = self.tokenizer(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )

        # Add pixel values
        images = features.pop("pixel_values")
        if return_overflowing_tokens is True:
            images = self.get_overflowing_images(images, encoded_inputs["overflow_to_sample_mapping"])
        encoded_inputs["pixel_values"] = images

        return encoded_inputs

    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # In case there's an overflow, ensure each `input_ids` sample is mapped to its corresponding image
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow

    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "pixel_values"]


__all__ = ["SuperDocProcessor"]
