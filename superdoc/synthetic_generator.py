"""
Synthetic Document Image Generator for training LayoutLMv3-style models.

Converts text into (words, bounding boxes, image tensor) with optimizations
for use as a PyTorch DataLoader collator.

Features structural diversity: columns, paragraphs, headers, lists, indentation.
"""

import os
import math
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Default fonts to try on different platforms
DEFAULT_FONT_PATHS = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Avenir.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Georgia.ttf",
    "/System/Library/Fonts/Palatino.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Times New Roman.ttf",
    "/Library/Fonts/Verdana.ttf",
    "/Library/Fonts/Tahoma.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/cour.ttf",
]

# Bullet characters for lists
BULLET_CHARS = ["•", "-", "◦", "▪", "►", "*", "·"]


class DocumentGenerator:
    """
    Fast CPU-based synthetic document image generator with structural diversity.

    Generates document images with:
    - Multi-column layouts (1-3 columns)
    - Paragraphs with indentation
    - Headers/titles (larger font)
    - Bullet points and lists
    - Variable spacing
    - Random fonts, sizes, colors, shifts
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        font_paths: Union[str, List[str], None] = None,
        font_sizes: Union[int, List[int], Tuple[int, int]] = 12,
        canvas_sizes: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
        canvas_area_weight_power: float = 0.0,
        margin_ratio: float = 0.045,
        line_spacing: float = 1.2,
        font_scale_power: float = 1.0,
        rgb: bool = True,
        background_color: Union[int, Tuple[int, int], List[int]] = 255,
        text_color: Union[int, Tuple[int, int], List[int]] = 0,
        horizontal_shift: Union[float, Tuple[float, float]] = 0.0,
        # Structural diversity options
        num_columns: Union[int, List[int]] = [1, 1, 1, 2, 2, 3],
        paragraph_spacing: Union[float, Tuple[float, float]] = (1.0, 1.8),
        indent_ratio: Union[float, Tuple[float, float]] = (0.0, 0.15),
        title_prob: float = 0.3,
        title_size_multiplier: Tuple[float, float] = (1.3, 1.8),
        list_prob: float = 0.2,
        list_indent_ratio: float = 0.05,
        bbox_max: int = 1023,
    ):
        self.output_size = output_size
        self.margin_ratio = margin_ratio
        self.line_spacing = line_spacing
        self._font_scale_power = max(0.0, float(font_scale_power))
        self._canvas_area_weight_power = max(0.0, float(canvas_area_weight_power))
        self.rgb = rgb
        self._mode = "RGB" if rgb else "L"
        self._bbox_max = bbox_max
        self._col_gap_ratio = 0.03
        # Keep output columns readable after any canvas downscaling.
        self._min_output_column_width = 72

        # Structural options
        self._num_columns_options = [num_columns] if isinstance(num_columns, int) else list(num_columns)
        self._num_columns_options = [max(1, int(c)) for c in self._num_columns_options]
        para_spacing = (paragraph_spacing, paragraph_spacing) if isinstance(paragraph_spacing, (int, float)) else paragraph_spacing
        indent_ratio = (indent_ratio, indent_ratio) if isinstance(indent_ratio, (int, float)) else indent_ratio
        self._para_spacing_range = self._normalize_range(para_spacing, min_value=1.0)
        self._indent_range = self._normalize_range(indent_ratio, min_value=0.0, max_value=0.4)
        self._title_prob = title_prob
        self._title_size_mult_range = title_size_multiplier
        self._list_prob = list_prob
        self._list_indent_ratio = list_indent_ratio

        # Parse font paths
        self._font_paths = self._resolve_font_paths(font_paths)
        if not self._font_paths:
            raise ValueError("No valid fonts found. Please specify font_paths.")

        # Parse font sizes
        if isinstance(font_sizes, int):
            base_sizes = [font_sizes]
        elif isinstance(font_sizes, tuple) and len(font_sizes) == 2:
            base_sizes = list(range(font_sizes[0], font_sizes[1] + 1))
        else:
            base_sizes = list(font_sizes)

        # Add larger sizes for titles
        all_sizes = set(base_sizes)
        for s in base_sizes:
            for mult in [1.3, 1.5, 1.8]:
                all_sizes.add(int(s * mult))
        self._font_sizes = sorted(all_sizes)
        self._body_font_sizes = base_sizes

        # Parse canvas sizes
        if canvas_sizes is None:
            self._canvas_sizes = [output_size]
            self._needs_resize = False
        elif isinstance(canvas_sizes, tuple) and isinstance(canvas_sizes[0], int):
            self._canvas_sizes = [canvas_sizes]
            self._needs_resize = canvas_sizes != output_size
        else:
            self._canvas_sizes = list(canvas_sizes)
            self._needs_resize = True
        if not self._canvas_sizes:
            raise ValueError("canvas_sizes must contain at least one entry.")

        output_area = float(self.output_size[0] * self.output_size[1])
        self._canvas_sample_weights: List[float] = []
        for canvas_w, canvas_h in self._canvas_sizes:
            relative_area = max(1.0, float(canvas_w * canvas_h) / output_area)
            weight = relative_area ** self._canvas_area_weight_power
            self._canvas_sample_weights.append(weight)

        resampling = getattr(Image, "Resampling", Image)
        self._resize_resample = resampling.BILINEAR

        # Parse colors
        self._bg_color_config = self._parse_color_config(background_color)
        self._text_color_config = self._parse_color_config(text_color)

        # Parse horizontal shift
        if isinstance(horizontal_shift, (int, float)):
            self._h_shift_range = (horizontal_shift, horizontal_shift)
        else:
            self._h_shift_range = horizontal_shift

        # Pre-load all font combinations
        self._fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._word_width_caches: Dict[Tuple[str, int], Dict[str, int]] = {}

        for font_path in self._font_paths:
            for size in self._font_sizes:
                key = (font_path, size)
                try:
                    self._fonts[key] = ImageFont.truetype(font_path, size)
                    self._word_width_caches[key] = {}
                except Exception:
                    pass

        # Current render state
        self._current_font_path: str = self._font_paths[0]
        self._current_font_size: int = self._body_font_sizes[0]
        self._current_canvas_size: Tuple[int, int] = self._canvas_sizes[0]
        self._current_bg_color: int = 255
        self._current_text_color: int = 0
        self._current_h_shift: int = 0
        self._current_num_columns: int = 1
        self._current_para_spacing: float = 1.0
        self._current_indent: float = 0.0
        self._current_has_title: bool = False
        self._current_title_size: int = 14
        self._current_is_list: bool = False
        self._current_bullet: str = "•"

    def _resolve_font_paths(self, font_paths: Union[str, List[str], None]) -> List[str]:
        if font_paths is None:
            available = []
            for path in DEFAULT_FONT_PATHS:
                if os.path.exists(path):
                    try:
                        ImageFont.truetype(path, 12)
                        available.append(path)
                    except Exception:
                        pass
            return available
        elif isinstance(font_paths, str):
            return [font_paths] if os.path.exists(font_paths) else []
        else:
            return [p for p in font_paths if os.path.exists(p)]

    def _parse_color_config(self, color) -> dict:
        if isinstance(color, int):
            return {"type": "fixed", "value": color}
        elif isinstance(color, tuple) and len(color) == 2:
            return {"type": "range", "min": color[0], "max": color[1]}
        elif isinstance(color, list):
            return {"type": "list", "values": color}
        return {"type": "fixed", "value": 128}

    def _normalize_range(
        self,
        value: Tuple[float, float],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Tuple[float, float]:
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            lo, hi = hi, lo
        if min_value is not None:
            lo = max(lo, min_value)
            hi = max(hi, min_value)
        if max_value is not None:
            lo = min(lo, max_value)
            hi = min(hi, max_value)
        return lo, hi

    def _sample_color(self, config: dict) -> int:
        if config["type"] == "fixed":
            return config["value"]
        elif config["type"] == "range":
            return random.randint(config["min"], config["max"])
        elif config["type"] == "list":
            return random.choice(config["values"])
        return 128

    def _get_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (font_path, size)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(font_path, size)
            self._word_width_caches[key] = {}
        return self._fonts[key]

    def _get_word_width(self, word: str, font_path: str, size: int) -> int:
        key = (font_path, size)
        cache = self._word_width_caches.get(key, {})
        if word in cache:
            return cache[word]

        font = self._get_font(font_path, size)
        width = int(math.ceil(font.getlength(word)))

        if len(cache) < 10000:
            if key not in self._word_width_caches:
                self._word_width_caches[key] = {}
            self._word_width_caches[key][word] = width

        return width

    def _get_font_height(self, font_path: str, size: int) -> int:
        font = self._get_font(font_path, size)
        ascent, descent = font.getmetrics()
        metrics_height = ascent + descent
        sample_bbox = font.getbbox("Ag")
        bbox_height = sample_bbox[3] - sample_bbox[1] if sample_bbox else 0
        return max(metrics_height, bbox_height)

    def _get_line_height(self, font_path: str, size: int) -> int:
        font_height = self._get_font_height(font_path, size)
        scaled_height = int(math.ceil(font_height * self.line_spacing))
        return max(font_height + 1, scaled_height)

    def _fit_word_to_width(self, word: str, font_path: str, size: int, max_width: int) -> Tuple[str, int]:
        """Fit a word into the available width by truncating when necessary."""
        if max_width <= 0:
            return "", 0

        width = self._get_word_width(word, font_path, size)
        if width <= max_width:
            return word, width

        truncated = word
        while len(truncated) > 1 and self._get_word_width(truncated, font_path, size) > max_width:
            truncated = truncated[:-1]

        if not truncated:
            return "", 0

        # Optionally add a hyphen if it still fits.
        if len(truncated) > 2:
            hyphenated = truncated + "-"
            hyphenated_width = self._get_word_width(hyphenated, font_path, size)
            if hyphenated_width <= max_width:
                return hyphenated, hyphenated_width

        return truncated, self._get_word_width(truncated, font_path, size)

    def _split_paragraphs(self, text: str) -> List[List[str]]:
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)

        result = []
        for para in paragraphs:
            words = re.split(r'\s+', para.strip())
            words = [w for w in words if w]
            if words:
                result.append(words)

        return result

    def _calculate_layout(self, text: str) -> Tuple[List[str], List[List[int]], List[Tuple[str, int]]]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return [], [], []

        canvas_w, canvas_h = self._current_canvas_size
        margin = int(canvas_w * self.margin_ratio)
        num_cols = self._current_num_columns

        col_gap = int(canvas_w * self._col_gap_ratio) if num_cols > 1 else 0
        total_col_width = canvas_w - 2 * margin - (num_cols - 1) * col_gap
        col_width = total_col_width // num_cols
        layout_width = num_cols * col_width + (num_cols - 1) * col_gap

        # Keep columns on-canvas even when random horizontal shift is sampled.
        min_shift = -margin
        max_shift = canvas_w - (margin + layout_width)
        shift = int(max(min(self._current_h_shift, max_shift), min_shift))

        col_x_starts = []
        for c in range(num_cols):
            col_x_starts.append(margin + c * (col_width + col_gap) + shift)

        visible_words = []
        bboxes = []
        font_info = []

        current_col = 0
        x_start = col_x_starts[0]
        x = x_start
        y = margin

        body_font_size = self._current_font_size
        body_line_height = self._get_line_height(self._current_font_path, body_font_size)

        for para_idx, para_words in enumerate(paragraphs):
            if not para_words:
                continue

            is_title = (para_idx == 0 and self._current_has_title)
            is_list = (not is_title and self._current_is_list and para_idx > 0)

            if is_title:
                para_font_size = self._current_title_size
            else:
                para_font_size = body_font_size

            para_font_height = self._get_font_height(self._current_font_path, para_font_size)
            para_line_height = self._get_line_height(self._current_font_path, para_font_size)
            para_space_width = self._get_word_width(" ", self._current_font_path, para_font_size)

            if is_title:
                indent = 0
            elif is_list:
                indent = int(col_width * self._list_indent_ratio)
            else:
                indent = int(col_width * self._current_indent)

            if para_idx > 0:
                paragraph_gap = int(round(body_line_height * (self._current_para_spacing - 1.0)))
                y += max(0, paragraph_gap)

            if y + para_font_height > canvas_h - margin:
                current_col += 1
                if current_col >= num_cols:
                    break
                x_start = col_x_starts[current_col]
                y = margin

            line_start_x = x_start + indent
            if is_list:
                bullet = self._current_bullet
                bullet_width = self._get_word_width(bullet, self._current_font_path, para_font_size)

                if line_start_x >= 0:
                    visible_words.append(bullet)
                    bboxes.append([line_start_x, y, line_start_x + bullet_width, y + para_font_height])
                    font_info.append((self._current_font_path, para_font_size))

                line_start_x += bullet_width + para_space_width

            x = line_start_x
            first_line = True

            for word in para_words:
                word_width = self._get_word_width(word, self._current_font_path, para_font_size)

                col_end = col_x_starts[current_col] + col_width
                if x + word_width > col_end and x > line_start_x:
                    x = x_start + (indent if not first_line or is_list else 0)
                    if is_list and not first_line:
                        x = line_start_x
                    y += para_line_height
                    first_line = False

                    if y + para_font_height > canvas_h - margin:
                        current_col += 1
                        if current_col >= num_cols:
                            break
                        x_start = col_x_starts[current_col]
                        x = x_start + indent
                        y = margin

                col_end = col_x_starts[current_col] + col_width
                available_width = col_end - x
                render_word, render_width = self._fit_word_to_width(
                    word,
                    self._current_font_path,
                    para_font_size,
                    available_width,
                )

                if not render_word:
                    x += para_space_width
                    continue

                if x + render_width < 0:
                    x += render_width + para_space_width
                    continue

                draw_x = max(0, x)
                draw_x1 = min(canvas_w, x + render_width)
                if draw_x1 <= draw_x:
                    x += render_width + para_space_width
                    continue
                visible_words.append(render_word)
                bboxes.append([draw_x, y, draw_x1, y + para_font_height])
                font_info.append((self._current_font_path, para_font_size))

                x += render_width + para_space_width

            if current_col >= num_cols:
                break

            y += para_line_height

        return visible_words, bboxes, font_info

    def _normalize_bboxes(self, bboxes: List[List[int]]) -> List[List[int]]:
        if not bboxes:
            return []

        w, h = self._current_canvas_size
        bbox_max = self._bbox_max
        normalized = []

        for x0, y0, x1, y1 in bboxes:
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))

            normalized.append([
                int(x0 * bbox_max / w),
                int(y0 * bbox_max / h),
                int(x1 * bbox_max / w),
                int(y1 * bbox_max / h),
            ])

        return normalized

    def _render_image(self, words: List[str], bboxes: List[List[int]],
                      font_info: List[Tuple[str, int]]) -> Image.Image:
        bg = self._current_bg_color if self._mode == "L" else (self._current_bg_color,) * 3
        image = Image.new(self._mode, self._current_canvas_size, bg)

        draw = ImageDraw.Draw(image)
        text_color = self._current_text_color if self._mode == "L" else (self._current_text_color,) * 3

        for word, bbox, (font_path, font_size) in zip(words, bboxes, font_info):
            font = self._get_font(font_path, font_size)
            draw.text((bbox[0], bbox[1]), word, font=font, fill=text_color)

        if self._needs_resize and self._current_canvas_size != self.output_size:
            image = image.resize(self.output_size, self._resize_resample)

        return image

    def _sample_parameters(self) -> None:
        self._current_font_path = random.choice(self._font_paths)
        self._current_canvas_size = random.choices(
            self._canvas_sizes,
            weights=self._canvas_sample_weights,
            k=1,
        )[0]

        # Font scaling can be made sublinear so larger pages render denser text after resize.
        scale_x = self._current_canvas_size[0] / self.output_size[0]
        scale_y = self._current_canvas_size[1] / self.output_size[1]
        font_scale = max(scale_x, scale_y) ** self._font_scale_power
        base_font_size = random.choice(self._body_font_sizes)
        self._current_font_size = int(round(base_font_size * font_scale))
        self._current_font_size = max(8, min(96, self._current_font_size))

        self._current_bg_color = self._sample_color(self._bg_color_config)
        self._current_text_color = self._sample_color(self._text_color_config)

        canvas_w = self._current_canvas_size[0]
        shift_ratio = random.uniform(self._h_shift_range[0], self._h_shift_range[1])
        self._current_h_shift = int(canvas_w * shift_ratio)

        # Prevent unreadable dense multi-column layouts at output resolution.
        output_w = self.output_size[0]
        output_margin = int(output_w * self.margin_ratio)
        valid_cols = []
        for cols in sorted(set(self._num_columns_options)):
            out_gap = int(output_w * self._col_gap_ratio) if cols > 1 else 0
            out_total_col_width = output_w - 2 * output_margin - (cols - 1) * out_gap
            out_col_width = out_total_col_width // cols
            if out_col_width >= self._min_output_column_width:
                valid_cols.append(cols)
        if not valid_cols:
            valid_cols = [1]
        weighted_valid_cols = [c for c in self._num_columns_options if c in valid_cols]
        self._current_num_columns = random.choice(weighted_valid_cols)
        self._current_para_spacing = random.uniform(*self._para_spacing_range)
        self._current_indent = random.uniform(*self._indent_range)

        self._current_has_title = random.random() < self._title_prob
        if self._current_has_title:
            mult = random.uniform(*self._title_size_mult_range)
            self._current_title_size = int(self._current_font_size * mult)

        self._current_is_list = random.random() < self._list_prob
        if self._current_is_list:
            self._current_bullet = random.choice(BULLET_CHARS)

    def generate(self, text: str) -> Dict:
        """
        Generate a document image from text.

        Returns:
            Dictionary with:
                - tokens: List[str] - visible words (tokens)
                - bboxes: List[List[int]] - bounding boxes normalized to 0-bbox_max
                - image: PIL.Image - RGB image
        """
        self._sample_parameters()

        visible_words, pixel_bboxes, font_info = self._calculate_layout(text)
        normalized_bboxes = self._normalize_bboxes(pixel_bboxes)
        image = self._render_image(visible_words, pixel_bboxes, font_info)

        return {
            "tokens": visible_words,
            "bboxes": normalized_bboxes,
            "image": image,
        }
