import textwrap
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LINE_WIDTH = 20


def add_text_to_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0),
                      min_lines: Optional[int] = None, add_below: bool = True):
    import textwrap
    lines = textwrap.wrap(text, width=LINE_WIDTH)
    if min_lines is not None and len(lines) < min_lines:
        if add_below:
            lines += [''] * (min_lines - len(lines))
        else:
            lines = [''] * (min_lines - len(lines)) + lines
    h, w, c = image.shape
    offset = int(h * .12)
    img = np.ones((h + offset * len(lines), w, c), dtype=np.uint8) * 255
    font_size = int(offset * .8)

    try:
        font = ImageFont.truetype("assets/OpenSans-Regular.ttf", font_size)
        textsize = font.getbbox(text)
        y_offset = (offset - textsize[3]) // 2
    except:
        font = ImageFont.load_default()
        y_offset = offset // 2

    if add_below:
        img[:h] = image
    else:
        img[-h:] = image
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        line_size = font.getbbox(line)
        text_x = (w - line_size[2]) // 2
        if add_below:
            draw.text((text_x, h + y_offset + offset * i), line, font=font, fill=text_color)
        else:
            draw.text((text_x, 0 + y_offset + offset * i), line, font=font, fill=text_color)
    return np.array(img)


def create_table_plot(images: List[Image.Image], titles: List[str]=None, captions: List[str]=None) -> Image.Image:
    title_max_lines = np.max([len(textwrap.wrap(text, width=LINE_WIDTH)) for text in titles]) if titles is not None else 0
    caption_max_lines = np.max([len(textwrap.wrap(text, width=LINE_WIDTH)) for text in captions]) if captions is not None else 0
    out_images = []
    for i in range(len(images)):
        im = np.array(images[i])
        if titles is not None:
            im = add_text_to_image(im, titles[i], add_below=False, min_lines=title_max_lines)
        if captions is not None:
            im = add_text_to_image(im, captions[i], add_below=True, min_lines=caption_max_lines)
        out_images.append(im)
    image = Image.fromarray(np.concatenate(out_images, axis=1))
    return image
