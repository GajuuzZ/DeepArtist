import os
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


def convert_range(x, new_min=0, new_max=1):
    return ((x - x.min()) * (new_max - new_min)) / (x.max() - x.min()) + new_min


def tensor2Image(image):
    return Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))


def numpy2Tensor(image):
    return torch.tensor(image.transpose((2, 0, 1))).contiguous().float()


def load_image(path, size=None, scale=None, keep_ratio=False):
    img = Image.open(path).convert('RGB')
    if size is not None:
        if keep_ratio:
            size2 = int(size / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    img = torch.tensor(np.array(img).transpose((2, 0, 1))).contiguous().float()

    return img


def load_image2canvas(canvas: tk.Canvas, image):
    d_size = (canvas.winfo_width(), canvas.winfo_height())
    max_size_idx = np.argmax(image.size)
    ratio = float(d_size[max_size_idx]) / max(image.size)
    n_size = tuple([int(np.round(x * ratio)) for x in image.size])

    if n_size > d_size:
        min_size_idx = int(not max_size_idx)
        ratio = float(d_size[min_size_idx]) / min(image.size)
        n_size = tuple([int(np.round(x * ratio)) for x in image.size])

    image = image.resize(n_size, Image.ANTIALIAS)
    canvas.image = ImageTk.PhotoImage(image=image)
    canvas.create_image(d_size[0] // 2, d_size[1] // 2,
                        image=canvas.image, anchor=tk.CENTER)


def overlay_bgra(image, overlay, x, y):
    assert overlay.shape[2] == 4, 'overlay must be bgra!!!'
    if x > image.shape[1] or y > image.shape[0]:
        return image

    h, w = overlay.shape[:2]
    if x + w > image.shape[1]:
        w = image.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > image.shape[0]:
        h = image.shape[0] - y
        overlay = overlay[:h, :]

    mask = overlay[..., 3:] / 255.
    res = np.copy(image)
    res[y:y+h, x:x+w] = (1. - mask) * res[y:y+h, x:x+w] + mask * overlay[..., :3]
    return res


