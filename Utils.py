import os
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


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
    img = torch.tensor(np.array(img)).permute(2, 0, 1).float()
    return img


def convert_range(x, new_min=0, new_max=1):
    return ((x - x.min()) * (new_max - new_min)) / (x.max() - x.min()) + new_min


def tensor2Image(image):
    return Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))
