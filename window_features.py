import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App import Main

import matplotlib.pyplot as plt
from Utils import convert_range, load_image2canvas
from widgets import ImageCanvas


class FeaturesWindow(tk.Toplevel):
    def __init__(self, parent: 'Main', name):
        tk.Toplevel.__init__(self, parent)
        self.master = parent
        self.title('Feature Channels: {}'.format(name))
        self.sz = int(self.master.width * .13)
        self.p_r = 6
        self.cmap = plt.get_cmap('viridis')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        panel_top = tk.Frame(self, bg='lightgray')
        panel_top.grid(row=0, column=0, sticky=('W', 'E'), columnspan=2)
        tk.Label(panel_top, text='{} layer: '.format(name), bg='lightgray',
                 font=(None, 12, 'bold')).pack(padx=5, pady=2, side=tk.LEFT, anchor=tk.W)

        self.name = name
        self.layers = parent.content_layers if name == 'content' else parent.style_layers
        self.combo_layers = ttk.Combobox(panel_top, values=self.layers, width=10)
        self.combo_layers.pack(padx=5, pady=2, side=tk.LEFT, anchor=tk.W)
        self.combo_layers.current(0)
        self.combo_layers.bind('<<ComboboxSelected>>', self.combo_selected)

        panel_cent = tk.Frame(self)
        panel_cent.grid(row=1, column=0, sticky=tk.NSEW)
        self.panel_canvas = tk.Canvas(panel_cent, width=int((self.sz + 4) * self.p_r),
                                      height=int(parent.height * .82), bd=2, bg='gray')
        self.panel_canvas.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Scrollbar(self, orient='vertical',
                                          command=self.panel_canvas.yview)
        #self.scroll_canvas.grid(side=tk.RIGHT, anchor=tk.NE, fill=tk.Y)
        self.scroll_canvas.grid(row=1, column=1, sticky=tk.NSEW)

        panel_bottom = tk.Frame(self, bg='lightgray')
        panel_bottom.grid(row=2, column=0, sticky=tk.NSEW)
        tk.Button(panel_bottom, text='Set Weights', font=(None, 12),
                  command=self.set_weights).pack(padx=5, pady=2, side=tk.LEFT)

        tk.Button(panel_bottom, text='All Zero', font=(None, 12),
                  command=lambda: self.set_values(0)).pack(padx=5, pady=2, side=tk.LEFT)
        tk.Button(panel_bottom, text='All One', font=(None, 12),
                  command=lambda: self.set_values(1)).pack(padx=5, pady=2, side=tk.LEFT)

        self.entry_value = tk.Entry(panel_bottom, width=5)
        self.entry_value.pack(padx=5, pady=2, side=tk.LEFT)
        tk.Button(panel_bottom, text='Set All', font=(None, 12),
                  command=lambda: self.set_values(float(self.entry_value.get()))
                  ).pack(padx=5, pady=2, side=tk.LEFT)

        self.gen_canvases()

    def combo_selected(self, event):
        self.gen_canvases()

    def gen_canvases(self):
        self.panel_canvas.delete('all')

        frame = tk.Frame(self.panel_canvas, bg='gray')
        frame.bind('<Configure>', lambda e: self.panel_canvas.configure(
            scrollregion=self.panel_canvas.bbox('all')))
        self.panel_canvas.create_window((0, 0), window=frame, anchor=tk.NW)
        self.panel_canvas.configure(yscrollcommand=self.scroll_canvas.set)

        ly = self.layers.index(self.combo_layers.get())
        if self.name == 'content':
            images = self.master.net.content_losses[ly].features
        else:
            images = self.master.net.style_losses[ly].features

        r, c = 0, 0
        self.canvases = []
        self.ch_wt_list = []
        for i, img in enumerate(images[0]):
            self.canvases.append(ImageCanvas(frame, c_width=self.sz, c_height=self.sz,
                                             c_name='ch_{}'.format(i), label='Ch-{}'.format(i)))
            self.canvases[-1].grid(row=r, column=c, padx=1, pady=1, sticky=tk.NSEW)
            self.canvases[-1].update_idletasks()

            entry = tk.Entry(self.canvases[-1].panel_top, width=5)
            entry.pack(padx=2, pady=1, side=tk.RIGHT, anchor=tk.NE)
            entry.insert(tk.END, '1')
            self.ch_wt_list.append(entry)

            image = convert_range(img.clone())
            image = (self.cmap(image.numpy()) * 255.).astype(np.uint8)
            image = Image.fromarray(image[:, :, :3])
            load_image2canvas(self.canvases[-1].canvas, image)

            if c < self.p_r - 1:
                c += 1
            else:
                c = 0
                r += 1

    def set_weights(self):
        ch_wt_list = [float(entry.get()) for entry in self.ch_wt_list]
        ch_wt_list = torch.tensor(ch_wt_list, dtype=torch.float32)
        print(ch_wt_list)

        ly = self.layers.index(self.combo_layers.get())
        if self.name == 'content':
            self.master.net.content_losses[ly].set_weight_channels(ch_wt_list)
        else:
            self.master.net.style_losses[ly].set_weight_channels(ch_wt_list)

    def set_values(self, value):
        for entry in self.ch_wt_list:
            entry.delete(0, tk.END)
            entry.insert(tk.END, str(value))
