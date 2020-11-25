import tkinter as tk


class ImageCanvas(tk.Frame):
    def __init__(self, parent, c_width, c_height, c_name, label='', *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.panel_top = tk.Frame(self, bg='lightgray')
        #self.panel_top.pack(side=tk.TOP, anchor=tk.NW, fill=tk.X, expand=True)
        self.panel_top.grid(row=0, column=0, sticky=('W', 'E'))

        self.label = tk.Label(self.panel_top, text=label, bg='lightgray', font=(None, 9))
        self.label.pack(padx=2, side=tk.LEFT, anchor=tk.NW)

        self.canvas = tk.Canvas(self, width=c_width, height=c_height,
                                bg='black', name=c_name)
        #self.canvas.pack(side=tk.TOP, anchor=tk.NW, fill=tk.BOTH, expand=True)
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)
        self.canvas.image = None

    def bind(self, sequence=None, func=None, add=None):
        self.canvas.bind(sequence, func, add)
