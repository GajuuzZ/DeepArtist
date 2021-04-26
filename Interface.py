import screeninfo
import tkinter as tk
import tkinter.filedialog as tkfd
import tkinter.simpledialog as smpd

from widgets import ImageCanvas
from window_models import ModelDialog
from window_features import FeaturesWindow

MODEL_LIST = ['vgg11', 'vgg16', 'vgg19']


def get_monitor_from_coord(x, y):  # multiple monitor dealing.
    monitors = screeninfo.get_monitors()
    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return m
    return monitors[0]


class MainInterface(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title('Deep Artist.')
        self.main_screen = get_monitor_from_coord(master.winfo_x(), master.winfo_y())

        self.width = int(self.main_screen.width * .75)
        self.height = int(self.main_screen.height * .85)
        self.master.geometry('{}x{}'.format(self.width, self.height))

        #------------------------------------------------WIDGETS-----------------------------------------------------#

        # ---------------------------------------------------------------MENU BAR.
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        menu_models = tk.Menu(menubar, tearoff=0)
        for m in MODEL_LIST:
            menu_models.add_command(label=m, command=lambda x=m: self._menuSelectModel(x),
                                    font=(None, 12))
        menubar.add_cascade(label='Models', menu=menu_models, font=(None, 12))
        # ---------------------------------------------------------------MENU BAR END.

        # ---------------------------------------------------------------FRAME CANVAS.
        self.frame_canvas = tk.Frame(self.master, bd=2, bg='gray')
        self.frame_canvas.pack(padx=5, pady=5, anchor=tk.CENTER, fill=tk.BOTH, expand=True)
        self.frame_canvas.columnconfigure(0, weight=1)
        self.frame_canvas.columnconfigure(1, weight=1)
        self.frame_canvas.rowconfigure(0, weight=1)
        self.frame_canvas.rowconfigure(1, weight=1)

        # ---------------------------------------------------------------CANVAS CONTENT IMAGE.
        self.content = None

        self.canvas_content = ImageCanvas(self.frame_canvas, c_width=int(self.width * .3), label='content',
                                          c_height=int(self.height * .40), c_name='content', bg='black')
        self.canvas_content.grid(row=0, column=0, padx=2, pady=2, sticky=tk.NSEW)
        self.canvas_content.bind('<Button-1>', self._clickCanvasOpen)
        self.canvas_content.bind('<Button-3>', self._clickViewFeatures)
        # ---------------------------------------------------------------CANVAS CONTENT IMAGE END.

        # ---------------------------------------------------------------CANVAS STYLE IMAGE.
        self.style = None

        self.canvas_style = ImageCanvas(self.frame_canvas, c_width=int(self.width * .3), label='style',
                                        c_height=int(self.height * .40), c_name='style', bg='lightgray')
        self.canvas_style.grid(row=1, column=0, padx=2, pady=2, sticky=tk.NSEW)
        self.canvas_style.bind('<Button-1>', self._clickCanvasOpen)
        self.canvas_style.bind('<Button-3>', self._clickViewFeatures)
        # ---------------------------------------------------------------CANVAS STYLE IMAGE END.

        # ---------------------------------------------------------------CANVAS OUTPUT IMAGE.
        self.output = None
        self.preserv_color = tk.BooleanVar(value=False)
        self.record = tk.BooleanVar(value=True)

        self.canvas_output = ImageCanvas(self.frame_canvas, c_width=int(self.width * .65), label='output',
                                         c_height=int(self.height * .82), c_name='output', bg='lightgray')
        self.canvas_output.grid(row=0, column=1, padx=2, pady=2, sticky=tk.NSEW,
                                columnspan=2, rowspan=2)
        self.canvas_output.bind('<Button-1>', self._clickSaveOutput)
        self.canvas_output.bind('<Button-3>', self._clickCanvasOpen)

        tk.Checkbutton(self.canvas_output.panel_top, text='Record', variable=self.record,
                       bg='gray', font=(None, 11)).pack(padx=5, side=tk.RIGHT)
        tk.Checkbutton(self.canvas_output.panel_top, text='Preserv Color', variable=self.preserv_color,
                       bg='gray', font=(None, 11)).pack(padx=5, side=tk.RIGHT)
        tk.Button(self.canvas_output.panel_top, text='Use Noise', font=(None, 9),
                  command=self._btnUseNoise).pack(padx=5, side=tk.RIGHT)
        #tk.Button(self.canvas_output.panel_top, text='Sign', font=(None, 9),
        #          command=self.insert_sign).pack(padx=5, side=tk.RIGHT)
        # ---------------------------------------------------------------CANVAS OUTPUT IMAGE END.
        # ---------------------------------------------------------------FRAME CANVAS END.

        # ---------------------------------------------------------------FRAME BOTTOM.
        self.frame_bottom = tk.Frame(self.master, bd=2, bg='gray')
        self.frame_bottom.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        tk.Button(self.frame_bottom, text='Set', command=self._btnSetImages).pack(
            padx=5, pady=5, side=tk.LEFT
        )
        tk.Button(self.frame_bottom, text='Transfer', command=self._btnTransfer).pack(
            padx=5, pady=5, side=tk.LEFT
        )

        label_iter = tk.Label(self.frame_bottom, text='Iteration: ', bg='gray',
                              font=('Times', 11))
        label_iter.pack(pady=5, side=tk.LEFT)
        self.entry_iter = tk.Entry(self.frame_bottom, width=10)
        self.entry_iter.pack(pady=5, side=tk.LEFT)
        self.entry_iter.insert(tk.END, '20')

        label_cw = tk.Label(self.frame_bottom, text='Content weight: ', bg='gray',
                            font=('Times', 11))
        label_cw.pack(padx=5, pady=5, side=tk.LEFT)
        self.entry_cw = tk.Entry(self.frame_bottom)
        self.entry_cw.pack(pady=5, side=tk.LEFT)
        self.entry_cw.insert(tk.END, '1.0')

        label_sw = tk.Label(self.frame_bottom, text='Style weight: ', bg='gray',
                            font=('Times', 11))
        label_sw.pack(padx=5, pady=5, side=tk.LEFT)
        self.entry_sw = tk.Entry(self.frame_bottom)
        self.entry_sw.pack(pady=5, side=tk.LEFT)
        self.entry_sw.insert(tk.END, '1000000.0')

        self.loss_label = tk.Label(self.frame_bottom, text='Run: 0 | Loss: 0.0', bg='gray',
                                   font=('Times', 10))
        self.loss_label.pack(padx=10, pady=5, side=tk.RIGHT)

        # Frame Layer weights.
        self.panel_layers = tk.Frame(self.master, bd=2, bg='gray')
        self.panel_layers.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # ----------------------------------------------END WIDGETS--------------------------------------------------#

    def _menuSelectModel(self, name):
        pass

    def _clickCanvasOpen(self, event):
        pass

    def _clickViewFeatures(self, event):
        pass

    def _clickSaveOutput(self, event):
        pass

    def _btnUseNoise(self):
        pass

    def _btnSetImages(self):
        pass

    def _btnTransfer(self):
        pass