import tkinter as tk

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App import Main

from Models import StyleModel, get_model_layers


class ModelDialog(tk.Toplevel):
    def __init__(self, parent: 'Main', name):
        tk.Toplevel.__init__(self, parent)
        self.master = parent
        self.title('Select model layers.')

        self.resizable(False, False)
        self.attributes('-topmost', 'true')
        self.focus_force()
        self.grab_set()

        self.pick_name = name
        self.layers, conv_list = get_model_layers(name)

        tk.Label(self, text='Model: {}'.format(name), bg='gray', font=('Times', 14)).grid(
            row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW
        )
        tk.Label(self, text='Content Layers', bg='gray', font=('Times', 12)).grid(
            row=1, column=0, padx=5, pady=2, sticky=tk.NSEW
        )
        tk.Label(self, text='Style Layers', bg='gray', font=('Times', 12)).grid(
            row=1, column=1, padx=5, pady=2, sticky=tk.NSEW
        )

        f_ct = tk.Frame(self, bd=2, bg='gray')
        f_ct.grid(row=2, column=0, padx=5, pady=5, sticky=tk.NSEW)
        self.ctly = {}
        for ly in conv_list:
            self.ctly[ly] = tk.BooleanVar(value=0)
            tk.Checkbutton(f_ct, text=ly, variable=self.ctly[ly], bg='gray', font=(None, 11)).pack(
                padx=10, pady=2, side=tk.TOP, anchor=tk.W
            )

        f_st = tk.Frame(self, bd=2, bg='gray')
        f_st.grid(row=2, column=1, padx=5, pady=5, sticky=tk.NSEW)
        self.stly = {}
        for ly in conv_list:
            self.stly[ly] = tk.BooleanVar(value=0)
            tk.Checkbutton(f_st, text=ly, variable=self.stly[ly], bg='gray', font=(None, 11)).pack(
                padx=10, pady=2, side=tk.TOP, anchor=tk.W
            )

        tk.Button(self, text='Ok', command=self.ok, font=(None, 12)).grid(
            row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW
        )
        self.wait_window()

    def ok(self):
        self.master.model_name = self.pick_name
        self.master.net = StyleModel(self.layers)
        content_layers = [k for k in self.ctly if self.ctly[k].get() == True]
        style_layers = [k for k in self.stly if self.stly[k].get() == True]
        self.master.content_layers = content_layers
        self.master.style_layers = style_layers
        self.master.net.set_layers(content_layers, style_layers)
        self.master.seted = False
        self.destroy()