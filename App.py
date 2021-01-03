import time
import torch
import screeninfo
import numpy as np
import tkinter as tk
import tkinter.filedialog as tkfd
import tkinter.simpledialog as smpd
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from cv2 import cvtColor, COLOR_RGB2YUV, COLOR_YUV2RGB

import torch.optim as optim
from Models import StyleModel, get_model_layers, hist_loss
from Utils import load_image, tensor2Image, load_image2canvas

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


# noinspection PyAttributeOutsideInit
class Main(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title('Deep Artist.')
        self.main_screen = get_monitor_from_coord(master.winfo_x(), master.winfo_y())

        self.width = int(self.main_screen.width * .75)
        self.height = int(self.main_screen.height * .85)
        self.master.geometry('{}x{}'.format(self.width, self.height))

        ### Init Default Model. ###
        self.model_name = 'vgg19'
        self.content_layers = ['conv_5']
        self.style_layers = ['conv_1', 'conv_3', 'conv_4', 'conv_5', 'conv_7']

        layers, _ = get_model_layers(self.model_name)
        self.net = StyleModel(layers)
        self.net.set_layers(self.content_layers, self.style_layers)
        self.seted = False

        self.preserv_color = tk.BooleanVar(value=False)

        ### Widgets. ###
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        menu_models = tk.Menu(menubar, tearoff=0)
        for m in MODEL_LIST:
            menu_models.add_command(label=m, command=lambda x=m: self.select_model(x),
                                    font=(None, 12))
        menubar.add_cascade(label='Models', menu=menu_models, font=(None, 12))

        # Frame Canvas.
        self.frame_canvas = tk.Frame(self.master, bd=2, bg='gray')
        self.frame_canvas.pack(padx=5, pady=5, anchor=tk.CENTER, fill=tk.BOTH, expand=True)
        self.frame_canvas.columnconfigure(0, weight=1)
        self.frame_canvas.columnconfigure(1, weight=1)
        self.frame_canvas.rowconfigure(0, weight=1)
        self.frame_canvas.rowconfigure(1, weight=1)
        # Canvas content image.
        self.content = None
        self.canvas_content = ImageCanvas(self.frame_canvas, c_width=int(self.width * .3), label='content',
                                          c_height=int(self.height * .40), c_name='content', bg='black')
        self.canvas_content.grid(row=0, column=0, padx=2, pady=2, sticky=tk.NSEW)
        self.canvas_content.bind('<Button-1>', self.click_canvas)
        self.canvas_content.bind('<Button-3>', self.view_features)
        # Canvas style image.
        self.style = None
        self.canvas_style = ImageCanvas(self.frame_canvas, c_width=int(self.width * .3), label='style',
                                        c_height=int(self.height * .40), c_name='style', bg='lightgray')
        self.canvas_style.grid(row=1, column=0, padx=2, pady=2, sticky=tk.NSEW)
        self.canvas_style.bind('<Button-1>', self.click_canvas)
        self.canvas_style.bind('<Button-3>', self.view_features)
        # Canvas output image.
        self.output = None
        self.canvas_output = ImageCanvas(self.frame_canvas, c_width=int(self.width * .65), label='output',
                                         c_height=int(self.height * .82), c_name='output', bg='lightgray')
        self.canvas_output.grid(row=0, column=1, padx=2, pady=2, sticky=tk.NSEW,
                                columnspan=2, rowspan=2)
        self.canvas_output.bind('<Button-1>', self.save_output)
        self.canvas_output.bind('<Button-3>', self.click_canvas)
        tk.Checkbutton(self.canvas_output.panel_top, text='Preserv Color', variable=self.preserv_color,
                       bg='gray', font=(None, 11)).pack(padx=5, side=tk.RIGHT)
        tk.Button(self.canvas_output.panel_top, text='Use Noise', font=(None, 9),
                  command=self.use_noise).pack(padx=5, side=tk.RIGHT)

        # Frame Bottom.
        self.frame_bottom = tk.Frame(self.master, bd=2, bg='gray')
        self.frame_bottom.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        tk.Button(self.frame_bottom, text='Set', command=self.set_images).pack(
            padx=5, pady=5, side=tk.LEFT
        )
        tk.Button(self.frame_bottom, text='Transfer', command=self.transfer).pack(
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
        self.create_layersweights_panel()

    def create_layersweights_panel(self):
        for widget in self.panel_layers.winfo_children():
            widget.destroy()

        tk.Label(self.panel_layers, text='Model: {}'.format(self.model_name), bg='gray',
                 font=(None, 11, 'bold')).pack(padx=1, pady=2, side=tk.TOP, anchor=tk.W)

        panel_content = tk.Frame(self.panel_layers, bg='gray')
        panel_content.pack(side=tk.TOP, anchor=tk.NW, fill=tk.BOTH, expand=True)
        tk.Label(panel_content, text='Content layers weight:', bg='gray',
                 font=('Times', 9)).pack(padx=7, pady=5, side=tk.LEFT)
        self.entry_cwls = {}
        for ly in self.content_layers:
            tk.Label(panel_content, text=ly, bg='gray', font=('Times', 8)).pack(side=tk.LEFT)
            entry_cwl = tk.Entry(panel_content, width=7)
            entry_cwl.pack(padx=2, side=tk.LEFT)
            entry_cwl.insert(tk.END, '1.0')
            self.entry_cwls[ly] = entry_cwl

        panel_style = tk.Frame(self.panel_layers, bg='gray')
        panel_style.pack(side=tk.TOP, anchor=tk.NW, fill=tk.BOTH, expand=True)
        label_swl = tk.Label(panel_style, text='Style layers weight:', bg='gray',
                             font=('Times', 9))
        label_swl.pack(padx=7, pady=5, side=tk.LEFT)
        self.entry_swls = {}
        for ly in self.style_layers:
            tk.Label(panel_style, text=ly, bg='gray', font=('Times', 8)).pack(side=tk.LEFT)
            entry_swl = tk.Entry(panel_style, width=7)
            entry_swl.pack(padx=2, side=tk.LEFT)
            entry_swl.insert(tk.END, '1.0')
            self.entry_swls[ly] = entry_swl

    def select_model(self, name):
        ModelDialog(self, name)
        self.create_layersweights_panel()

    def click_canvas(self, event):
        c_name = str(event.widget).split('.')[-1]
        filename = tkfd.askopenfilename(initialdir='.', title='Select an Image.',
                                        filetypes=(('JPG files', '*.jpg'), ('JPEG files', '*.jpeg'),
                                                   ('PNG files', '*.png'), ("All files", "*.*")))
        if len(filename) > 0:
            img_size = Image.open(filename).size
            scale_img = smpd.askfloat(title='Scaling the image', prompt='Original size: {}'.format(img_size),
                                      initialvalue=1.0)
            image = load_image(filename, scale=scale_img)
            setattr(self, c_name, image)
            if c_name == 'output':
                print(c_name)
                self.output = self.output.div(255.).unsqueeze(0).cuda()
                self.optimizer = optim.LBFGS([self.output.requires_grad_()], lr=0.1)
            load_image2canvas(event.widget, tensor2Image(image))
            event.widget.master.label['text'] = '{} | size: {}x{}'.format(c_name, image.shape[1], image.shape[2])

    def save_output(self, event):
        if self.output is None:
            return
        file_name = tkfd.asksaveasfilename(initialdir='.', title='Save output image',
                                           defaultextension='.jpg',
                                           filetypes=(('jpeg files', '*.jpg'), ('all files', '*.*')))
        if len(file_name) > 0:
            image = tensor2Image(self.output.clone().detach().cpu().data.clamp(0, 1).squeeze(0).mul(255.))
            if self.preserv_color.get():
                content_yuv = cvtColor(self.content.numpy().transpose((1, 2, 0)), COLOR_RGB2YUV)
                yuv = cvtColor(np.array(image).astype('float32'), COLOR_RGB2YUV)
                yuv[:, :, 1:3] = content_yuv[:, :, 1:3]
                image = Image.fromarray(np.clip(cvtColor(yuv, COLOR_YUV2RGB), 0, 255).astype(np.uint8))
            image.save(file_name)

    def use_noise(self):
        if self.content is None:
            return

        self.output = torch.rand(self.content.unsqueeze(0).shape).cuda()
        self.optimizer = optim.LBFGS([self.output.requires_grad_()], lr=0.1)

        res = tensor2Image(self.output.clone().detach().cpu().data.clamp(0, 1).squeeze(0).mul(255.))
        load_image2canvas(self.canvas_output.canvas, res)
        self.canvas_output.label['text'] = 'output | size: {}x{}'.format(self.output.shape[2], self.output.shape[3])
        self.run = 1

    def set_images(self):
        if self.content is None or self.style is None:
            return

        content_img = self.content.div(255.).unsqueeze(0).cuda()
        style_img = self.style.div(255.).unsqueeze(0).cuda()
        self.net.set_target(content_img, style_img)
        self.output = content_img.clone()
        self.optimizer = optim.LBFGS([self.output.requires_grad_()], lr=0.1)

        res = tensor2Image(self.output.clone().detach().cpu().data.clamp(0, 1).squeeze(0).mul(255.))
        load_image2canvas(self.canvas_output.canvas, res)
        self.canvas_output.label['text'] = 'output | size: {}x{}'.format(self.output.shape[2], self.output.shape[3])
        self.run = 1
        self.seted = True

    def transfer(self):
        if self.content is None or self.style is None or not self.seted:
            return
        st = time.time()
        if self.preserv_color.get():
            content_yuv = cvtColor(self.content.numpy().transpose((1, 2, 0)), COLOR_RGB2YUV)

        output = self.output
        optimizer = self.optimizer
        net = self.net
        cw = float(self.entry_cw.get())
        sw = float(self.entry_sw.get())
        cwl = [float(self.entry_cwls[ly].get()) for ly in list(self.entry_cwls.keys())]
        swl = [float(self.entry_swls[ly].get()) for ly in list(self.entry_swls.keys())]

        label = self.loss_label

        itr = self.run + int(self.entry_iter.get())
        while self.run < itr:
            r = self.run

            def closure():
                output.data.clamp_(0, 1)

                optimizer.zero_grad()
                net(output)

                style_loss = 0
                content_loss = 0
                for i, sl in enumerate(net.style_losses):
                    style_loss += sl.loss * swl[i]
                for i, cl in enumerate(net.content_losses):
                    content_loss += cl.loss * cwl[i]

                content_loss *= cw
                style_loss *= sw

                #loss_hist = hist_loss(output.squeeze(0).clamp(0, 1), self.content.div(255.).cuda())

                loss = content_loss + style_loss  # + loss_hist
                loss.backward()

                """res = output.clone().detach().cpu().data.clamp_(0, 1).squeeze(0).mul(255.)
                loadfn(canvas, tensor2Image(res))
                canvas.update()"""

                label['text'] = 'Run: {} | Style Loss : {:4f} Content Loss: {:4f}'.format(
                    r, style_loss.item(), content_loss.item())
                label.update()
                return loss

            loss = optimizer.step(closure)

            res = tensor2Image(output.clone().detach().cpu().data.clamp_(0, 1).squeeze(0).mul(255.))
            if self.preserv_color.get():
                yuv = cvtColor(np.array(res).astype('float32'), COLOR_RGB2YUV)
                yuv[:, :, 1:3] = content_yuv[:, :, 1:3]
                res = Image.fromarray(np.clip(cvtColor(yuv, COLOR_YUV2RGB), 0, 255).astype(np.uint8))
            load_image2canvas(self.canvas_output.canvas, res)
            self.canvas_output.canvas.update()
            self.run += 1

        self.loss_label['text'] = 'Done. ' + self.loss_label['text']
        self.loss_label.update()

        elp = time.time() - st
        print('run time: {} m : {} s'.format(elp // 60, elp % 60))

    def view_features(self, event):
        if not self.seted:
            return

        c_name = str(event.widget).split('.')[-1]
        self.fw = FeaturesWindow(self, c_name)


root = tk.Tk()
app = Main(root)
root.mainloop()

