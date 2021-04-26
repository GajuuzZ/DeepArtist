import os
import time
import torch
import numpy as np
from PIL import Image, ImageTk
from cv2 import resize, cvtColor, COLOR_RGB2YUV, COLOR_YUV2RGB
from cv2 import VideoWriter_fourcc, VideoWriter

import torch.optim as optim
from Models import StyleModel, get_model_layers, hist_loss
from Utils import load_image, tensor2Image, load_image2canvas, overlay_bgra, numpy2Tensor

from Interface import *

SIGNATURE_IMAGE = './signature.png'
TMP_VIDEO = 'tmp.mp4'
VIDEO_SEC = 15


# noinspection PyAttributeOutsideInit
class Main(MainInterface):
    def __init__(self, master):
        MainInterface.__init__(self, master)

        ### Init Default Model. ###
        self.model_name = 'vgg19'
        self.content_layers = ['conv_5']
        self.style_layers = ['conv_1', 'conv_3', 'conv_4', 'conv_5', 'conv_7']

        layers, _ = get_model_layers(self.model_name)
        self.net = StyleModel(layers)
        self.net.set_layers(self.content_layers, self.style_layers)
        self.seted = False

        self.createLayersWeightsPanel()

    def _menuSelectModel(self, name):
        ModelDialog(self, name)
        self.createLayersWeightsPanel()

    def _clickCanvasOpen(self, event):
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

    def _clickViewFeatures(self, event):
        if not self.seted:
            return

        c_name = str(event.widget).split('.')[-1]
        self.fw = FeaturesWindow(self, c_name)

    def _clickSaveOutput(self, event):
        if self.output is None:
            return
        file_name = tkfd.asksaveasfilename(initialdir='.', title='Save output image',
                                           defaultextension='.jpg',
                                           filetypes=(('jpeg files', '*.jpg'), ('all files', '*.*')))
        if len(file_name) > 0:
            image = self.getOutputImage()
            if self.preserv_color.get():
                content_yuv = cvtColor(self.content.numpy().transpose((1, 2, 0)), COLOR_RGB2YUV)
                yuv = cvtColor(np.array(image).astype('float32'), COLOR_RGB2YUV)
                yuv[:, :, 1:3] = content_yuv[:, :, 1:3]
                image = Image.fromarray(np.clip(cvtColor(yuv, COLOR_YUV2RGB), 0, 255).astype(np.uint8))
            image = self.insertSign(image)
            image.save(file_name)

    def _btnUseNoise(self):
        if self.content is None:
            return

        self.output = torch.rand(self.content.unsqueeze(0).shape).cuda()
        self.optimizer = optim.LBFGS([self.output.requires_grad_()], lr=0.1)

        res = self.getOutputImage()
        load_image2canvas(self.canvas_output.canvas, res)
        self.canvas_output.label['text'] = 'output | size: {}x{}'.format(self.output.shape[2], self.output.shape[3])
        self.run = 1

    def _btnSetImages(self):
        if self.content is None or self.style is None:
            return

        content_img = self.content.div(255.).unsqueeze(0).cuda()
        style_img = self.style.div(255.).unsqueeze(0).cuda()
        self.net.set_target(content_img, style_img)
        self.output = content_img.clone()
        self.optimizer = optim.LBFGS([self.output.requires_grad_()], lr=0.1)

        res = self.getOutputImage()
        load_image2canvas(self.canvas_output.canvas, res)
        self.canvas_output.label['text'] = 'output | size: {}x{}'.format(self.output.shape[2], self.output.shape[3])
        self.run = 1
        self.seted = True

    def _btnTransfer(self):
        if self.content is None or self.style is None or not self.seted:
            return

        num_iter = int(self.entry_iter.get())

        if self.preserv_color.get():
            content_yuv = cvtColor(self.content.numpy().transpose((1, 2, 0)), COLOR_RGB2YUV)
        if self.record.get():
            fps = int(num_iter / VIDEO_SEC)
            writer = VideoWriter(TMP_VIDEO, VideoWriter_fourcc(*'MP4V'), fps,
                                 (self.output.shape[3], self.output.shape[2]))
            writer.write(np.array(self.insertSign(self.getOutputImage()))[:, :, ::-1])

        output = self.output
        optimizer = self.optimizer
        net = self.net
        cw = float(self.entry_cw.get())
        sw = float(self.entry_sw.get())
        cwl = [float(self.entry_cwls[ly].get()) for ly in list(self.entry_cwls.keys())]
        swl = [float(self.entry_swls[ly].get()) for ly in list(self.entry_swls.keys())]
        label = self.loss_label

        # RUN START.
        st = time.time()
        itr = self.run + num_iter
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
            if self.record.get():
                writer.write(np.array(self.insertSign(res))[:, :, ::-1])

            load_image2canvas(self.canvas_output.canvas, res)
            self.canvas_output.canvas.update()
            self.run += 1
        elp = time.time() - st

        if self.record.get():
            writer.release()

        # RUN FINISHED.
        self.loss_label['text'] = 'Done. ' + self.loss_label['text']
        self.loss_label.update()
        print('run time: {} m : {} s'.format(elp // 60, elp % 60))

    def insertSign(self, image):
        if self.output is None:
            return
        if not os.path.exists(SIGNATURE_IMAGE):
            print('Not have {} file!!!'.format(SIGNATURE_IMAGE))
            return

        sign = np.array(Image.open(SIGNATURE_IMAGE))
        assert sign.shape[2] == 4, 'sign image must be png transparent file!!!'

        image = np.array(image)

        idx = np.argmin(image.shape[:2])
        sz = round(image.shape[:2][idx] * 0.075)
        ratio = sz / sign.shape[:2][idx]
        sign = resize(sign, (0, 0), fx=ratio, fy=ratio, interpolation=1)

        margin = 0.025
        x, y = image.shape[1] - sign.shape[1], image.shape[0] - sign.shape[0]
        x, y = x - round(image.shape[1] * margin), y - round(image.shape[0] * margin)

        image = overlay_bgra(image, sign, x, y)
        return Image.fromarray(image.astype(np.uint8))

    def getOutputImage(self):
        return tensor2Image(self.output.clone().detach().cpu().data.clamp(0, 1).squeeze(0).mul(255.))

    def createLayersWeightsPanel(self):
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


root = tk.Tk()
app = Main(root)
root.mainloop()

