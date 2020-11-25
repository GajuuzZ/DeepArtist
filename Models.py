import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models


def gram_matrix(inp):
    b, ch, h, w = inp.size()
    features = inp.view(b * ch, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * ch * h * w)


def get_model_layers(cnn_name):
    if cnn_name == 'vgg11':
        cnn = models.vgg11(pretrained=True).features.eval()
    elif cnn_name == 'vgg16':
        cnn = models.vgg16(pretrained=True).features.eval()
    elif cnn_name == 'vgg19':
        cnn = models.vgg19(pretrained=True).features.eval()
    else:
        raise ValueError('model name is not available!')

    model = nn.Sequential()
    conv_layers = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            conv_layers.append(name)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

    return model, conv_layers


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, losser=F.mse_loss):
        super(ContentLoss, self).__init__()
        self.losser = losser
        self.weighted = lambda x, w: x * w[None, :, None, None]

    def set_target(self, target):
        self.features = target.detach().cpu()
        self.target = target.detach()
        self.set_weight_channels(torch.ones([self.features.shape[1]]))

    def set_weight_channels(self, ch_wt):
        self.ch_wt = ch_wt.to(self.target.device)

    def forward(self, inp):
        if inp.shape == self.target.shape:
            #self.loss = self.losser(inp, self.weighted(
            #    self.target, self.ch_wt))
            self.loss = self.losser(inp, self.target)
        return inp


class StyleLoss(nn.Module):
    def __init__(self, method='gram', losser=F.mse_loss):
        super(StyleLoss, self).__init__()
        self.losser = losser
        if method == 'gram':
            self.pool = gram_matrix
            self.weighted = lambda x, w: (x * w) * w[:, None]
        elif method == 'globalavgpool':
            self.pool = lambda x: torch.mean(x, dim=[2, 3])
            self.weighted = lambda x, w: x * w

    def set_target(self, target_feature):
        self.features = target_feature.detach().cpu()
        self.target = self.pool(target_feature).detach()
        self.set_weight_channels(torch.ones([self.features.shape[1]]))

    def set_weight_channels(self, ch_wt):
        self.ch_wt = ch_wt.to(self.target.device)

    def forward(self, inp):
        ft = self.pool(inp)
        #self.loss = self.losser(ft, self.weighted(
        #    self.target, self.ch_wt))
        self.loss = self.losser(ft, self.target)
        return inp


class StyleModel(nn.Module):
    def __init__(self, cnn_layers, device='cuda'):
        super(StyleModel, self).__init__()
        self.device = device

        """mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.norm = Normalization(mean, std)"""

        self.layers = cnn_layers
        self.model = None

    def set_layers(self, content_layers, style_layers):
        self.model = nn.Sequential()
        self.content_layers = content_layers
        self.content_losses = []
        self.style_layers = style_layers
        self.style_losses = []

        for name, ly in self.layers.named_children():
            self.model.add_module(name, ly)
            i = name.split('_')[-1]

            if name in content_layers:
                content_loss = ContentLoss()
                self.model.add_module('content_loss_{}'.format(i), content_loss)
                self.content_losses.append(content_loss)
            if name in style_layers:
                style_loss = StyleLoss()
                self.model.add_module('style_loss_{}'.format(i), style_loss)
                self.style_losses.append(style_loss)

        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)].to(self.device)
        #self.model.add_module('sigmoid', nn.Sigmoid())

    def set_target(self, content_img, style_img):
        x = content_img
        y = style_img
        for layer in self.model:
            if isinstance(layer, ContentLoss):
                layer.set_target(x.detach())
            elif isinstance(layer, StyleLoss):
                layer.set_target(y.detach())
            else:
                x = layer(x)
                y = layer(y)

    def forward(self, inp):
        #inp = self.norm(inp)
        self.model(inp)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utils import load_image
    from tqdm import tqdm, trange

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    norm = Normalization(mean, std)

    style_img = load_image('./style_images/shipwreck.jpg', size=512).div(255.).unsqueeze(0).cuda()
    content_img = load_image('./test.jpg', keep_ratio=True, size=512).div(255.).unsqueeze(0).cuda()

    net = StyleModel('vgg19', norm, content_layers_default, style_layers_default)
    net.set_target(content_img, style_img)

    output = content_img.clone()
    optimizer = optim.LBFGS([output.requires_grad_()])

    itr = 300
    run = [0]
    #while run[0] <= itr:

    def closure():
        output.data.clamp_(0, 1)

        optimizer.zero_grad()
        net(output)

        style_loss = 0
        content_loss = 0
        for sl in net.style_losses:
            style_loss += sl.loss
        for cl in net.content_losses:
            content_loss += cl.loss

        content_loss *= 1.0
        style_loss *= 1000000.0

        loss = content_loss + style_loss
        loss.backward()

        run[0] += 1
        print("run {}:".format(run))
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            style_loss.item(), content_loss.item()))
        print()

        return loss

    optimizer.step(closure)

    """output.data.clamp_(0, 1)

    unloader = transforms.ToPILImage()
    output = output.detach().cpu().squeeze(0)
    output = unloader(output)

    plt.figure()
    plt.imshow(output)"""


