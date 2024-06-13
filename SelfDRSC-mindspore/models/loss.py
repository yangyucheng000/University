import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
import numpy as np
import mindcv
"""
Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2*): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7*): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16*): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Cell):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def construct(self, x, y):
        diff = x - y
        loss = O.mean(O.sqrt((diff * diff) + self.eps))
        return loss

class Charbonnier:
    # def __init__(self, args):
    #     self.epsilon = 1e-6
    #
    # def __call__(self, pred, gt):
    #     return (((pred - gt) ** 2 + self.epsilon) ** 0.5).mean()

    def __init__(self, args):
        # super(CharbonnierLoss, self).__init__()
        self.eps = 1e-9

    def __call__(self, x, y):
        diff = x - y
        loss = O.mean(O.sqrt((diff * diff) + self.eps))
        return loss


def Perceptual(args):
    return PerceptualLoss(loss=nn.L1Loss())


class PerceptualLoss:
    def contentFunc(self):
        conv_3_3_layer = 14
        vgg19=mindcv.create_model("vgg19",pretrained=True)
        cnn = vgg19.name_cells()['features']
        model = nn.SequentialCell([])
        #model = model.cuda()
        for i,layer in enumerate(cnn.cells()):
            model.append(layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def __call__(self, fake_img, real_img):
        n, c, h, w = fake_img.shape
        fake_img = fake_img.reshape(n * int(c / 3), 3, h, w)
        real_img = real_img.reshape(n * int(c / 3), 3, h, w)
        f_fake = self.contentFunc(fake_img)
        f_real = self.contentFunc(real_img)
        f_real_no_grad = f_real
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

class GridGradientCentralDiff(nn.Cell):
    def __init__(self, nc, padding=True, diagonal=False):
        super().__init__()
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, pad_mode="pad", padding=0, has_bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, pad_mode="pad", padding=0, has_bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1,pad_mode="pad", padding=0, has_bias=False)

        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d((0, 1, 0, 1))

        fx = O.zeros((nc, nc, 2, 2)).float()
        fy = O.zeros((nc, nc, 2, 2)).float()
        if diagonal:
            fxy = O.zeros(nc, nc, 2, 2).float()

        fx_ = ms.tensor([[1, -1], [0, 0]])
        fy_ = ms.tensor([[1, 0], [-1, 0]])
        if diagonal:
            fxy_ = ms.tensor([[1, 0], [0, -1]])

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_

        self.conv_x.weight = ms.Parameter(fx)
        self.conv_y.weight = ms.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = ms.Parameter(fxy)

    def construct(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy

    
class VariationLoss(nn.Cell):
    def __init__(self, nc, grad_fn=GridGradientCentralDiff):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)

    def construct(self, image, weight=None, mean=False):
        if isinstance(image, list):
            total = None
            for i in range(len(image)):
                dx, dy = self.grad_fn(image[i])
                variation = dx ** 2 + dy ** 2
                if i == 0:
                    total = variation
                else:
                    total = total + variation
            variation = total / len(image)
            if weight is not None:
                variation = variation * weight.float()
                if mean != False:
                    return variation.sum() / weight.sum()
            if mean != False:
                return variation.mean()
            return variation.sum()


        dx, dy = self.grad_fn(image)
        variation = dx ** 2 + dy ** 2

        if weight is not None:
            variation = variation * weight.float()
            if mean != False:
                return variation.sum() / weight.sum()
        if mean != False:
            return variation.mean()
        return variation.sum()


# Variance loss
def Variation(args):
    return VariationLoss(nc=2)


def loss_parse(loss_str):
    """
    parse loss parameters
    """
    ratios = []
    losses = []
    str_temp = loss_str.split('|')
    for item in str_temp:
        substr_temp = item.split('*')
        ratios.append(float(substr_temp[0]))
        losses.append(substr_temp[1])
    return ratios, losses
