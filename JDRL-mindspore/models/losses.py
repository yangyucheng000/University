import mindspore
import mindspore.nn as nn
import mindspore.ops as O


class CharbonnierLoss(nn.Cell):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def construct(self, x, y):
        diff = x - y
        loss = O.reduce_mean(O.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class ConvGauss(nn.Cell):
    def __init__(self, kernel):
        super(ConvGauss, self).__init__()
        self.kernel=kernel
        n_channels, _, kw, kh = self.kernel.shape
        self.pad = nn.Pad(paddings=((0,0),(0,0),(kw//2, kh//2),(kw//2, kh//2)), mode='SYMMETRIC')
        self.conv = O.Conv2D(out_channel=n_channels,kernel_size=(kw,kh),group=n_channels)

    def construct(self, x):
        x = self.pad(x)
        x = self.conv(x, self.kernel)
        return x


class LaplacianKernel(nn.Cell):
    def __init__(self,kernel):
        super(LaplacianKernel, self).__init__()
        self.conv_gauss=ConvGauss(kernel)

    def construct(self,current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = O.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

class EdgeLoss(nn.Cell):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = mindspore.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = mindspore.numpy.tile(O.expand_dims(O.matmul(k.T,k),0),(3,1,1,1))
        self.loss = CharbonnierLoss()
        self.laplacian_kernel=LaplacianKernel(self.kernel)

    def construct(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class NetWithLoss(nn.Cell):
    def __init__(self, net, warp_fn, reblur, mask):
        super(NetWithLoss, self).__init__()
        self.net = net
        self.get_backwarp = warp_fn
        self.criterion_char = CharbonnierLoss()
        self.criterion_edge = EdgeLoss()
        self.z_tensor = mindspore.Tensor(0)
        self.o_tensor = mindspore.Tensor(1)
        self.reblur_model = reblur
        self.mask_gene = mask

    def construct(self, input_, target):
        restored = self.net(input_)
        output = O.clip_by_value(restored[0], self.z_tensor, self.o_tensor)

        warp_gt, flow = self.get_backwarp(output, target)
        warp_out, flow_ = self.get_backwarp(target, output)

        reblur_out_0 = self.reblur_model(restored[0], input_)
        reblur_out_1 = self.reblur_model(restored[1], input_)
        reblur_out_2 = self.reblur_model(restored[2], input_)

        mask = self.mask_gene(flow, 0.4)
        mask_ = self.mask_gene(flow_, 0.4)

        # Compute loss at each stage
        loss_char = sum([self.criterion_char(restored[j]*mask, warp_gt*mask) for j in range(len(restored))])
        loss_edge = sum([self.criterion_edge(restored[j]*mask, warp_gt*mask) for j in range(len(restored))])
        loss_reblur = self.criterion_char(reblur_out_0, input_)+self.criterion_char(reblur_out_1, input_)+self.criterion_char(reblur_out_2, input_)
        # loss_reblur = criterion_char(reblur_out_0, input_)
        loss_out = self.criterion_char(warp_out*mask_, target*mask_)
        loss = (loss_char) + loss_out + 0.5*loss_reblur + (0.05*loss_edge)
        return loss