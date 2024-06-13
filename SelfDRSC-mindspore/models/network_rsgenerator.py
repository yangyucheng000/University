import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
from .warplayer import multi_warp

from models import flow_pwc

def CFR_flow_t_align(device, flow_01, flow_10, t_value):
    """ modified from https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py"""
    ## Feature warping
    flow_01, norm0 = fwarp(device, flow_01,
                           t_value * flow_01)  ## Actually, F (t) -> (t+1). Translation. Not normalized yet
    flow_10, norm1 = fwarp(device, flow_10, (
            1 - t_value) * flow_10)  ## Actually, F (1-t) -> (-t). Translation. Not normalized yet

    flow_t0 = -(1 - t_value) * (t_value) * flow_01 + (t_value) * (t_value) * flow_10
    flow_t1 = (1 - t_value) * (1 - t_value) * flow_01 - (t_value) * (1 - t_value) * flow_10

    norm = (1 - t_value) * norm0 + t_value * norm1
    mask_ = (norm > 0).type(norm.type())
    flow_t0 = (1 - mask_) * flow_t0 + mask_ * (flow_t0.copy() / (norm.copy() + (1 - mask_)))
    flow_t1 = (1 - mask_) * flow_t1 + mask_ * (flow_t1.copy() / (norm.copy() + (1 - mask_)))

    return flow_t0, flow_t1

def fwarp(device, img, flo):
    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy

    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.shape

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.tile((1, C, 1, 1))
    y = y.tile((1, C, 1, 1))

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = O.floor(x)
    x2 = x1 + 1
    y1 = O.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(device, img, x1, y1, w11)
    img12, o12 = sample_one(device, img, x1, y2, w12)
    img21, o21 = sample_one(device, img, x2, y1, w21)
    img22, o22 = sample_one(device, img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o

def get_gaussian_weights(x, y, x1, x2, y1, y2):
    w11 = O.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = O.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = O.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = O.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(device, img, shiftx, shifty, weight):
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """

    N, C, H, W = img.shape

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = O.arange(0, H).view(-1, 1)[None, None].long().tile((N, C,
                                                                                                          1,
                                                                                                          W)).view(
        -1)
    flat_basey = O.arange(0, W).view(1, -1)[None, None].long().tile((N, C,
                                                                                                          H,
                                                                                                          1)).view(
        -1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = O.arange(0, N).view(N, 1, 1, 1).long().tile((1, C, H, W)).view(
        -1)
    idxc = O.arange(0, C).view(1, C, 1, 1).long().tile((N, 1, H, W)).view(
        -1)
    # ttype = flat_basex.type()
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = O.masked_select(ids, mask).copy()

    # Note here! accmulate fla must be true for proper bp
    img_warp = O.zeros([N * C * H * W, ])
    img_warp.index_put([ids_mask], O.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = O.zeros([N * C * H * W, ])
    one_warp.index_put([ids_mask], O.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential([
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode="pad",
                  padding=padding, dilation=dilation, has_bias=True),
        nn.PReLU(out_planes)]
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential([
        nn.Conv2dTranspose(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=kernel_size, stride=stride, pad_mode="pad", padding=padding, has_bias=True),
        nn.PReLU(out_planes)]
    )

def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential([
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode="pad",
                  padding=padding, dilation=dilation, has_bias=True)]
    )

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class IFBlock(nn.Cell):
    def __init__(self, in_planes, scale=1, c=64, num_flows=3, mode='backward'):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential([
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2 * c, 3, 2, 1)]
        )
        self.convblock = nn.Sequential([
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c)]
        )
        if mode == 'backward':
            self.conv1 = nn.Conv2dTranspose(2 * c, 2 * num_flows * 2, 4, 2, "pad", 1, has_bias=True)  # TODO WARNING: Notable change
        elif mode == 'forward':
            self.conv1 = nn.Conv2dTranspose(2 * c, 2 * num_flows * 3, 4, 2, "pad", 1, has_bias=True)  # TODO WARNING: Notable change
        else:
            raise ValueError

    def construct(self, x):
        if self.scale != 1:
            x = O.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                              align_corners=False,recompute_scale_factor=True)
            # print(x.size())

        x = self.conv0(x)
        x = self.convblock(x)
        flow = self.conv1(x)
        # print(flow.size())
        if self.scale != 1:
            flow = O.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False,recompute_scale_factor=True)
            # print(flow.size())
        return flow

class ConvDS(nn.Cell):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ConvDS, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SmallMaskNet(nn.Cell):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = conv(input, 32, 5, padding=2)
        self.conv2 = conv(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IFEDNet(nn.Cell):
    def __init__(self, c=16, num_flows=1):
        # self.num_flows = num_flows
        super(IFEDNet, self).__init__()
        self.conv0 = ConvDS(3 * 3, c)
        self.down0 = ConvDS(c*1, 2 * c)  # +c
        self.down1 = ConvDS(2 * c, 4 * c)  # +c
        self.down2 = ConvDS(4 * c, 8 * c)  # +2c
        self.down3 = ConvDS(8 * c, 16 * c)  # +4c
        self.up0 = deconv(16 * c, 8 * c)  # +8c
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv1 = deconv(c, 3, 4, 2, 1)
        # self.out = nn.Conv2d(c, 3, 3, 1, 1)
        self.masknet = SmallMaskNet(2 * 3 + 4, 1)
        #self.device = 'cuda'
    # def gen_coding_time(self, x, reverse):
    #     b, c, h, w = x.size()
    #     time_coding1, _, _ = time_map(h, w, h, reverse)
    #     time_coding1, _, _ = time_coding1.permute(2, 0, 1) #, time_coding2.permute(2, 0, 1), warpmask.permute(2, 0, 1)
    #     time_coding1, _, _ = time_coding1.repeat(b, 1, 1, 1) #, time_coding2.repeat(b, 1, 1, 1), warpmask.repeat(b, 1, 1, 1)
    #
    #     return time_coding1 #, time_coding2, warpmask

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.shape
        # mesh grid
        xx = O.arange(0, W).view(1, -1).tile((H, 1))
        yy = O.arange(0, H).view(-1, 1).tile((1, W))
        xx = xx.view(1, 1, H, W).tile((B, 1, 1, 1))
        yy = yy.view(1, 1, H, W).tile((B, 1, 1, 1))
        grid = O.cat((xx, yy), 1).float()
        #grid = grid.to(self.device)
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].copy() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].copy() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = O.grid_sample(x, vgrid, padding_mode='border')
        mask = O.ones(x.shape)
        mask = O.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output

    def construct(self, x0, x2, flow02, flow20, encoding, is_b2t):  #rs_cube channel: c,c,2c,4c,8c   rs_cube,
        time_code = [encoding[:, i] for i in range(encoding.shape[1])]
        time_coding = time_code[0]
        ft0, ft2 = CFR_flow_t_align('cuda', flow02, flow20, time_coding)
        occ = self.masknet(O.cat([x0, x2, ft0, ft2], axis=1))
        occ_0 = O.sigmoid(occ)
        occ_1 = 1 - occ_0
        warped_img = (1 - time_coding) * occ_0 * self.warp(x0, ft0) + time_coding * occ_1 * self.warp(x2, ft2)

        d0 = self.conv0(O.cat((warped_img, x0, x2), axis=1))
        d0 = self.down0(d0)   #torch.cat([d0, rs_cube[0]], axis=1)
        d1 = self.down1(d0)    #torch.cat((d0, rs_cube[1]), axis=1)
        d2 = self.down2(d1)    #torch.cat((d1, rs_cube[2]), axis=1)
        d3 = self.down3(d2)     #torch.cat((d2, rs_cube[3]), axis=1)
        out = self.up0(d3)   #torch.cat((d3, rs_cube[4]), axis=1)
        out = self.up1(O.cat((out, d2), axis=1))
        out = self.up2(O.cat((out, d1), axis=1))
        out = self.up3(O.cat((out, d0), axis=1))
        out = self.conv1(out)
        # out = self.out(out)
        res = O.sigmoid(out) * 2 - 1
        pred = warped_img + res
        pred = O.clamp(pred, 0, 1)

        return pred



class RSG(nn.Cell):
    def __init__(self, num_frames, n_feats, load_flow_net, flow_pretrain_fn):
        super(RSG, self).__init__()
        # self.warped_context_net = WarpedContextNet(c=n_feats, num_flows=num_frames)
        self.ife_net = IFEDNet(c=n_feats, num_flows=num_frames)
        self.pwcflow = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device='cuda')

    def construct(self, x, encoding, is_b2t):
        #x:b,3,c,h,w        encoding: b,2,3,1,h,w

        flow02 = self.pwcflow(x[:, 0], x[:, 2])
        flow20 = self.pwcflow(x[:, 2], x[:, 0])

        out = self.ife_net(x[:, 0], x[:, 2], flow02, flow20, encoding[:, 1], is_b2t)  #rs_cube,

        return out


#
# if __name__ == '__main__':
#     from para import Parameter
#
#     args = Parameter().args
#     inputs = torch.randn(4, 18, 256, 256).cuda()
#     encodings = torch.randn(4, 6, 256, 256).cuda()
#     model = Model(args).cuda()
#     outputs, flows = model(inputs, encodings)
#     print(outputs.shape)
#     for flow in flows:
#         print(flow.shape)

