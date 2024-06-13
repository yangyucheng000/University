import math
import mindspore as ms
import mindspore.nn as nn
from models.pwcnet_model import PWCNet
from models.pwc_modules import WarpingLayer as Warp


class GetWarp(nn.Cell):
    def __init__(self, ckpt_p="./pwc/pwcnet_ms.ckpt", warp_type='bilinear'):
        super(GetWarp, self).__init__()
        self.pwcnet = PWCNet()
        param_dict = ms.load_checkpoint(ckpt_p)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith(('moment1.', 'moment2', 'global_step', 'beta1_power',
                               'beta2_power', 'learning_rate')):
                continue
            elif key.startswith('network.'):
                nkey = key[8:]
            else:
                nkey = key
            nnkey = nkey.replace("0.conv1.","")
            if nnkey == "0.bias" or nnkey == "0.weight":
                nnkey = "flow_estimators.0.conv1."+nnkey
            elif nkey.startswith("0"):
                nnkey = "flow_estimators."+nnkey
            param_dict_new["pwcnet."+nnkey] = values
        ms.load_param_into_net(self.pwcnet, param_dict_new)
        self.pwcnet.set_train(False)
        self.warp = Warp(warp_type=warp_type)
        self.warp.set_train(False)

    def estimate(self, tenFirst, tenSecond, net):
        assert (tenFirst.shape[3] == tenSecond.shape[3])
        assert (tenFirst.shape[2] == tenSecond.shape[2])
        intWidth = tenFirst.shape[3]
        intHeight = tenFirst.shape[2]

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        interp = nn.ResizeBilinear()
        tenPreprocessedFirst = interp(tenFirst, size=(intPreprocessedHeight, intPreprocessedWidth), align_corners=False)
        tenPreprocessedSecond = interp(tenSecond, size=(intPreprocessedHeight, intPreprocessedWidth), align_corners=False)

        flow = net(tenPreprocessedFirst, tenPreprocessedSecond, training=False)
        tenFlow = 20.0 * interp(flow, size=(intHeight, intWidth), align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[:, :, :, :]

    def construct(self, im_b, im_gt, is_esti=False):
        if is_esti:
            flow = self.estimate(im_b, im_gt, self.pwcnet)
        else:
            flow = self.pwcnet(im_b, im_gt, training=False)
        warp_gt = self.warp(im_gt, flow)
        return warp_gt, flow


def get_backwarp(im_b, im_gt, ckpt_p="./pwc/pwcnet_ms.ckpt", is_esti=False):
    net = GetWarp(ckpt_p)
    return net(im_b, im_gt, is_esti)


def GetBackwarp(ckpt_p="./pwc/pwcnet_ms.ckpt"):
    net = GetWarp(ckpt_p)
    return net