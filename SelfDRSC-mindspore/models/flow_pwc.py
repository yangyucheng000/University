import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
from models.pwcnet_model import PWCNet
from models.pwc_modules import WarpingLayer as Warp

class Flow_PWC(nn.Cell):
    def __init__(self, load_pretrain=False, pretrain_fn=''):
        super(Flow_PWC, self).__init__()
        self.moduleNetwork = PWCNet()
        print("Creating Flow PWC")

        if load_pretrain:
            param_dict = ms.load_checkpoint(pretrain_fn)
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
            ms.load_param_into_net(self.moduleNetwork, param_dict_new)
            print('Loading Flow PWC pretrain model from {}'.format(pretrain_fn))
            self.moduleNetwork.set_train(False)

    def estimate_flow(self, tensorFirst, tensorSecond):
        b, c, intHeight, intWidth = tensorFirst.shape

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = O.interpolate(input=tensorFirst,
                                                size=(intPreprocessedHeight, intPreprocessedWidth),
                                                mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = O.interpolate(input=tensorSecond,
                                                 size=(intPreprocessedHeight, intPreprocessedWidth),
                                                 mode='bilinear', align_corners=False)

        outputFlow = self.moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)

        tensorFlow = 20.0 * O.interpolate(input=outputFlow, size=(intHeight, intWidth),
                                          mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.shape
        # mesh grid
        xx = O.arange(0, W).view(1, -1).repeat(H, 1)
        yy = O.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = O.cat((xx, yy), 1).float()
        #grid = grid.to(self.device)
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = O.grid_sample(x, vgrid, padding_mode='border')
        mask = O.ones(x.shape)
        mask = O.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output, mask
    
    def construct(self, frame_1, frame_2):
        # flow
        flow = self.estimate_flow(frame_1, frame_2)
        return flow
