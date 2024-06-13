import mindspore
import mindspore.nn as nn
import mindspore.ops as O
import numpy as np
from mindspore.common import initializer as init
import functools

class t_split(nn.Cell):
    def __init__(self):
        super(t_split, self).__init__()

    def construct(self, x, session, dim):
        shape = x.shape
        assert sum(session) == shape[dim]
        y = x.swapaxes(0, dim)
        out = []
        for i in session:
            if y.shape[0]>i:
                out.append(y[:i].swapaxes(0,dim))
                y = y[i:]
            else:
                out.append(y.swapaxes(0,dim))
        return out

class ResidualBlock_noBN(nn.Cell):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, pad_mode="pad", padding=1, has_bias=True, weight_init="XavierUniform")
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, pad_mode="pad", padding=1, has_bias=True, weight_init="XavierUniform")
        self.relu = nn.ReLU()

    def construct(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


# KPN基本网路单元

def flipcat(x, dim):
    cat=O.Concat(axis=dim)
    flip=O.ReverseV2([dim])
    y=flip(x).swapaxes(0,dim)
    y=y[1:].swapaxes(0,dim)
    return cat((x,y))
    
class KPN(nn.Cell):
    def __init__(self, nf=64):
        super(KPN,self).__init__()
        
        self.conv_first = nn.Conv2d(6, nf, 3, 1, pad_mode="pad", padding=1, has_bias=True, weight_init="XavierUniform")
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(64, 35, 3, 1, pad_mode="pad", padding=1, has_bias=True, weight_init="XavierUniform")
        self.kernel_pred = KernelConv()
        
    def construct(self, data_with_est, data):
        x = self.conv_first(data_with_est)
        x = self.block3(self.block2(self.block1(x)))
        core = self.out(x)
        
        return self.kernel_pred(data, core)


class KernelConv(nn.Cell):
    """
    the class of computing prediction
    """
    def __init__(self):
        super(KernelConv, self).__init__()
        self.split = t_split()
    
    def _list_core(self, core_list, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        core_out = {}
    
        for i in range(len(core_list)):
            core = core_list[i]
            core = O.absolute(core)
            wide = core.shape[1]
            final_wide = (core.shape[1]-1)*2+1
            kernel = O.zeros((batch_size,wide,wide,color,height, width),mindspore.float32)
            mid = mindspore.Tensor([core.shape[1]-1],dtype=mindspore.float32)
            for i in range(wide):
                for j in range(wide):
                    distance = O.sqrt((i-mid)**2 + (j-mid)**2)
                    low = O.floor(distance)
                    high = O.floor(distance+0.99)
                    if distance > mid:
                        kernel[:,i,j,0,:,:] = 0
                    elif low == high:
                        kernel[:,i,j,0,:,:] = core[:,int(low),:,:]
                    else:
                        y0 = core[:,int(low),:,:]
                        y1 = core[:,int(low)+1,:,:]
                        kernel[:,i,j,0,:,:] = (distance-low)*y1 + (high-distance)*y0
                    
            kernel = flipcat(flipcat(kernel,1), 2)   
            core_ori = kernel.view(batch_size, N, final_wide * final_wide, color, height, width)
            softmax = O.Softmax(axis=2)
            core_out[final_wide] = softmax(core_ori)
        # it is a dict
        return core_out

    def construct(self, frames, core):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        pred_img = [frames]
        batch_size, N, height, width = frames.shape
        color = 1

        section = [2,3,4,5,6,7,8]
        core_list = []
        core_list = self.split(core,section,1)

        core_out = self._list_core(core_list, batch_size, 1, color, height, width)
        kernel_list = [3,5,7,9,11,13,15]


        for index, K in enumerate(kernel_list):
            img_stack = []
            pad = nn.Pad(paddings=((0, 0), (0, 0), (K//2, K//2), (K//2, K//2)))
            frame_pad = pad(frames)
            _, _, nh, nw = frame_pad.shape
            frame_pad = frame_pad.view(batch_size, N, color, nh, nw)
            for i in range(K):
                for j in range(K):
                    img_stack.append(frame_pad[..., i:i + height, j:j + width])
            stack = O.Stack(axis=2)
            img_stack = stack(img_stack)
            pred = O.reduce_sum(O.mul(core_out[K],img_stack), axis=2)
            if batch_size == 1:
                pred = O.expand_dims(O.squeeze(pred),axis=0)
            else:
                pred = O.squeeze(pred)
            pred_img.append(pred)

        return pred_img


class KERNEL_MAP(nn.Cell):
    def __init__(self, nf=64):
        super(KERNEL_MAP,self).__init__()
        
        self.conv_first = nn.Conv2d(6, nf, 3, 1, pad_mode="pad", padding=1, has_bias=True, weight_init="XavierUniform")
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(64, 8, 3, 1, pad_mode="pad", padding=1, has_bias=True, weight_init="XavierUniform")
        
    def construct(self, x):
        x = self.conv_first(x)
        x = self.block3(self.block2(self.block1(x)))
        kernel_map = self.out(x)
        softmax = O.Softmax(axis=1)
        kernel_map = softmax(kernel_map)
        return kernel_map
    
class Reblur_Model(nn.Cell):
    def __init__(self):
        super(Reblur_Model, self).__init__()
        self.kernel_map_gene = KERNEL_MAP()
        self.kpn = KPN()
    
    def construct(self,im_s,im_b):
        cat=O.Concat(axis=1)
        est_input = cat([im_s,im_b])
        pred_img = self.kpn(est_input, im_s)
        kernel_map = self.kernel_map_gene(est_input)
        
        split=O.Split(axis=1,output_num=8)
        map0,map1,map2,map3,map4,map5,map6,map7 = split(kernel_map)
        map_list = [map0,map1,map2,map3,map4,map5,map6,map7]
        output = map0*pred_img[0]
        for n in range(1,8):
            output += (pred_img[n])*(map_list[n])

        '''    
        import cv2
        for j in range(8):
            cv2.imwrite("./train_temp_dpdd_begin/%s_blur.png"%(j),np.uint8(255*pred_img[j][0,...].permute(1,2,0).cpu().detach().numpy()))
            cv2.imwrite("./train_temp_dpdd_begin/%s_map.png"%(j),np.uint8(255*map_list[j][0,0,...].cpu().detach().numpy()))
            
        cv2.imwrite("./train_temp_dpdd_begin/output.png",np.uint8(255*output[0,...].permute(1,2,0).cpu().detach().numpy()))
        cv2.imwrite("./train_temp_dpdd_begin/xgt.png",np.uint8(255*im_s[0,...].permute(1,2,0).cpu().detach().numpy()))
        cv2.imwrite("./train_temp_dpdd_begin/input.png",np.uint8(255*im_b[0,...].permute(1,2,0).cpu().detach().numpy()))
        '''

        return output