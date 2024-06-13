
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
__author__ = 'chuyao'

from core.layers.LSTMCell import LSTMCell
from core.layers.ST_LSTMCell import SpatioTemporalLSTMCell
from core.layers.CausalLSTMCell import CausalLSTMCell
from core.layers.GradientHighwayUnit import GHU

from core.layers.InterLSTMCell import InterLSTMCell
from core.layers.InterST_LSTMCell import InterSpatioTemporalLSTMCell
from core.layers.InteractCausalLSTMCell import InteractCausalLSTMCell

from core.layers.CST_LSTMCell import CST_LSTMCell
from core.layers.SST_LSTMCell import SST_LSTMCell
from core.layers.DST_LSTMCell import DST_LSTMCell
from core.layers.InterDST_LSTMCell import InterDST_LSTMCell


class ConvLSTM(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(ConvLSTM, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def construct(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = x2ms_adapter.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)


        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4, 2))

        return next_frames

class PredRNN(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def construct(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = x2ms_adapter.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)

        # memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        memory = x2ms_adapter.zeros([batch, self.num_hidden[0], height, width])
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4, 2))

        return next_frames

class SAC_LSTM(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(SAC_LSTM, self).__init__()
        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []
        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size

        self.gradient_highway = GHU(
            self.num_hidden[0],
            self.num_hidden[0],
            height,
            width,
            self.configs.filter_size,
            self.configs.stride,
        )
        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden_in, self.num_hidden[i], height,
                                       width, configs.filter_size,
                                       configs.stride,
                                       configs.layer_norm
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))
        
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = mindspore.ops.zeros(
                (batch, self.num_hidden[i], height, width), mindspore.float32)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width])

        z_t = mindspore.ops.zeros((batch, self.num_hidden[0], height,
                           width), mindspore.float32)

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):
            x_gen = 0
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
                
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0], memory)
            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1],c_t[1], memory)

            for i in range(2, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],
                                                                memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))
        return next_frames    
    
    
class PredRNN_Plus(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN_Plus, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size

        self.gradient_highway = GHU(
            self.num_hidden[0],
            self.num_hidden[0],
            height,
            width,
            self.configs.filter_size,
            self.configs.stride,
        )
        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden_in, self.num_hidden[i], height,
                                       width, configs.filter_size,
                                       configs.stride,
                                       configs.layer_norm
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))
        # print("frames",frames[0,0])
        # print(frames.max())
        # print("make_true", mask_true[0,0])
        # print(mask_true.max())
        
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = mindspore.ops.zeros(
                (batch, self.num_hidden[i], height, width), mindspore.float32)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width])


        z_t = mindspore.ops.zeros((batch, self.num_hidden[0], height,
                           width), mindspore.float32)

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):
            x_gen = 0
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
                
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0], memory)
            # print("h_t[0]",h_t[0][0,0])
            # print("h_t[0]",h_t[0][0,1])
            # exit(0)
            # print("h_t[0]", h_t[0][0,0])
            # print("c_t[0]", c_t[0][0,0])
            # print("memory", memory[0,0])
            # exit(0)
            z_t = self.gradient_highway(h_t[0], z_t)

            # print("z_t", z_t[0][0])
            # exit()

            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1],c_t[1], memory)
            # print("h_t[1]", h_t[1][0,0])
            # print("h_t[1]", h_t[1][0,1])
            # print("c_t[1]", c_t[1][0,0]) 
            # print("memory", memory[0,0])
            # exit()

            "这里的h_t[1]有点问题"
            for i in range(2, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],
                                                                memory)
            #     print("h_t[i]", h_t[i][0,0])
            #     print("c_t[i]", c_t[i][0,0])
            # exit()

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # print("x_gen", x_gen[0][0,0])
            next_frames.append(x_gen)
            

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))

        return next_frames



class InteractionConvLSTM(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionConvLSTM, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                InterLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm, configs.r)
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def construct(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = x2ms_adapter.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)


        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4, 2))

        return next_frames

class InteractionPredRNN(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionPredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                InterSpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm,configs.r)
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def construct(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = x2ms_adapter.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)

        # memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        memory = x2ms_adapter.zeros([batch, self.num_hidden[0], height, width])
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4, 2))

        return next_frames

class InteractionPredRNN_Plus(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionPredRNN_Plus, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size

        self.gradient_highway = GHU(
            self.num_hidden[0],
            self.num_hidden[0],
            height,
            width,
            self.configs.filter_size,
            self.configs.stride,
        )
        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                InteractCausalLSTMCell(in_channel, num_hidden_in, self.num_hidden[i], height,
                                       width, configs.filter_size,
                                       configs.stride,
                                       configs.layer_norm,
                                       configs.r
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = x2ms_adapter.zeros(
                [batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
        memory = x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width])


        z_t = x2ms_adapter.zeros([batch, self.num_hidden[0], height,
                           width])

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0], memory)

            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1],c_t[1], memory)

            for i in range(2, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],
                                                                memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))

        return next_frames



class DST_PredRNN(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(DST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                DST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = x2ms_adapter.zeros(
                [batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(x2ms_adapter.tensor_api.unsqueeze(zeros, 1))
        memory = x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width])


        z_t = x2ms_adapter.zeros([batch, self.num_hidden[0], height,
                           width])

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = x2ms_adapter.cat([c_t_history[0],x2ms_adapter.tensor_api.unsqueeze(c_t[0], 1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = x2ms_adapter.cat([c_t_history[i], x2ms_adapter.tensor_api.unsqueeze(c_t[i], 1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))

        return next_frames

class SST_PredRNN(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(SST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                SST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = x2ms_adapter.zeros(
                [batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(x2ms_adapter.tensor_api.unsqueeze(zeros, 1))
        memory = x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width])


        z_t = x2ms_adapter.zeros([batch, self.num_hidden[0], height,
                           width])

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = x2ms_adapter.cat([c_t_history[0],x2ms_adapter.tensor_api.unsqueeze(c_t[0], 1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = x2ms_adapter.cat([c_t_history[i], x2ms_adapter.tensor_api.unsqueeze(c_t[i], 1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))

        return next_frames

class CST_PredRNN(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(CST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                CST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = x2ms_adapter.zeros(
                [batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(x2ms_adapter.tensor_api.unsqueeze(zeros, 1))
        memory = x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width])


        z_t = x2ms_adapter.zeros([batch, self.num_hidden[0], height,
                           width])

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = x2ms_adapter.cat([c_t_history[0],x2ms_adapter.tensor_api.unsqueeze(c_t[0], 1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = x2ms_adapter.cat([c_t_history[i], x2ms_adapter.tensor_api.unsqueeze(c_t[i], 1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))

        return next_frames

class InteractionDST_PredRNN(nn.Cell):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionDST_PredRNN, self).__init__()

        #channel:1   pathch_size = 4
        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        #width:128  patch_size:4  = 32
        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):#num_layers:4
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                InterDST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm,configs.r
                               )
            )
        self.cell_list = x2ms_nn.ModuleList(cell_list)
        self.conv_last = x2ms_nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.drop_path_prob = 0.0




    def construct(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(frames, 0, 1, 4, 2, 3))
        mask_true = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(mask_true, 0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = x2ms_adapter.to(x2ms_adapter.zeros(
                [batch, self.num_hidden[i], height, width]), self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(x2ms_adapter.tensor_api.unsqueeze(zeros, 1))
        memory = x2ms_adapter.to(x2ms_adapter.zeros([batch, self.num_hidden[-1], height, width]), self.configs.device)


        z_t = x2ms_adapter.zeros([batch, self.num_hidden[0], height,
                           width])

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = x2ms_adapter.cat([c_t_history[0],x2ms_adapter.tensor_api.unsqueeze(c_t[0], 1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = x2ms_adapter.cat([c_t_history[i], x2ms_adapter.tensor_api.unsqueeze(c_t[i], 1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x2ms_adapter.stack(next_frames, dim=0), 1, 0, 3, 4,
                                                              2))

        return next_frames











