
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
__author__ = 'chuyao'

class SST_LSTMCell(nn.Cell):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SST_LSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.layer_norm = x2ms_nn.LayerNorm([num_hidden,width,width])
        self.conv_x = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            x2ms_nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            x2ms_nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            x2ms_nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            x2ms_nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = x2ms_nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)


    def _attn_spatial(self,in_query,in_keys,in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = x2ms_adapter.tensor_api.permute(in_query.reshape([batch,num_channels,-1]), (0,2,1))
        key = x2ms_adapter.tensor_api.permute(in_keys, (0,1,3,4,2)).reshape([batch,-1,num_channels])
        value = x2ms_adapter.tensor_api.permute(in_values, (0, 1, 3, 4, 2)).reshape([batch, -1, num_channels])
        attn = x2ms_adapter.matmul(query,x2ms_adapter.tensor_api.permute(key, 0,2,1))
        attn = x2ms_nn.Softmax(dim=2)(attn)
        attn = x2ms_adapter.matmul(attn,value)
        attn = x2ms_adapter.tensor_api.permute(attn.reshape([batch,width,height,num_channels]), 0,3,1,2)

        return attn


    def construct(self, x_t, h_t, c_t,c_historys, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = x2ms_adapter.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = x2ms_adapter.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = x2ms_adapter.split(m_concat, self.num_hidden, dim=1)

        i_t = x2ms_adapter.sigmoid(i_x + i_h)
        f_t = x2ms_adapter.sigmoid(f_x + f_h + self._forget_bias)
        g_t = x2ms_adapter.tanh(g_x + g_h)

        c_new = c_t + self._attn_spatial(f_t,c_historys,c_historys)
        c_new = self.layer_norm(c_new) + i_t * g_t

        i_t_prime = x2ms_adapter.sigmoid(i_x_prime + i_m)
        f_t_prime = x2ms_adapter.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = x2ms_adapter.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = x2ms_adapter.cat((c_new, m_new), 1)
        o_t = x2ms_adapter.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * x2ms_adapter.tanh(self.conv_last(mem))

        return h_new, c_new, m_new









