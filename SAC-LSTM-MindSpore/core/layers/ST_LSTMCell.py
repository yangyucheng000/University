import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
__author__ = 'chuyao'

class SpatioTemporalLSTMCell(nn.Cell):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
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



    def construct(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = x2ms_adapter.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = x2ms_adapter.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = x2ms_adapter.split(m_concat, self.num_hidden, dim=1)

        i_t = x2ms_adapter.sigmoid(i_x + i_h)
        f_t = x2ms_adapter.sigmoid(f_x + f_h + self._forget_bias)
        g_t = x2ms_adapter.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = x2ms_adapter.sigmoid(i_x_prime + i_m)
        f_t_prime = x2ms_adapter.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = x2ms_adapter.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = x2ms_adapter.cat((c_new, m_new), 1)
        o_t = x2ms_adapter.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * x2ms_adapter.tanh(self.conv_last(mem))

        return h_new, c_new, m_new









