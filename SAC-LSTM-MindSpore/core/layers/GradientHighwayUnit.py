import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class GHU(nn.Cell):
    def __init__(self, in_channel, num_hidden, height, width, filter_size,
                 stride):
        super(GHU, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(in_channel,
                      num_hidden * 2,
                      kernel_size=filter_size,
                      stride=stride,
                      padding=self.padding),
            x2ms_nn.LayerNorm([num_hidden * 2, height, width]))
        self.conv_z = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(num_hidden,
                      num_hidden * 2,
                      kernel_size=filter_size,
                      stride=stride,
                      padding=self.padding),
            x2ms_nn.LayerNorm([num_hidden * 2, height, width]))

    def construct(self, x, z):

        z_concat = self.conv_z(z)
        x_concat = self.conv_x(x)

        gates = x_concat + z_concat
        p, u = x2ms_adapter.split(gates, self.num_hidden, dim=1)
        p = x2ms_adapter.tanh(p)
        u = x2ms_adapter.sigmoid(u)
        z_new = u * p + (1 - u) * z

        return z_new
