# -*- coding: utf-8 -*-
# @Author  : ZhengChang
# @Email   : changzheng18@mails.ucas.ac.cn
# @Software: PyCharm
import math
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


class Discriminatorr(nn.Cell):
    def __init__(self, height, width, in_channels, hidden_channels):
        super(Discriminatorr, self).__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.main = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels,
                      kernel_size=3, stride=2, padding=1),
            x2ms_nn.GroupNorm(4, self.hidden_channels),
            x2ms_nn.ReLU()
        )
        self.n = int(x2ms_adapter.tensor_api.log2(math, height))
        # print(self.n)
        for i in range(self.n - 1):
            self.main.add_module(name='conv_{0}'.format(i + 1),
                                 module=x2ms_nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                                                  kernel_size=3, stride=2, padding=1))
            self.main.add_module(name='gn_{0}'.format(i + 1),
                                 module=x2ms_nn.GroupNorm(4, self.hidden_channels))
            self.main.add_module(name='relu_{0}'.format(i + 1),
                                 module=x2ms_nn.ReLU())
        self.linear_in_channels = int(math.ceil(float(width) / (2 ** self.n)) * self.hidden_channels)
        self.linear = x2ms_nn.Sequential(
            x2ms_nn.Linear(self.linear_in_channels  , 1),
            x2ms_nn.Sigmoid()
        )

    def construct(self, input_tensor):
        output_tensor = []
        output_features = []
        for i in range(input_tensor.shape[1]):
            # print(input_tensor.shape)
            # print(self.main)
            features = self.main(input_tensor[:, i, :])
            # print(features.shape)
            features = features.reshape([features.shape[0], -1])
            # print(features.shape)
            output_features.append(features)
            # print(self.linear)
            output = self.linear(features)
            output_tensor.append(output)
        output_tensor = x2ms_adapter.cat(output_tensor, dim=1)
        output_tensor = x2ms_adapter.x2ms_mean(output_tensor, dim=1)
        output_features = x2ms_adapter.stack(output_features, dim=1)
        return output_tensor, output_features


