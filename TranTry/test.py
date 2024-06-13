import mindspore as ms
from mindspore_gl import Graph, GraphField
from mindspore_gl.nn import GNNCell
from mindspore import nn
from mindspore import Tensor
import numpy as np

net = nn.Dense(3, 23)
# x1 = Tensor(np.array([[180, 234, 154], [244, 48, 247]]).astype("float64"))
#
# # x1 = x1.astype("float32")
#
# output1 = net(x1)
#
# print(output1)

x2 = Tensor(np.array([[180, 234, 154], [244, 48, 247]]).astype("float64"))

x2 = x2.astype("float32")

output2 = net(x2)

print(output2)