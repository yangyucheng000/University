from mindspore import ops, Tensor
from mindspore import numpy as np
import mindspore as ms


x = Tensor([1, 2, 3], dtype=ms.float32)
y = x.tile((4, 2))
print(y)
