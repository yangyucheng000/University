from modules.multihead_attention import MultiheadAttention
from mindspore import Tensor
import mindspore as ms


embed_num = 3
num_heads = 3


x = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=ms.float32).view(2, 2, 3)
b = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           dtype=ms.float32).view(6, 2, 2)
MA = MultiheadAttention(embed_num, num_heads, self_attention=True, q_noise=0.5, qn_block_size=3)
y = MA(x, x, x, b)
print("x:\t", x)
print("y:\t", y)

