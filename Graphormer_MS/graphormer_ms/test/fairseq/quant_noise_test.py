from fairseq.modules.quant_noise import quant_noise
from mindspore import nn, Tensor
import mindspore as ms


k_proj = quant_noise(
            nn.Dense(10, 16, has_bias=True), 0.5, 5
        )
k_proj.training = True
x = Tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=ms.float32)
y = k_proj(x)
print(y)