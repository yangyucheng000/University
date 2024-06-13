import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as np


class quant_noise(nn.Cell):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Cell
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    def __init__(self, module, p, block_size):
        # if no quantization noise, don't register hook
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.is_conv = False

        if p <= 0:
            self.module = module
            return

        # supported modules
        assert isinstance(module, (nn.Dense, nn.Embedding, nn.Conv2d))

        # test whether module.weight has the right sizes wrt block_size
        self.is_conv = module.weight.ndim == 4

        # 2D matrix
        if not self.is_conv:
            assert (
                module.weight.shape[1] % block_size == 0
            ), "Input features must be a multiple of block sizes"

        # 4D matrix
        else:
            # 1x1 convolutions
            if module.kernel_size == (1, 1):
                assert (
                    module.in_channels % block_size == 0
                ), "Input channels must be a multiple of block sizes"
            # regular convolutions
            else:
                k = module.kernel_size[0] * module.kernel_size[1]
                assert k % block_size == 0, "Kernel size must be a multiple of block size"

        self.module = module

    def construct(self, input):
        # no noise for evaluation
        self.module.training = True
        if self.module.training:
            if not self.is_conv:
                # gather weight and sizes
                weight = self.module.weight
                in_features = weight.shape[1]
                out_features = weight.shape[0]

                # split weight matrix into blocks and randomly drop selected blocks
                mask = np.zeros(
                    in_features // self.block_size * out_features,
                    dtype=ms.float32
                )
                mask = ops.bernoulli(mask, p=self.p)
                mask = mask.repeat_interleave(self.block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = self.module.weight
                in_channels = self.module.in_channels
                out_channels = self.module.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if self.module.kernel_size == (1, 1):
                    mask = np.zeros(
                        int(in_channels // self.block_size * out_channels),
                        dtype=ms.float32
                    )
                    mask.bernoulli_(self.p)
                    mask = mask.repeat_interleave(self.block_size, -1).view(-1, in_channels)
                else:
                    mask = np.zeros(
                        shape=(weight.shape[0], weight.shape[1]),
                        dtype=ms.float32
                    )
                    mask.bernoulli_(self.p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, self.module.kernel_size[0], self.module.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.bool()
            # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - self.p)
            self.module.weight = s * weight.masked_fill(mask, 0)
            return self.module(input)

