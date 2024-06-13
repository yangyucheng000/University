import mindspore as ms
from mindspore import nn, Tensor
from mindspore import numpy as np
from mindspore.nn.probability.distribution import Uniform


class LayerDropModuleList(nn.CellList):
    """
        A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

        We refresh the choice of which layers to drop every time we iterate
        over the LayerDropModuleList instance. During evaluation we always
        iterate over all layers.

        Usage::

            layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
            for layer in layers:  # this might iterate over layers 1 and 3
                x = layer(x)
            for layer in layers:  # this might iterate over all layers
                x = layer(x)
            for layer in layers:  # this might not iterate over any layers
                x = layer(x)

        Args:
            p (float): probability of dropping out each layer
            modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = Tensor(np.empty(len(self)))
        unif = Uniform(low=0.0, high=1.0)
        dropout_probs = unif.sample(dropout_probs.shape, 0.0, 1.0)
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m








