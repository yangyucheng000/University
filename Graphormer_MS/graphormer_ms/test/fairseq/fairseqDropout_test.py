from fairseq.modules.fairseqDropout import FairseqDropout
import mindspore as ms
from mindspore import Tensor

x = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=ms.float32)
dropout = FairseqDropout(0.2, module_name='drop')
dropout.training = True
y = dropout(x)
print(y)