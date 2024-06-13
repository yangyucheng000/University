from modules.graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer
from mindspore import Tensor
import mindspore as ms


x = Tensor([range(768 * 32)], dtype=ms.float32).view(8, 4, 768)
print("x:\t", x.shape, "\n", x)


graphormerGraphEncoderLayer = GraphormerGraphEncoderLayer()
output, attn = graphormerGraphEncoderLayer(x)
print("output:\t", output.shape, "\n", output)
print("attn:\t", "\n", attn)