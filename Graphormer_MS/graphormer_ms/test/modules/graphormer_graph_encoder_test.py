from modules.graphormer_graph_encoder import GraphormerGraphEncoder
from mindspore import Tensor
import mindspore as ms


gfe = GraphormerGraphEncoder(
    num_atoms=10,
    num_in_degree=10,
    num_out_degree=5,
    num_edges=66,
    num_spatial=7,
    num_edge_dis=8,
    edge_type='edgetypehhh',
    multi_hop_max_dist=9,
    activation_fn='relu'
)


x = Tensor(list(range(32 * 768)), dtype=ms.float32).view(8, 4, 768)
y = Tensor(list(range(32)), dtype=ms.float32).view(8, 4)
attn = Tensor(list(range(40 * 5)), dtype=ms.float32).view(8, 5, 5)
spos= Tensor(list(range(32 * 4)), dtype=ms.float32).view(8, 4, 4)
input = {'x':x, 'in_degree':y, 'out_degree':y, 'attn_bias':attn, 'spatial_pos':spos, 'edge_input':x, 'attn_edge_type':x}
output = gfe(input)
print(output)


