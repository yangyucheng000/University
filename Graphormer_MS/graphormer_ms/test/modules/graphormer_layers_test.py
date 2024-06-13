from modules.graphormer_layers import GraphAttnBias, GraphNodeFeature
from mindspore import Tensor
import mindspore as ms


num_atoms = 10
num_edges = 66
num_spatial = 7
num_edge_dis = 8
edge_type = 'edgetypehhh'
multi_hop_max_dist = 9


gab = GraphAttnBias(
    num_heads=32,
    num_atoms=num_atoms,
    num_edges=num_edges,
    num_spatial=num_spatial,
    num_edge_dis=num_edge_dis,
    hidden_dim=768,
    edge_type=edge_type,
    multi_hop_max_dist=multi_hop_max_dist,
    n_layers=12,
)


x = Tensor(list(range(32 * 768)), dtype=ms.float32).view(8, 4, 768)
y = Tensor(list(range(32)), dtype=ms.float32).view(8, 4)
attn = Tensor(list(range(40 * 769)), dtype=ms.float32).view(8, 5, 769)
input = {'x':x, 'in_degree':y, 'out_degree':y, 'attn_bias':attn, 'spatial_pos':x, 'edge_input':x, 'attn_edge_type':x}
output = gab(input)
print(output.shape)