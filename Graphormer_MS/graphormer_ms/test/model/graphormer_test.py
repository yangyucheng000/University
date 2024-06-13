from mindspore import Tensor
import mindspore as ms

from model.graphormer import GraphormerModel, GraphormerEncoder

import argparse


x = Tensor(list(range(32 * 768)), dtype=ms.float32).view(8, 4, 768)
y = Tensor(list(range(32)), dtype=ms.float32).view(8, 4)
attn = Tensor(list(range(40 * 5)), dtype=ms.float32).view(8, 5, 5)
spos= Tensor(list(range(32 * 4)), dtype=ms.float32).view(8, 4, 4)
input = {'x':x, 'in_degree':y, 'out_degree':y, 'attn_bias':attn, 'spatial_pos':spos, 'edge_input':x, 'attn_edge_type':x}


parser = argparse.ArgumentParser(description="hello")
args = parser.parse_args()
args.max_nodes = 50
args.num_atoms = 50
args.num_in_degree = 50
args.num_out_degree = 50
args.num_edges = 50
args.num_spatial = 50
args.num_edge_dis = 50
args.edge_type = 5
args.multi_hop_max_dist = 50

args.encoder_layers = 12
args.encoder_embed_dim = 768
args.encoder_ffn_embed_dim = 768
args.encoder_attention_heads = 32
args.dropout = 0.1
args.attention_dropout = 0.1
args.act_dropout = 0.1
args.encoder_normalize_before = False
args.pre_layernorm = False
args.apply_graphormer_init = False
args.activation_fn = "relu"

args.share_encoder_input_output_embed = False
args.num_classes = 2
args.pretrained_model_name = "none"


g_encoder = GraphormerEncoder(args)


gm = GraphormerModel(args, g_encoder)
output = gm(input)
print(output)

