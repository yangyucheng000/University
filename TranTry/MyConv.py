import mindspore
from mindspore import nn

from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl import Graph


class MyConv(GNNCell):
    def __init__(self, hidden_size, dropout_rate, nonlinear):
        super().__init__()
        self.pre_ffn = nn.SequentialCell(
            nn.Dense(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )

        self.preffn_dropout = nn.Dropout(keep_prob=1-dropout_rate)  # done

        if nonlinear == 'ReLU':
            _nonlinear = nn.ReLU()
        else:
            _nonlinear = nn.GELU()
        self.ffn = nn.SequentialCell(
            nn.Dense(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            _nonlinear,
            nn.Dense(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            _nonlinear
        )
        self.ffn_dropout = nn.Dropout(keep_prob=1-dropout_rate)  # done

    def construct(self, x_feat, bases, g: Graph):
        x_feat = x_feat.astype("float32")
        bases = bases.astype("float32")
        # return x_feat
        # g.ndata['x'] = self.pre_ffn(x_feat)
        g.set_vertex_attr({'x': self.pre_ffn(x_feat)})

        # graph.edata['v'] = bases
        g.set_edge_attr({'v': bases})

        # graph.update_all(fn.u_mul_e('x', 'v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
        for v in g.dst_vertex:
            v.aggr_e = g.sum([u.x * e.v for u, e in v.inedges])

        # y = graph.ndata['aggr_e']
        y = [v.aggr_e for v in g.dst_vertex]

        y = self.preffn_dropout(y)
        x = x_feat + y
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        x = x + y
        return x

    # def forward(self, graph, x_feat, bases):
    #     with graph.local_scope():
    #         graph.ndata['x'] = self.pre_ffn(x_feat)
    #         graph.edata['v'] = bases
    #         graph.update_all(fn.u_mul_e('x', 'v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
    #         y = graph.ndata['aggr_e']
    #         y = self.preffn_dropout(y)
    #         x = x_feat + y
    #         y = self.ffn(x)
    #         y = self.ffn_dropout(y)
    #         x = x + y
    #         return x

