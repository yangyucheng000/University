import mindspore as ms
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore import nn, ops, Tensor
from mindspore_gl import BatchedGraph
from mindspore_gl.nn import SumPooling, AvgPooling, MaxPooling

from dgl_ms.ops import edge_softmax
from utils.jumping_knowledge import JumpingKnowledge


class Net(GNNCell):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_basis,
                 config):
        super().__init__()

        self.layers = config.layers
        self.lin0 = nn.Dense(input_dim, config.hidden)

        if config.nonlinear == 'ReLU':
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.GELU()

        if config.get('bath_norm', 'Y') == 'Y':
            batch_norm = True
            print('With batch_norm')
        else:
            batch_norm = False
            print('Without batch_norm')
        if config.get('edge_softmax', 'Y') == 'Y':
            self.edge_softmax = True
            print('With edge_softmax.')
        else:
            self.edge_softmax = False
            print('Without edge_softmax.')

        self.convs = nn.CellList()
        for i in range(config.layers):
            self.convs.append(Conv(hidden_size=config.hidden,
                                   dropout_rate=config.dropout,
                                   nonlinear=config.nonlinear,
                                   batch_norm=batch_norm))

        self.emb_jk = JumpingKnowledge('L')
        self.lin1 = nn.Dense(config.hidden, config.hidden)
        self.final_drop = nn.Dropout(p=config.dropout)
        self.lin2 = nn.Dense(config.hidden, output_dim)

        if config.pooling == 'S':
            self.pool = SumPooling()
        elif config.pooling == 'M':
            self.pool = AvgPooling()
        elif config.pooling == 'X':
            self.pool = MaxPooling()

        if batch_norm:
            self.filter_encoder = nn.SequentialCell(
                nn.Dense(num_basis, config.hidden),
                nn.GELU(),
                nn.Dense(config.hidden, config.hidden),
                nn.GELU()
            )
        self.filter_drop = nn.Dropout(p=config.dropout)

    def construct(self, h, bases, g: BatchedGraph):
        x = self.lin0(h)
        bases = self.filter_drop(self.filter_encoder(bases))
        # if self.edge_softmax:
        #     bases = edge_softmax(g, bases)
        xs = []
        for conv in self.convs:
            x = conv(x, bases, g)
            xs = xs + [x]
        x = self.emb_jk(xs)
        h_graph = self.pool(x, g)
        h_graph = self.nonlinear(self.lin1(h_graph))
        h_graph = self.final_drop(h_graph)
        h_graph = self.lin2(h_graph)
        return h_graph


class Conv(GNNCell):
    def __init__(self, hidden_size, dropout_rate, nonlinear, batch_norm):
        super().__init__()
        self.pre_ffn = nn.SequentialCell(
            nn.Dense(hidden_size, hidden_size),
            nn.GELU()
        )
        self.preffn_dropout = nn.Dropout(dropout_rate)

        if nonlinear == 'ReLU':
            _nonlinear = nn.ReLU()
        else:
            _nonlinear = nn.GELU()
        if batch_norm:
            self.ffn = nn.SequentialCell(
                nn.Dense(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                _nonlinear,
                nn.Dense(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                _nonlinear
            )
        else:
            self.ffn = nn.SequentialCell(
                nn.Dense(hidden_size, hidden_size),
                _nonlinear,
                nn.Dense(hidden_size, hidden_size),
                _nonlinear
            )
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def construct(self, x_feat, bases, graph: BatchedGraph):
        graph.set_vertex_attr({"x": self.pre_ffn(x_feat)})
        graph.set_edge_attr({"v": bases})

        for ndata in graph.dst_vertex:
            ndata.aggr_e = graph.avg([u.x * e.v for u, e in ndata.inedges])
        y = ms.Tensor([v.aggr_e for v in graph.dst_vertex])
        y = self.preffn_dropout(y)
        x = x_feat
        if ops.sum(y) >= 0:
            x = x + y
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        if ops.sum(y) >= 0:
            x = x + y
        return x
