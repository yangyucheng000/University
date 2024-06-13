import mindspore as ms
from mindspore import nn

from mindspore_gl.nn import SumPooling, AvgPooling, MaxPooling
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl import BatchedGraph
import MyConv as Conv


# from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
# import dglFunction as fn


class MyNet(GNNCell):
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

        self.convs = ms.nn.CellList()
        for i in range(config.layers):
            self.convs.append(cell=Conv.MyConv(hidden_size=config.hidden,
                                   dropout_rate=config.dropout,
                                   nonlinear=config.nonlinear))

        self.lin1 = nn.Dense(config.hidden, config.hidden)
        self.final_drop = nn.Dropout(keep_prob=1-config.dropout)  # done
        self.lin2 = nn.Dense(config.hidden, output_dim)

        if config.pooling == 'S':
            self.pool = SumPooling()
        elif config.pooling == 'M':
            self.pool = AvgPooling()
        elif config.pooling == 'X':
            self.pool = MaxPooling()

        self.filter_encoder = nn.SequentialCell(
            nn.Dense(num_basis, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.GELU(),
            nn.Dense(config.hidden, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.GELU()
        )
        self.filter_drop = nn.Dropout(keep_prob=1-config.dropout)  # done

    # def forward(self, g, h, bases):
    def construct(self, h, bases, g: BatchedGraph):
        """construct function"""
        h = h.astype("float32")
        bases = bases.astype("float32")
        h = self.lin0(h)
        bases = self.filter_drop(self.filter_encoder(bases))
        # bases = edge_softmax(g, bases)
        bases = g.softmax_edges(bases)
        for conv in self.convs:
            h = conv(h, bases, g)
        nr = self.pool(h, g)
        nr = self.nonlinear(self.lin1(nr))
        nr = self.final_drop(nr)
        nr = self.lin2(nr)
        return nr


'''
backup for translate from torch to mindspore

torch.nn.Module	mindspore.nn.Cell
-

torch.nn.Linear	 mindspore.nn.Dense	 
https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/api_mapping/pytorch_diff/Dense.html

torch.nn.Sequential	mindspore.nn.SequentialCell
-

torch.nn.ModuleList	mindspore.nn.CellList
https://pytorch.org/docs/1.5.0/nn.html#torch.nn.ModuleList
https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.CellList.html#mindspore.nn.CellList

'''
