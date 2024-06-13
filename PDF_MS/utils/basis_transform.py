import mindspore as ms
from mindspore import Tensor, ops
from mindspore_gl import Graph, MindHomoGraph, GraphField
from mindspore_gl.nn import GNNCell


GNNCell.disable_display()


class AdjToDense(GNNCell):
    def construct(self, g: Graph):
        return g.adj_to_dense()


class SetNodesFeat(GNNCell):
    def construct(self, tsr, g: Graph):
        g.set_vertex_attr({"feat": tsr})
        return g


def power_computation(power, graph_matrix):
    if isinstance(power, list):
        left = power[0]
        right = power[1]
    else:
        left = 1
        right = power
    if left <= 0 or right <= 0 or left > right:
        raise ValueError('Invalid power {}'.format(power))

    bases = []
    graph_matrix_n = ops.eye(graph_matrix.shape[0], dtype=graph_matrix.dtype)
    for _ in range(left - 1):
        graph_matrix_n = ops.matmul(graph_matrix_n, graph_matrix)
    for _ in range(left, right + 1):
        graph_matrix_n = ops.matmul(graph_matrix_n, graph_matrix)
        bases = bases + [graph_matrix_n]
    return bases


def basis_transform(g: MindHomoGraph,
                    feat,
                    basis,
                    power,
                    epsilon,
                    degs,
                    edgehop=None,
                    basis_norm=False):
    edge_idx = g.adj_coo
    src_index = Tensor(edge_idx[0])
    dst_index = Tensor(edge_idx[1])
    n_nodes = g.node_count.item()
    n_edges = g.edge_count.item()
    graphfield = GraphField(src_index, dst_index, n_nodes, n_edges)

    adj = AdjToDense()(*graphfield.get_graph()).asnumpy().tolist()
    adj = Tensor(adj)

    bases = [ops.eye(adj.shape[0], dtype=adj.dtype)]
    deg = adj.sum(1)
    for i, eps in enumerate(epsilon):
        sym_basis = deg.pow(eps).unsqueeze(-1)
        graph_matrix = ops.matmul(sym_basis, sym_basis.transpose(1, 0)) * adj
        bases = bases + power_computation(power[i], graph_matrix)
    for e in degs:
        bases = bases + [deg.pow(e).diag()]

    if basis == 'DEN':
        for i in range(len(bases)):
            bases[i] = bases[i].flatten(start_dim=0).astype(ms.dtype.float32)
        new_edge_idx = ops.ones_like(adj, dtype=adj.dtype).nonzero().transpose(1, 0)
        new_edge_idx = tuple(new_edge_idx)
    elif basis == 'SPA':
        if edgehop is None:
            edgehop = max([power[i][-1] if isinstance(power[i], list) else power[i] for i in range(len(power))])
        pos_edge_idx = adj
        for i in range(1, edgehop):
            pos_edge_idx = ops.matmul(pos_edge_idx, adj)
        pos_edge_idx = tuple(pos_edge_idx.nonzero().transpose(1, 0))
        for i in range(len(bases)):
            bases[i] = bases[i][pos_edge_idx]
        new_edge_idx = pos_edge_idx
    else:
        raise ValueError('Unknown basis called {}'.format(basis))

    bases = ops.stack(bases, axis=0)
    if basis_norm:
        std = ops.std(bases, 1, keepdims=True)
        mean = ops.mean(bases, 1, keep_dims=True)
        bases = (bases - mean) / (std + 1e-5)
    t_params = list(range(len(bases.shape)))
    t_params[-2], t_params[-1] = t_params[-1], t_params[-2]
    bases = Tensor(bases.transpose(t_params))   # .contiguous()

    new_g = GraphField(new_edge_idx[0], new_edge_idx[1], n_nodes, len(new_edge_idx[0]))
    assert (n_nodes == g.node_count)

    return new_g, feat, bases








