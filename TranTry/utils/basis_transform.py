import mindspore as ms
import mindspore.ops as P
from mindspore_gl import Graph, GraphField
from mindspore_gl import graph

# from mindspore_gl import Graph, GNNCell
# import dglFunction
# from torch_geometric.utils import to_dense_adj

from .to_dense_adj import to_dense_adj
from mindspore.scipy import linalg


def to_dense(edge_idx):
    num_nodes = edge_idx.max().item() + 1
    dense = ms.ops.zeros([num_nodes, num_nodes])
    for i in range(edge_idx.shape[1]):
        e1 = edge_idx[0][i]
        e2 = edge_idx[1][i]
        dense[e1][e2] += 1
    return dense


def check_edge_max_equal_num_nodes(g):
    edge_idx = ms.ops.stack([g.edges()[0], g.edges()[1]])
    assert (g.num_nodes() == edge_idx.max().item() + 1)


def check_dense(g, dense_used):
    edge_idx = ms.ops.stack([g.edges()[0], g.edges()[1]])
    dense = to_dense(edge_idx)
    assert (dense_used.equal(dense))
    assert (dense_used.max().item() <= 1)
    assert (len(dense_used.shape) == 2)
    assert (dense_used.equal(dense_used.transpose(0, 1)))


def check_repeated_edges(g):
    edge_idx = ms.ops.stack([g.edges()[0], g.edges()[1]])
    edge_unique = ms.ops.unique(edge_idx)
    assert (edge_unique.shape[1] == edge_idx.shape[1])


def basis_transform(g: GraphField,
                    basis,
                    power,
                    epsilon,
                    identity=1):
    # check_edge_max_equal_num_nodes(g)  # with self_loop added, this should be held
    # check_repeated_edges(g)
    row = g.src_idx
    col = g.dst_idx

    (e_idx0, e_idx1) = (row, col)
    edge_idx = ms.ops.stack([e_idx0, e_idx1])
    adj = to_dense_adj(edge_idx).squeeze(0)  # Graphs may have only one node.
    # check_dense(g, adj)

    # adding self loop
    temp_edge_weight = ms.ops.ones_like(edge_idx[0])
    temp_fill_val = ms.ops.ones(g.n_nodes, dtype=ms.int32)
    adj = graph.add_self_loop(edge_idx, temp_edge_weight, g.n_nodes, temp_fill_val, mode='dense')

    if basis == 'eps' and identity == 0:
        adj = adj - ms.ops.eye(adj.shape[0], dtype=adj.dtype) * (1 - identity)
        deg_nopad = adj.sum(1)
        isolated = ms.ops.where(deg_nopad <= 1e-6, ms.ops.ones_like(deg_nopad), ms.ops.zeros_like(deg_nopad))
        if isolated.sum(0) != 0:
            adj = adj + isolated.diag_embed()

    if isinstance(epsilon, str):
        eps = epsilon.split('d')
        epsilon = float(eps[0]) / float(eps[1])

    if basis == 'rho':
        eig_val, eig_vec = linalg.eigh(adj)
        padding = ms.ops.ones_like(eig_val)
        eig_sign = ms.ops.where(eig_val >= 0, padding, padding * -1)
        eig_val_nosign = eig_val.abs()
        eig_val_nosign = ms.ops.where(eig_val_nosign > 1e-6, eig_val_nosign, ms.ops.zeros_like(eig_val_nosign))  # Precision limitation
        eig_val_smoothed = eig_val_nosign.pow(epsilon) * eig_sign
        graph_matrix = ms.ops.matmul(eig_vec, ms.ops.matmul(P.diag_embed(eig_val_smoothed), eig_vec.transpose(-1, -2)))
    elif basis == 'eps':
        deg = adj.sum(1)
        sym_basis = deg.pow(epsilon).unsqueeze(-1)
        graph_matrix = ms.ops.matmul(sym_basis, sym_basis.transpose(0, 1)) * adj
        # # modify the spectrum with a shift operation
        # graph_matrix = torch.matmul(sym_basis, sym_basis.transpose(0, 1)) * adj + torch.eye(adj.shape[0], dtype=adj.dtype) * shift_ratio
    else:
        raise ValueError('Unknown basis called {}'.format(basis))

    identity = ms.ops.eye(graph_matrix.shape[0], dtype=graph_matrix.dtype)
    bases = [identity.flatten()]

    graph_matrix_n = identity
    for shift in range(power):
        graph_matrix_n = ms.ops.matmul(graph_matrix_n, graph_matrix)
        bases = bases + [graph_matrix_n.flatten()]

    # warning! "transpose" in ms and torch, their arguments order is the opposite!
    # e.g. ms.transpose(-1, 2) is equal as torch.transpose(-2, 1)
    bases = ms.ops.stack(bases).transpose(-1, -2)

    # full_one = torch.ones_like(graph_matrix, dtype=graph_matrix.dtype).nonzero(as_tuple=True)
    full_one = ms.ops.ones_like(graph_matrix, dtype=graph_matrix.dtype).nonzero()
    new_g_edges_list = ms.ops.t(full_one)

    return [new_g_edges_list[0], new_g_edges_list[1]], bases
