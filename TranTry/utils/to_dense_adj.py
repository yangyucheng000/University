from typing import Optional

from mindspore import Tensor
import mindspore.ops as P
# from torch_geometric.typing import OptTensor
# from torch_geometric.utils import scatter

# from torch_scatter import scatter
# from mindspore.ops import tensor_scatter_elements as scatter


def broadcast(src: Tensor, other: Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = [src.size]
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = P.zeros(size[0], dtype=src.dtype)
        return P.scatter_add(out, index, src)
    else:
        return P.scatter_add(out, index, src)


def scatter_add(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
    return scatter_sum(src, index, dim, out, dim_size)

def scatter(src: Tensor, index: Tensor, dim: int = -1,
            out: Optional[Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> Tensor:
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1` in ascending order.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import scatter

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])

        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    else:
        raise ValueError


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        intval = int((edge_index.max() + 1).int())
        batch = edge_index.new_zeros(intval)

    batch_size = batch.max() + 1
    one = batch.new_ones(batch.size)
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    # num_nodes = scatter(one, indices=batch, updates=batch_size, axis=0 reduction='add')
    cum_nodes = P.cat([batch.new_zeros(1), num_nodes.cumsum()])

    if edge_attr is None:
        edge_attr = P.ones(edge_index.shape[1])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    size = [int(batch_size), max_num_nodes, max_num_nodes]
    size += [edge_attr.size][1:]
    adj = P.zeros(size, dtype=edge_attr.dtype)

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view(int([flattened_size][0]))
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2

    adj = scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    # adj = adj.view(size)
    adj = adj.view(size[0], size[1], size[2])

    return adj
