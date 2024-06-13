import numpy as np
import mindspore as ms
from mindspore_gl import BatchedGraphField
from mindspore_gl.graph import BatchHomoGraph, MindHomoGraph
from mindspore_gl.dataloader import Dataset
from .enzymes import Enzymes
from typing import Union


class YouziGraphDataset(Dataset):
    """YouziGraphDataset Dataset"""

    def __init__(self, dataset: Enzymes, batch_size, length):
        self._dataset = dataset
        self._batch_size = batch_size
        self.length = length
        self.batch_fn = BatchHomoGraph()

    def __getitem__(self, batch_graph_idx):
        ds = self._dataset
        graph_list = []
        node_feat_list = []
        edge_feat_list = []

        shape = np.array(batch_graph_idx).shape

        # print("Shape of batch_graph_idx:", shape)

        for idx in range(batch_graph_idx.shape[0]):
            single_g_idx = batch_graph_idx[idx]
            graph_list.append(ds.get_new_single_graph(single_g_idx))
            node_feat_list.append(ds.graph_node_feat(single_g_idx))
            edge_feat_list.append(ds.edge_feat(single_g_idx))
            # print("Single graph idx is", idx, "ok")

        batch_graph = self.batch_fn(graph_list)
        batched_node_feat = np.concatenate(node_feat_list)
        batched_edge_feat = np.concatenate(edge_feat_list)
        batched_label = self._dataset.graph_label[batch_graph_idx]

        np_graph_mask = [1] * self._batch_size
        # np_graph_mask[-1] = 0
        constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
        batchedgraphfiled = self.get_batched_graph_field(batch_graph, constant_graph_mask)
        row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = batchedgraphfiled.get_batched_graph()
        # print("Batched graph: row =", row, ", col =", col, ", node_count =", node_count, ", edge_count =", edge_count,
        #       "batched_node_feat=", batched_node_feat, "batched_edge_feat", batched_edge_feat, "is ok")

        return row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, batched_label, \
            batched_node_feat, batched_edge_feat

        # label = ds.graph_label[idx]
        #
        # node_start_idx = int(ds.graph_nodes[idx])
        # node_end_idx = int(ds.graph_nodes[idx + 1])
        # node_feat = ds.node_feat[node_start_idx: node_end_idx]
        #
        # edge_feat = ds.edge_feat_list[idx]
        #
        # row = ds.graph_edges(idx=idx)[0]
        # col = ds.graph_edges(idx=idx)[1]
        #
        # node_count = node_end_idx - node_start_idx
        # edge_count = edge_feat.shape[0]
        #
        # return label, node_feat, edge_feat, row, col, node_count, edge_count
        # , node_map_idx, edge_map_idx, graph_mask

    def get_batched_graph_field(self, batch_graph, constant_graph_mask):
        return BatchedGraphField(
            ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
            ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
            ms.Tensor(batch_graph.node_count, ms.int32),
            ms.Tensor(batch_graph.edge_count, ms.int32),
            ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
            ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
            constant_graph_mask
        )

    def __len__(self):
        return self.length
