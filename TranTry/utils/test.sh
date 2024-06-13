echo("/home/youzi/code/t2/venv/bin/python /home/youzi/code/t2/graphlearning/model_zoo/diffpool/trainval_enzymes.py
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, h, g: BatchedGraph):                                                  1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             h,                                                                                   |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             ver_subgraph_idx,                                                                    |
|                                                                                                  ||             edge_subgraph_idx,                                                                   |
|                                                                                                  ||             graph_mask                                                                           |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|                                                                                                  ||  14         scatter_ver_subgraph_idx = RESHAPE(ver_subgraph_idx, (SHAPE(ver_subgraph_idx)[0], 1))|
|                                                                                                  ||  15         scatter_edge_subgraph_idx = RESHAPE(                                                 |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 (SHAPE(edge_subgraph_idx)[0], 1)                                                 |
|                                                                                                  ||             )                                                                                    |
|                                                                                                  ||  16         n_graphs = SHAPE(graph_mask)[0]                                                      |
|        out_all = []                                                                          2   ||  17         out_all = []                                                                         |
|        g_embedding = self.gcn_construct(h, self.gc_before_pool, self.concat, g)              3   ||  18         g_embedding = self.gcn_construct(                                                    |
|                                                                                                  ||                 h,                                                                               |
|                                                                                                  ||                 self.gc_before_pool,                                                             |
|                                                                                                  ||                 self.concat,                                                                     |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
|        readout = g.max_nodes(g_embedding)                                                    4   ||  19         SCATTER_INPUT_SNAPSHOT1 = g_embedding                                                |
|                                                                                                  ||  20         readout = MASKED_FILL(                                                               |
|                                                                                                  ||                 SCATTER_MAX(                                                                     |
|                                                                                                  ||                     FILL(                                                                        |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT1.dtype,                                           |
|                                                                                                  ||                         (n_graphs,) + SHAPE(SCATTER_INPUT_SNAPSHOT1)[1:],                        |
|                                                                                                  ||                         -1e1000                                                                  |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_ver_subgraph_idx,                                                    |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT1                                                      |
|                                                                                                  ||                 ),                                                                               |
|                                                                                                  ||                 IS_INF(                                                                          |
|                                                                                                  ||                     SCATTER_MAX(                                                                 |
|                                                                                                  ||                         FILL(                                                                    |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT1.dtype,                                       |
|                                                                                                  ||                             (n_graphs,) + SHAPE(SCATTER_INPUT_SNAPSHOT1)[1:],                    |
|                                                                                                  ||                             -1e1000                                                              |
|                                                                                                  ||                         ),                                                                       |
|                                                                                                  ||                         scatter_ver_subgraph_idx,                                                |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT1                                                  |
|                                                                                                  ||                     )                                                                            |
|                                                                                                  ||                 ),                                                                               |
|                                                                                                  ||                 0.0                                                                              |
|                                                                                                  ||             )                                                                                    |
|        out_all.append(readout)                                                               5   ||  21         out_all.append(readout)                                                              |
|        if self.num_aggs == 2:                                                                6   ||  22         if self.num_aggs == 2:                                                               |
|            readout = g.sum_nodes(g_embedding)                                                7   ||  23             SCATTER_INPUT_SNAPSHOT2 = g_embedding                                            |
|                                                                                                  ||  24             readout = SCATTER_ADD(                                                           |
|                                                                                                  ||                     ZEROS(                                                                       |
|                                                                                                  ||                         (n_graphs,) + SHAPE(SCATTER_INPUT_SNAPSHOT2)[1:],                        |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT2.dtype                                            |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_ver_subgraph_idx,                                                    |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT2                                                      |
|                                                                                                  ||                 )                                                                                |
|            out_all.append(readout)                                                           8   ||  25             out_all.append(readout)                                                          |
|        adj, h = self.first_diffpool_layer(g_embedding, g)                                    9   ||  26         adj, h = self.first_diffpool_layer(                                                  |
|                                                                                                  ||                 g_embedding,                                                                     |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
|        node_per_pool_graph = ops.Shape()(adj)[0] // ops.Shape()(g.graph_mask)[0]             10  ||  27         node_per_pool_graph = ops.Shape()(adj)[0] // ops.Shape()(graph_mask)[0]              |
|        h, adj = batch2tensor(adj, h, node_per_pool_graph)                                    11  ||  28         h, adj = batch2tensor(adj, h, node_per_pool_graph)                                   |
|        h = self.gcn_construct_tensorized(h, adj, self.gc_after_pool[0], self.concat)         12  ||  29         h = self.gcn_construct_tensorized(h, adj, self.gc_after_pool[0], self.concat)        |
|        readout = ops.ReduceMax()(h, 1)                                                       13  ||  30         readout = ops.ReduceMax()(h, 1)                                                      |
|        out_all.append(readout)                                                               14  ||  31         out_all.append(readout)                                                              |
|        if self.num_aggs == 2:                                                                15  ||  32         if self.num_aggs == 2:                                                               |
|            readout = ops.ReduceSum()(h, 1)                                                   16  ||  33             readout = ops.ReduceSum()(h, 1)                                                  |
|            out_all.append(readout)                                                           17  ||  34             out_all.append(readout)                                                          |
|        if self.n_pooling == 2:                                                               18  ||  35         if self.n_pooling == 2:                                                              |
|            h, adj = self.second_diffpool_layer(h, adj)                                       19  ||  36             h, adj = self.second_diffpool_layer(h, adj)                                      |
|            h = self.gcn_construct_tensorized(h, adj, self.gc_after_pool[1], self.concat)     20  ||  37             h = self.gcn_construct_tensorized(h, adj, self.gc_after_pool[1], self.concat)    |
|            readout = ops.ReduceMax()(h, 1)                                                   21  ||  38             readout = ops.ReduceMax()(h, 1)                                                  |
|            out_all.append(readout)                                                           22  ||  39             out_all.append(readout)                                                          |
|            if self.num_aggs == 2:                                                            23  ||  40             if self.num_aggs == 2:                                                           |
|                readout = ops.ReduceSum()(h, 1)                                               24  ||  41                 readout = ops.ReduceSum()(h, 1)                                              |
|                out_all.append(readout)                                                       25  ||  42                 out_all.append(readout)                                                      |
|        if self.concat:                                                                       26  ||  43         if self.concat:                                                                      |
|            final_readout = ops.Concat(1)(out_all)                                            27  ||  44             final_readout = ops.Concat(1)(out_all)                                           |
|        else:                                                                                 28  ||  45         else:                                                                                |
|            final_readout = readout                                                           29  ||  46             final_readout = readout                                                          |
|        ypred = self.pred_layer(final_readout)                                                30  ||  47         ypred = self.pred_layer(final_readout)                                               |
|        return ypred                                                                          31  ||  48         return ypred                                                                         |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def gcn_construct(self, h, gc_layers, cat, g: BatchedGraph):                              1   ||  1      def gcn_construct(                                                                       |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             h,                                                                                   |
|                                                                                                  ||             gc_layers,                                                                           |
|                                                                                                  ||             cat,                                                                                 |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             ver_subgraph_idx,                                                                    |
|                                                                                                  ||             edge_subgraph_idx,                                                                   |
|                                                                                                  ||             graph_mask                                                                           |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|                                                                                                  ||  14         scatter_ver_subgraph_idx = RESHAPE(ver_subgraph_idx, (SHAPE(ver_subgraph_idx)[0], 1))|
|                                                                                                  ||  15         scatter_edge_subgraph_idx = RESHAPE(                                                 |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 (SHAPE(edge_subgraph_idx)[0], 1)                                                 |
|                                                                                                  ||             )                                                                                    |
|                                                                                                  ||  16         n_graphs = SHAPE(graph_mask)[0]                                                      |
|        block_readout = []                                                                    2   ||  17         block_readout = []                                                                   |
|        for gc_layer in gc_layers[:-1]:                                                       3   ||  18         for gc_layer in gc_layers[:-1]:                                                      |
|            h = gc_layer(h, None, g)                                                          4   ||  19             h = gc_layer(                                                                    |
|                                                                                                  ||                     h,                                                                           |
|                                                                                                  ||                     None,                                                                        |
|                                                                                                  ||                     src_idx,                                                                     |
|                                                                                                  ||                     dst_idx,                                                                     |
|                                                                                                  ||                     n_nodes,                                                                     |
|                                                                                                  ||                     n_edges,                                                                     |
|                                                                                                  ||                     ver_subgraph_idx,                                                            |
|                                                                                                  ||                     edge_subgraph_idx,                                                           |
|                                                                                                  ||                     graph_mask                                                                   |
|                                                                                                  ||                 )                                                                                |
|            block_readout.append(h)                                                           5   ||  20             block_readout.append(h)                                                          |
|        h = gc_layers[-1](h, None, g)                                                         6   ||  21         h = gc_layers[-1](                                                                   |
|                                                                                                  ||                 h,                                                                               |
|                                                                                                  ||                 None,                                                                            |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
|        block_readout.append(h)                                                               7   ||  22         block_readout.append(h)                                                              |
|        if cat:                                                                               8   ||  23         if cat:                                                                              |
|            block = ops.Concat(1)(block_readout)                                              9   ||  24             block = ops.Concat(1)(block_readout)                                             |
|        else:                                                                                 10  ||  25         else:                                                                                |
|            block = h                                                                         11  ||  26             block = h                                                                        |
|        return block                                                                          12  ||  27         return block                                                                         |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def loss(self, pred, label, g: BatchedGraph):                                             1   ||  1      def loss(                                                                                |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             pred,                                                                                |
|                                                                                                  ||             label,                                                                               |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             ver_subgraph_idx,                                                                    |
|                                                                                                  ||             edge_subgraph_idx,                                                                   |
|                                                                                                  ||             graph_mask                                                                           |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|                                                                                                  ||  14         scatter_ver_subgraph_idx = RESHAPE(ver_subgraph_idx, (SHAPE(ver_subgraph_idx)[0], 1))|
|                                                                                                  ||  15         scatter_edge_subgraph_idx = RESHAPE(                                                 |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 (SHAPE(edge_subgraph_idx)[0], 1)                                                 |
|                                                                                                  ||             )                                                                                    |
|                                                                                                  ||  16         n_graphs = SHAPE(graph_mask)[0]                                                      |
|        criterion = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')      2   ||  17         criterion = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')     |
|        loss = criterion(pred, label)                                                         3   ||  18         loss = criterion(pred, label)                                                        |
|        loss = ops.ReduceMean()(loss * g.graph_mask)                                          4   ||  19         loss = ops.ReduceMean()(loss * graph_mask)                                           |
|        if self.n_pooling == 2:                                                               5   ||  20         if self.n_pooling == 2:                                                              |
|            if self.link_pred:                                                                6   ||  21             if self.link_pred:                                                               |
|                loss = loss + self.second_diffpool_layer.link_pred_loss                       7   ||  22                 loss = loss + self.second_diffpool_layer.link_pred_loss                      |
|            if self.entropy:                                                                  8   ||  23             if self.entropy:                                                                 |
|                loss = loss + self.second_diffpool_layer.entropy_loss                         9   ||  24                 loss = loss + self.second_diffpool_layer.entropy_loss                        |
|        if self.link_pred:                                                                    10  ||  25         if self.link_pred:                                                                   |
|            loss = loss + self.first_diffpool_layer.link_pred_loss                            11  ||  26             loss = loss + self.first_diffpool_layer.link_pred_loss                           |
|        return loss                                                                           12  ||  27         return loss                                                                          |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, node_feat, edge_weight, g: Graph):                                    1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             node_feat,                                                                           |
|                                                                                                  ||             edge_weight,                                                                         |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             UNUSED_0=None,                                                                       |
|                                                                                                  ||             UNUSED_1=None,                                                                       |
|                                                                                                  ||             UNUSED_2=None,                                                                       |
|                                                                                                  ||             UNUSED_3=None,                                                                       |
|                                                                                                  ||             UNUSED_4=None                                                                        |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|        g.set_vertex_attr({'h': node_feat})                                                   2   ||  14         h, = [node_feat]                                                                     |
|        if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:                3   ||  15         if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:               |
|            g.set_vertex_attr({'h': self.dense_neigh(node_feat)})                             4   ||  16             h, = [self.dense_neigh(node_feat)]                                               |
|        if self.agg_type == 'pool':                                                           5   ||  17         if self.agg_type == 'pool':                                                          |
|            g.set_vertex_attr({'h': ms.ops.ReLU()(self.fc_pool(node_feat))})                  6   ||  18             h, = [ms.ops.ReLU()(self.fc_pool(node_feat))]                                    |
|        for v in g.dst_vertex:                                                                7   ||                                                                                                  |
|            if edge_weight is not None:                                                       8   ||  19         if edge_weight is not None:                                                          |
|                g.set_edge_attr({'w': edge_weight})                                           9   ||  20             w, = [edge_weight]                                                               |
|                neigh_feat = [s.h * e.w for s, e in v.inedges]                                10  ||  21             neigh_feat = GATHER(h, src_idx, 0) * w                                           |
|            else:                                                                             11  ||  22         else:                                                                                |
|                neigh_feat = [u.h for u in v.innbs]                                           12  ||  23             neigh_feat = GATHER(h, src_idx, 0)                                               |
|            if self.agg_type == 'mean':                                                       13  ||  24         if self.agg_type == 'mean':                                                          |
|                v.h = g.avg(neigh_feat)                                                       14  ||  25             SCATTER_INPUT_SNAPSHOT3 = neigh_feat                                             |
|                                                                                                  ||  26             h = SCATTER_ADD(                                                                 |
|                                                                                                  ||                     ZEROS(                                                                       |
|                                                                                                  ||                         (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT3)[1:],                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT3.dtype                                            |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT3                                                      |
|                                                                                                  ||                 ) / (SCATTER_ADD(                                                                |
|                                                                                                  ||                     ZEROS((n_nodes, 1), scatter_dst_idx.dtype),                                  |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     ms.ops.ones_like(scatter_dst_idx)                                            |
|                                                                                                  ||                 ) + 1e-15)                                                                       |
|                if self.in_feat_size <= self.out_feat_size:                                   15  ||  27             if self.in_feat_size <= self.out_feat_size:                                      |
|                    v.h = self.dense_neigh(v.h)                                               16  ||  28                 h = self.dense_neigh(h)                                                      |
|            if self.agg_type == 'pool':                                                       17  ||  29         if self.agg_type == 'pool':                                                          |
|                v.h = g.max(neigh_feat)                                                       18  ||  30             SCATTER_INPUT_SNAPSHOT4 = neigh_feat                                             |
|                                                                                                  ||  31             h = MASKED_FILL(                                                                 |
|                                                                                                  ||                     SCATTER_MAX(                                                                 |
|                                                                                                  ||                         FILL(                                                                    |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT4.dtype,                                       |
|                                                                                                  ||                             (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT4)[1:],                     |
|                                                                                                  ||                             -1e1000                                                              |
|                                                                                                  ||                         ),                                                                       |
|                                                                                                  ||                         scatter_dst_idx,                                                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT4                                                  |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     IS_INF(                                                                      |
|                                                                                                  ||                         SCATTER_MAX(                                                             |
|                                                                                                  ||                             FILL(                                                                |
|                                                                                                  ||                                 SCATTER_INPUT_SNAPSHOT4.dtype,                                   |
|                                                                                                  ||                                 (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT4)[1:],                 |
|                                                                                                  ||                                 -1e1000                                                          |
|                                                                                                  ||                             ),                                                                   |
|                                                                                                  ||                             scatter_dst_idx,                                                     |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT4                                              |
|                                                                                                  ||                         )                                                                        |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     0.0                                                                          |
|                                                                                                  ||                 )                                                                                |
|                v.h = self.dense_neigh(v.h)                                                   19  ||  32             h = self.dense_neigh(h)                                                          |
|            if self.agg_type == 'lstm':                                                       20  ||  33         if self.agg_type == 'lstm':                                                          |
|                init_h = (                                                                    21  ||  34             init_h = (                                                                       |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                )                                                                                 ||                 )                                                                                |
|                _, (v.h, _) = self.lstm(neigh_feat, init_h)                                   22  ||  35             _, (h, _) = self.lstm(neigh_feat, init_h)                                        |
|                v.h = self.dense_neigh(ms.ops.Squeeze()(v.h, 0))                              23  ||  36             h = self.dense_neigh(ms.ops.Squeeze()(h, 0))                                     |
|        out_feat = [v.h for v in g.dst_vertex]                                                24  ||  37         out_feat = h                                                                         |
|        ret = self.dense_self(node_feat) + out_feat                                           25  ||  38         ret = self.dense_self(node_feat) + out_feat                                          |
|        if self.bias is not None:                                                             26  ||  39         if self.bias is not None:                                                            |
|            ret = ret + self.bias                                                             27  ||  40             ret = ret + self.bias                                                            |
|        if self.activation is not None:                                                       28  ||  41         if self.activation is not None:                                                      |
|            ret = self.activation(ret)                                                        29  ||  42             ret = self.activation(ret)                                                       |
|        if self.norm is not None:                                                             30  ||  43         if self.norm is not None:                                                            |
|            ret = self.norm(self.ret)                                                         31  ||  44             ret = self.norm(self.ret)                                                        |
|        return ret                                                                            32  ||  45         return ret                                                                           |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, node_feat, edge_weight, g: Graph):                                    1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             node_feat,                                                                           |
|                                                                                                  ||             edge_weight,                                                                         |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             UNUSED_0=None,                                                                       |
|                                                                                                  ||             UNUSED_1=None,                                                                       |
|                                                                                                  ||             UNUSED_2=None,                                                                       |
|                                                                                                  ||             UNUSED_3=None,                                                                       |
|                                                                                                  ||             UNUSED_4=None                                                                        |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|        g.set_vertex_attr({'h': node_feat})                                                   2   ||  14         h, = [node_feat]                                                                     |
|        if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:                3   ||  15         if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:               |
|            g.set_vertex_attr({'h': self.dense_neigh(node_feat)})                             4   ||  16             h, = [self.dense_neigh(node_feat)]                                               |
|        if self.agg_type == 'pool':                                                           5   ||  17         if self.agg_type == 'pool':                                                          |
|            g.set_vertex_attr({'h': ms.ops.ReLU()(self.fc_pool(node_feat))})                  6   ||  18             h, = [ms.ops.ReLU()(self.fc_pool(node_feat))]                                    |
|        for v in g.dst_vertex:                                                                7   ||                                                                                                  |
|            if edge_weight is not None:                                                       8   ||  19         if edge_weight is not None:                                                          |
|                g.set_edge_attr({'w': edge_weight})                                           9   ||  20             w, = [edge_weight]                                                               |
|                neigh_feat = [s.h * e.w for s, e in v.inedges]                                10  ||  21             neigh_feat = GATHER(h, src_idx, 0) * w                                           |
|            else:                                                                             11  ||  22         else:                                                                                |
|                neigh_feat = [u.h for u in v.innbs]                                           12  ||  23             neigh_feat = GATHER(h, src_idx, 0)                                               |
|            if self.agg_type == 'mean':                                                       13  ||  24         if self.agg_type == 'mean':                                                          |
|                v.h = g.avg(neigh_feat)                                                       14  ||  25             SCATTER_INPUT_SNAPSHOT5 = neigh_feat                                             |
|                                                                                                  ||  26             h = SCATTER_ADD(                                                                 |
|                                                                                                  ||                     ZEROS(                                                                       |
|                                                                                                  ||                         (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT5)[1:],                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT5.dtype                                            |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT5                                                      |
|                                                                                                  ||                 ) / (SCATTER_ADD(                                                                |
|                                                                                                  ||                     ZEROS((n_nodes, 1), scatter_dst_idx.dtype),                                  |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     ms.ops.ones_like(scatter_dst_idx)                                            |
|                                                                                                  ||                 ) + 1e-15)                                                                       |
|                if self.in_feat_size <= self.out_feat_size:                                   15  ||  27             if self.in_feat_size <= self.out_feat_size:                                      |
|                    v.h = self.dense_neigh(v.h)                                               16  ||  28                 h = self.dense_neigh(h)                                                      |
|            if self.agg_type == 'pool':                                                       17  ||  29         if self.agg_type == 'pool':                                                          |
|                v.h = g.max(neigh_feat)                                                       18  ||  30             SCATTER_INPUT_SNAPSHOT6 = neigh_feat                                             |
|                                                                                                  ||  31             h = MASKED_FILL(                                                                 |
|                                                                                                  ||                     SCATTER_MAX(                                                                 |
|                                                                                                  ||                         FILL(                                                                    |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT6.dtype,                                       |
|                                                                                                  ||                             (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT6)[1:],                     |
|                                                                                                  ||                             -1e1000                                                              |
|                                                                                                  ||                         ),                                                                       |
|                                                                                                  ||                         scatter_dst_idx,                                                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT6                                                  |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     IS_INF(                                                                      |
|                                                                                                  ||                         SCATTER_MAX(                                                             |
|                                                                                                  ||                             FILL(                                                                |
|                                                                                                  ||                                 SCATTER_INPUT_SNAPSHOT6.dtype,                                   |
|                                                                                                  ||                                 (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT6)[1:],                 |
|                                                                                                  ||                                 -1e1000                                                          |
|                                                                                                  ||                             ),                                                                   |
|                                                                                                  ||                             scatter_dst_idx,                                                     |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT6                                              |
|                                                                                                  ||                         )                                                                        |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     0.0                                                                          |
|                                                                                                  ||                 )                                                                                |
|                v.h = self.dense_neigh(v.h)                                                   19  ||  32             h = self.dense_neigh(h)                                                          |
|            if self.agg_type == 'lstm':                                                       20  ||  33         if self.agg_type == 'lstm':                                                          |
|                init_h = (                                                                    21  ||  34             init_h = (                                                                       |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                )                                                                                 ||                 )                                                                                |
|                _, (v.h, _) = self.lstm(neigh_feat, init_h)                                   22  ||  35             _, (h, _) = self.lstm(neigh_feat, init_h)                                        |
|                v.h = self.dense_neigh(ms.ops.Squeeze()(v.h, 0))                              23  ||  36             h = self.dense_neigh(ms.ops.Squeeze()(h, 0))                                     |
|        out_feat = [v.h for v in g.dst_vertex]                                                24  ||  37         out_feat = h                                                                         |
|        ret = self.dense_self(node_feat) + out_feat                                           25  ||  38         ret = self.dense_self(node_feat) + out_feat                                          |
|        if self.bias is not None:                                                             26  ||  39         if self.bias is not None:                                                            |
|            ret = ret + self.bias                                                             27  ||  40             ret = ret + self.bias                                                            |
|        if self.activation is not None:                                                       28  ||  41         if self.activation is not None:                                                      |
|            ret = self.activation(ret)                                                        29  ||  42             ret = self.activation(ret)                                                       |
|        if self.norm is not None:                                                             30  ||  43         if self.norm is not None:                                                            |
|            ret = self.norm(self.ret)                                                         31  ||  44             ret = self.norm(self.ret)                                                        |
|        return ret                                                                            32  ||  45         return ret                                                                           |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, node_feat, edge_weight, g: Graph):                                    1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             node_feat,                                                                           |
|                                                                                                  ||             edge_weight,                                                                         |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             UNUSED_0=None,                                                                       |
|                                                                                                  ||             UNUSED_1=None,                                                                       |
|                                                                                                  ||             UNUSED_2=None,                                                                       |
|                                                                                                  ||             UNUSED_3=None,                                                                       |
|                                                                                                  ||             UNUSED_4=None                                                                        |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|        g.set_vertex_attr({'h': node_feat})                                                   2   ||  14         h, = [node_feat]                                                                     |
|        if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:                3   ||  15         if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:               |
|            g.set_vertex_attr({'h': self.dense_neigh(node_feat)})                             4   ||  16             h, = [self.dense_neigh(node_feat)]                                               |
|        if self.agg_type == 'pool':                                                           5   ||  17         if self.agg_type == 'pool':                                                          |
|            g.set_vertex_attr({'h': ms.ops.ReLU()(self.fc_pool(node_feat))})                  6   ||  18             h, = [ms.ops.ReLU()(self.fc_pool(node_feat))]                                    |
|        for v in g.dst_vertex:                                                                7   ||                                                                                                  |
|            if edge_weight is not None:                                                       8   ||  19         if edge_weight is not None:                                                          |
|                g.set_edge_attr({'w': edge_weight})                                           9   ||  20             w, = [edge_weight]                                                               |
|                neigh_feat = [s.h * e.w for s, e in v.inedges]                                10  ||  21             neigh_feat = GATHER(h, src_idx, 0) * w                                           |
|            else:                                                                             11  ||  22         else:                                                                                |
|                neigh_feat = [u.h for u in v.innbs]                                           12  ||  23             neigh_feat = GATHER(h, src_idx, 0)                                               |
|            if self.agg_type == 'mean':                                                       13  ||  24         if self.agg_type == 'mean':                                                          |
|                v.h = g.avg(neigh_feat)                                                       14  ||  25             SCATTER_INPUT_SNAPSHOT7 = neigh_feat                                             |
|                                                                                                  ||  26             h = SCATTER_ADD(                                                                 |
|                                                                                                  ||                     ZEROS(                                                                       |
|                                                                                                  ||                         (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT7)[1:],                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT7.dtype                                            |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT7                                                      |
|                                                                                                  ||                 ) / (SCATTER_ADD(                                                                |
|                                                                                                  ||                     ZEROS((n_nodes, 1), scatter_dst_idx.dtype),                                  |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     ms.ops.ones_like(scatter_dst_idx)                                            |
|                                                                                                  ||                 ) + 1e-15)                                                                       |
|                if self.in_feat_size <= self.out_feat_size:                                   15  ||  27             if self.in_feat_size <= self.out_feat_size:                                      |
|                    v.h = self.dense_neigh(v.h)                                               16  ||  28                 h = self.dense_neigh(h)                                                      |
|            if self.agg_type == 'pool':                                                       17  ||  29         if self.agg_type == 'pool':                                                          |
|                v.h = g.max(neigh_feat)                                                       18  ||  30             SCATTER_INPUT_SNAPSHOT8 = neigh_feat                                             |
|                                                                                                  ||  31             h = MASKED_FILL(                                                                 |
|                                                                                                  ||                     SCATTER_MAX(                                                                 |
|                                                                                                  ||                         FILL(                                                                    |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT8.dtype,                                       |
|                                                                                                  ||                             (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT8)[1:],                     |
|                                                                                                  ||                             -1e1000                                                              |
|                                                                                                  ||                         ),                                                                       |
|                                                                                                  ||                         scatter_dst_idx,                                                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT8                                                  |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     IS_INF(                                                                      |
|                                                                                                  ||                         SCATTER_MAX(                                                             |
|                                                                                                  ||                             FILL(                                                                |
|                                                                                                  ||                                 SCATTER_INPUT_SNAPSHOT8.dtype,                                   |
|                                                                                                  ||                                 (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT8)[1:],                 |
|                                                                                                  ||                                 -1e1000                                                          |
|                                                                                                  ||                             ),                                                                   |
|                                                                                                  ||                             scatter_dst_idx,                                                     |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT8                                              |
|                                                                                                  ||                         )                                                                        |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     0.0                                                                          |
|                                                                                                  ||                 )                                                                                |
|                v.h = self.dense_neigh(v.h)                                                   19  ||  32             h = self.dense_neigh(h)                                                          |
|            if self.agg_type == 'lstm':                                                       20  ||  33         if self.agg_type == 'lstm':                                                          |
|                init_h = (                                                                    21  ||  34             init_h = (                                                                       |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                )                                                                                 ||                 )                                                                                |
|                _, (v.h, _) = self.lstm(neigh_feat, init_h)                                   22  ||  35             _, (h, _) = self.lstm(neigh_feat, init_h)                                        |
|                v.h = self.dense_neigh(ms.ops.Squeeze()(v.h, 0))                              23  ||  36             h = self.dense_neigh(ms.ops.Squeeze()(h, 0))                                     |
|        out_feat = [v.h for v in g.dst_vertex]                                                24  ||  37         out_feat = h                                                                         |
|        ret = self.dense_self(node_feat) + out_feat                                           25  ||  38         ret = self.dense_self(node_feat) + out_feat                                          |
|        if self.bias is not None:                                                             26  ||  39         if self.bias is not None:                                                            |
|            ret = ret + self.bias                                                             27  ||  40             ret = ret + self.bias                                                            |
|        if self.activation is not None:                                                       28  ||  41         if self.activation is not None:                                                      |
|            ret = self.activation(ret)                                                        29  ||  42             ret = self.activation(ret)                                                       |
|        if self.norm is not None:                                                             30  ||  43         if self.norm is not None:                                                            |
|            ret = self.norm(self.ret)                                                         31  ||  44             ret = self.norm(self.ret)                                                        |
|        return ret                                                                            32  ||  45         return ret                                                                           |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, h, g: BatchedGraph):                                                  1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             h,                                                                                   |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             ver_subgraph_idx,                                                                    |
|                                                                                                  ||             edge_subgraph_idx,                                                                   |
|                                                                                                  ||             graph_mask                                                                           |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|                                                                                                  ||  14         scatter_ver_subgraph_idx = RESHAPE(ver_subgraph_idx, (SHAPE(ver_subgraph_idx)[0], 1))|
|                                                                                                  ||  15         scatter_edge_subgraph_idx = RESHAPE(                                                 |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 (SHAPE(edge_subgraph_idx)[0], 1)                                                 |
|                                                                                                  ||             )                                                                                    |
|                                                                                                  ||  16         n_graphs = SHAPE(graph_mask)[0]                                                      |
|        feat = self.feat_gc(h, None, g)                                                       2   ||  17         feat = self.feat_gc(                                                                 |
|                                                                                                  ||                 h,                                                                               |
|                                                                                                  ||                 None,                                                                            |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
|        assign_tensor = self.pool_gc(h, None, g)                                              3   ||  18         assign_tensor = self.pool_gc(                                                        |
|                                                                                                  ||                 h,                                                                               |
|                                                                                                  ||                 None,                                                                            |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
|        assign_tensor = ops.Softmax(1)(assign_tensor)                                         4   ||  19         assign_tensor = ops.Softmax(1)(assign_tensor)                                        |
|        node_size, assign_size = ops.Shape()(assign_tensor)                                   5   ||  20         node_size, assign_size = ops.Shape()(assign_tensor)                                  |
|        graph_size = ops.Shape()(g.graph_mask)[0]                                             6   ||  21         graph_size = ops.Shape()(graph_mask)[0]                                              |
|        assign_tensor = ops.TensorScatterAdd()(                                               7   ||  22         assign_tensor = ops.TensorScatterAdd()(                                              |
|            ops.Zeros()((graph_size, node_size, assign_size), ms.float32),                        ||                 ops.Zeros()((graph_size, node_size, assign_size), ms.float32),                   |
|            ops.Transpose()(                                                                      ||                 ops.Transpose()(                                                                 |
|                ops.Stack()([g.ver_subgraph_idx, ms.nn.Range(0, node_size, 1)()]),                ||                     ops.Stack()([ver_subgraph_idx, ms.nn.Range(0, node_size, 1)()]),             |
|                (1, 0)                                                                            ||                     (1, 0)                                                                       |
|            ),                                                                                    ||                 ),                                                                               |
|            assign_tensor                                                                         ||                 assign_tensor                                                                    |
|        )                                                                                         ||             )                                                                                    |
|        assign_tensor = ops.Transpose()(assign_tensor, (1, 0, 2))                             8   ||  23         assign_tensor = ops.Transpose()(assign_tensor, (1, 0, 2))                            |
|        assign_tensor = ops.Reshape()(assign_tensor, (node_size, graph_size * assign_size))   9   ||  24         assign_tensor = ops.Reshape()(assign_tensor, (node_size, graph_size * assign_size))  |
|        node_mask = ops.Gather()(g.graph_mask, g.ver_subgraph_idx, 0)                         10  ||  25         node_mask = ops.Gather()(graph_mask, ver_subgraph_idx, 0)                            |
|        assign_tensor = assign_tensor * ops.Reshape()(node_mask, (node_size, 1))              11  ||  26         assign_tensor = assign_tensor * ops.Reshape()(node_mask, (node_size, 1))             |
|        h = ops.MatMul(transpose_a=True)(assign_tensor, feat)                                 12  ||  27         h = ops.MatMul(transpose_a=True)(assign_tensor, feat)                                |
|        adj_new = ops.TensorScatterAdd()(                                                     13  ||  28         adj_new = ops.TensorScatterAdd()(                                                    |
|            ops.Zeros()((node_size, graph_size * assign_size), ms.float32),                       ||                 ops.Zeros()((node_size, graph_size * assign_size), ms.float32),                  |
|            ops.Reshape()(g.dst_idx, (ops.Shape()(g.dst_idx)[0], 1)),                             ||                 ops.Reshape()(dst_idx, (ops.Shape()(dst_idx)[0], 1)),                            |
|            ops.Gather()(assign_tensor, g.src_idx, 0)                                             ||                 ops.Gather()(assign_tensor, src_idx, 0)                                          |
|        )                                                                                         ||             )                                                                                    |
|        adj_new = ops.MatMul(transpose_a=True)(assign_tensor, adj_new)                        14  ||  29         adj_new = ops.MatMul(transpose_a=True)(assign_tensor, adj_new)                       |
|        node_count = ops.Shape()(assign_tensor)[0]                                            15  ||  30         node_count = ops.Shape()(assign_tensor)[0]                                           |
|        adj = ops.ScatterNd()(                                                                16  ||  31         adj = ops.ScatterNd()(                                                               |
|            ms.ops.Transpose()(ms.ops.Stack()([g.src_idx, g.dst_idx]), (1, 0)),                   ||                 ms.ops.Transpose()(ms.ops.Stack()([src_idx, dst_idx]), (1, 0)),                  |
|            ops.Ones()(ops.Shape()(g.src_idx), ms.int32),                                         ||                 ops.Ones()(ops.Shape()(src_idx), ms.int32),                                      |
|            (node_count, node_count)                                                              ||                 (node_count, node_count)                                                         |
|        )                                                                                         ||             )                                                                                    |
|        adj = adj.astype('float32')                                                           17  ||  32         adj = adj.astype('float32')                                                          |
|        adj[-1][-1] = 0                                                                       18  ||  33         adj[-1][-1] = 0                                                                      |
|        if self.link_pred:                                                                    19  ||  34         if self.link_pred:                                                                   |
|            self.link_pred_loss = LinkPredLoss()(                                             20  ||  35             self.link_pred_loss = LinkPredLoss()(                                            |
|                adj,                                                                              ||                     adj,                                                                         |
|                assign_tensor,                                                                    ||                     assign_tensor,                                                               |
|                g.ver_subgraph_idx,                                                               ||                     ver_subgraph_idx,                                                            |
|                g.graph_mask                                                                      ||                     graph_mask                                                                   |
|            )                                                                                     ||                 )                                                                                |
|        self.entropy_loss = EntropyLoss()(assign_tensor, node_mask)                           21  ||  36         self.entropy_loss = EntropyLoss()(assign_tensor, node_mask)                          |
|        return adj_new, h                                                                     22  ||  37         return adj_new, h                                                                    |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, node_feat, edge_weight, g: Graph):                                    1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             node_feat,                                                                           |
|                                                                                                  ||             edge_weight,                                                                         |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             UNUSED_0=None,                                                                       |
|                                                                                                  ||             UNUSED_1=None,                                                                       |
|                                                                                                  ||             UNUSED_2=None,                                                                       |
|                                                                                                  ||             UNUSED_3=None,                                                                       |
|                                                                                                  ||             UNUSED_4=None                                                                        |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|        g.set_vertex_attr({'h': node_feat})                                                   2   ||  14         h, = [node_feat]                                                                     |
|        if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:                3   ||  15         if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:               |
|            g.set_vertex_attr({'h': self.dense_neigh(node_feat)})                             4   ||  16             h, = [self.dense_neigh(node_feat)]                                               |
|        if self.agg_type == 'pool':                                                           5   ||  17         if self.agg_type == 'pool':                                                          |
|            g.set_vertex_attr({'h': ms.ops.ReLU()(self.fc_pool(node_feat))})                  6   ||  18             h, = [ms.ops.ReLU()(self.fc_pool(node_feat))]                                    |
|        for v in g.dst_vertex:                                                                7   ||                                                                                                  |
|            if edge_weight is not None:                                                       8   ||  19         if edge_weight is not None:                                                          |
|                g.set_edge_attr({'w': edge_weight})                                           9   ||  20             w, = [edge_weight]                                                               |
|                neigh_feat = [s.h * e.w for s, e in v.inedges]                                10  ||  21             neigh_feat = GATHER(h, src_idx, 0) * w                                           |
|            else:                                                                             11  ||  22         else:                                                                                |
|                neigh_feat = [u.h for u in v.innbs]                                           12  ||  23             neigh_feat = GATHER(h, src_idx, 0)                                               |
|            if self.agg_type == 'mean':                                                       13  ||  24         if self.agg_type == 'mean':                                                          |
|                v.h = g.avg(neigh_feat)                                                       14  ||  25             SCATTER_INPUT_SNAPSHOT9 = neigh_feat                                             |
|                                                                                                  ||  26             h = SCATTER_ADD(                                                                 |
|                                                                                                  ||                     ZEROS(                                                                       |
|                                                                                                  ||                         (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT9)[1:],                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT9.dtype                                            |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT9                                                      |
|                                                                                                  ||                 ) / (SCATTER_ADD(                                                                |
|                                                                                                  ||                     ZEROS((n_nodes, 1), scatter_dst_idx.dtype),                                  |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     ms.ops.ones_like(scatter_dst_idx)                                            |
|                                                                                                  ||                 ) + 1e-15)                                                                       |
|                if self.in_feat_size <= self.out_feat_size:                                   15  ||  27             if self.in_feat_size <= self.out_feat_size:                                      |
|                    v.h = self.dense_neigh(v.h)                                               16  ||  28                 h = self.dense_neigh(h)                                                      |
|            if self.agg_type == 'pool':                                                       17  ||  29         if self.agg_type == 'pool':                                                          |
|                v.h = g.max(neigh_feat)                                                       18  ||  30             SCATTER_INPUT_SNAPSHOT10 = neigh_feat                                            |
|                                                                                                  ||  31             h = MASKED_FILL(                                                                 |
|                                                                                                  ||                     SCATTER_MAX(                                                                 |
|                                                                                                  ||                         FILL(                                                                    |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT10.dtype,                                      |
|                                                                                                  ||                             (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT10)[1:],                    |
|                                                                                                  ||                             -1e1000                                                              |
|                                                                                                  ||                         ),                                                                       |
|                                                                                                  ||                         scatter_dst_idx,                                                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT10                                                 |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     IS_INF(                                                                      |
|                                                                                                  ||                         SCATTER_MAX(                                                             |
|                                                                                                  ||                             FILL(                                                                |
|                                                                                                  ||                                 SCATTER_INPUT_SNAPSHOT10.dtype,                                  |
|                                                                                                  ||                                 (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT10)[1:],                |
|                                                                                                  ||                                 -1e1000                                                          |
|                                                                                                  ||                             ),                                                                   |
|                                                                                                  ||                             scatter_dst_idx,                                                     |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT10                                             |
|                                                                                                  ||                         )                                                                        |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     0.0                                                                          |
|                                                                                                  ||                 )                                                                                |
|                v.h = self.dense_neigh(v.h)                                                   19  ||  32             h = self.dense_neigh(h)                                                          |
|            if self.agg_type == 'lstm':                                                       20  ||  33         if self.agg_type == 'lstm':                                                          |
|                init_h = (                                                                    21  ||  34             init_h = (                                                                       |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                )                                                                                 ||                 )                                                                                |
|                _, (v.h, _) = self.lstm(neigh_feat, init_h)                                   22  ||  35             _, (h, _) = self.lstm(neigh_feat, init_h)                                        |
|                v.h = self.dense_neigh(ms.ops.Squeeze()(v.h, 0))                              23  ||  36             h = self.dense_neigh(ms.ops.Squeeze()(h, 0))                                     |
|        out_feat = [v.h for v in g.dst_vertex]                                                24  ||  37         out_feat = h                                                                         |
|        ret = self.dense_self(node_feat) + out_feat                                           25  ||  38         ret = self.dense_self(node_feat) + out_feat                                          |
|        if self.bias is not None:                                                             26  ||  39         if self.bias is not None:                                                            |
|            ret = ret + self.bias                                                             27  ||  40             ret = ret + self.bias                                                            |
|        if self.activation is not None:                                                       28  ||  41         if self.activation is not None:                                                      |
|            ret = self.activation(ret)                                                        29  ||  42             ret = self.activation(ret)                                                       |
|        if self.norm is not None:                                                             30  ||  43         if self.norm is not None:                                                            |
|            ret = self.norm(self.ret)                                                         31  ||  44             ret = self.norm(self.ret)                                                        |
|        return ret                                                                            32  ||  45         return ret                                                                           |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, node_feat, edge_weight, g: Graph):                                    1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             node_feat,                                                                           |
|                                                                                                  ||             edge_weight,                                                                         |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             UNUSED_0=None,                                                                       |
|                                                                                                  ||             UNUSED_1=None,                                                                       |
|                                                                                                  ||             UNUSED_2=None,                                                                       |
|                                                                                                  ||             UNUSED_3=None,                                                                       |
|                                                                                                  ||             UNUSED_4=None                                                                        |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|        g.set_vertex_attr({'h': node_feat})                                                   2   ||  14         h, = [node_feat]                                                                     |
|        if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:                3   ||  15         if self.agg_type == 'mean' and self.in_feat_size > self.out_feat_size:               |
|            g.set_vertex_attr({'h': self.dense_neigh(node_feat)})                             4   ||  16             h, = [self.dense_neigh(node_feat)]                                               |
|        if self.agg_type == 'pool':                                                           5   ||  17         if self.agg_type == 'pool':                                                          |
|            g.set_vertex_attr({'h': ms.ops.ReLU()(self.fc_pool(node_feat))})                  6   ||  18             h, = [ms.ops.ReLU()(self.fc_pool(node_feat))]                                    |
|        for v in g.dst_vertex:                                                                7   ||                                                                                                  |
|            if edge_weight is not None:                                                       8   ||  19         if edge_weight is not None:                                                          |
|                g.set_edge_attr({'w': edge_weight})                                           9   ||  20             w, = [edge_weight]                                                               |
|                neigh_feat = [s.h * e.w for s, e in v.inedges]                                10  ||  21             neigh_feat = GATHER(h, src_idx, 0) * w                                           |
|            else:                                                                             11  ||  22         else:                                                                                |
|                neigh_feat = [u.h for u in v.innbs]                                           12  ||  23             neigh_feat = GATHER(h, src_idx, 0)                                               |
|            if self.agg_type == 'mean':                                                       13  ||  24         if self.agg_type == 'mean':                                                          |
|                v.h = g.avg(neigh_feat)                                                       14  ||  25             SCATTER_INPUT_SNAPSHOT11 = neigh_feat                                            |
|                                                                                                  ||  26             h = SCATTER_ADD(                                                                 |
|                                                                                                  ||                     ZEROS(                                                                       |
|                                                                                                  ||                         (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT11)[1:],                        |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT11.dtype                                           |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT11                                                     |
|                                                                                                  ||                 ) / (SCATTER_ADD(                                                                |
|                                                                                                  ||                     ZEROS((n_nodes, 1), scatter_dst_idx.dtype),                                  |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     ms.ops.ones_like(scatter_dst_idx)                                            |
|                                                                                                  ||                 ) + 1e-15)                                                                       |
|                if self.in_feat_size <= self.out_feat_size:                                   15  ||  27             if self.in_feat_size <= self.out_feat_size:                                      |
|                    v.h = self.dense_neigh(v.h)                                               16  ||  28                 h = self.dense_neigh(h)                                                      |
|            if self.agg_type == 'pool':                                                       17  ||  29         if self.agg_type == 'pool':                                                          |
|                v.h = g.max(neigh_feat)                                                       18  ||  30             SCATTER_INPUT_SNAPSHOT12 = neigh_feat                                            |
|                                                                                                  ||  31             h = MASKED_FILL(                                                                 |
|                                                                                                  ||                     SCATTER_MAX(                                                                 |
|                                                                                                  ||                         FILL(                                                                    |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT12.dtype,                                      |
|                                                                                                  ||                             (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT12)[1:],                    |
|                                                                                                  ||                             -1e1000                                                              |
|                                                                                                  ||                         ),                                                                       |
|                                                                                                  ||                         scatter_dst_idx,                                                         |
|                                                                                                  ||                         SCATTER_INPUT_SNAPSHOT12                                                 |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     IS_INF(                                                                      |
|                                                                                                  ||                         SCATTER_MAX(                                                             |
|                                                                                                  ||                             FILL(                                                                |
|                                                                                                  ||                                 SCATTER_INPUT_SNAPSHOT12.dtype,                                  |
|                                                                                                  ||                                 (n_nodes,) + SHAPE(SCATTER_INPUT_SNAPSHOT12)[1:],                |
|                                                                                                  ||                                 -1e1000                                                          |
|                                                                                                  ||                             ),                                                                   |
|                                                                                                  ||                             scatter_dst_idx,                                                     |
|                                                                                                  ||                             SCATTER_INPUT_SNAPSHOT12                                             |
|                                                                                                  ||                         )                                                                        |
|                                                                                                  ||                     ),                                                                           |
|                                                                                                  ||                     0.0                                                                          |
|                                                                                                  ||                 )                                                                                |
|                v.h = self.dense_neigh(v.h)                                                   19  ||  32             h = self.dense_neigh(h)                                                          |
|            if self.agg_type == 'lstm':                                                       20  ||  33         if self.agg_type == 'lstm':                                                          |
|                init_h = (                                                                    21  ||  34             init_h = (                                                                       |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                    ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),                ||                     ms.ops.Zeros()((1, n_edges, self.in_feat_size), ms.float32),                 |
|                )                                                                                 ||                 )                                                                                |
|                _, (v.h, _) = self.lstm(neigh_feat, init_h)                                   22  ||  35             _, (h, _) = self.lstm(neigh_feat, init_h)                                        |
|                v.h = self.dense_neigh(ms.ops.Squeeze()(v.h, 0))                              23  ||  36             h = self.dense_neigh(ms.ops.Squeeze()(h, 0))                                     |
|        out_feat = [v.h for v in g.dst_vertex]                                                24  ||  37         out_feat = h                                                                         |
|        ret = self.dense_self(node_feat) + out_feat                                           25  ||  38         ret = self.dense_self(node_feat) + out_feat                                          |
|        if self.bias is not None:                                                             26  ||  39         if self.bias is not None:                                                            |
|            ret = ret + self.bias                                                             27  ||  40             ret = ret + self.bias                                                            |
|        if self.activation is not None:                                                       28  ||  41         if self.activation is not None:                                                      |
|            ret = self.activation(ret)                                                        29  ||  42             ret = self.activation(ret)                                                       |
|        if self.norm is not None:                                                             30  ||  43         if self.norm is not None:                                                            |
|            ret = self.norm(self.ret)                                                         31  ||  44             ret = self.norm(self.ret)                                                        |
|        return ret                                                                            32  ||  45         return ret                                                                           |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, x, label, g: BatchedGraph):                                           1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             x,                                                                                   |
|                                                                                                  ||             label,                                                                               |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             ver_subgraph_idx,                                                                    |
|                                                                                                  ||             edge_subgraph_idx,                                                                   |
|                                                                                                  ||             graph_mask                                                                           |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          FILL = ms.ops.Fill()                                                                 |
|                                                                                                  ||  8          MASKED_FILL = ms.ops.MaskedFill()                                                    |
|                                                                                                  ||  9          IS_INF = ms.ops.IsInf()                                                              |
|                                                                                                  ||  10         SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  11         RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  12         scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  13         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|                                                                                                  ||  14         scatter_ver_subgraph_idx = RESHAPE(ver_subgraph_idx, (SHAPE(ver_subgraph_idx)[0], 1))|
|                                                                                                  ||  15         scatter_edge_subgraph_idx = RESHAPE(                                                 |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 (SHAPE(edge_subgraph_idx)[0], 1)                                                 |
|                                                                                                  ||             )                                                                                    |
|                                                                                                  ||  16         n_graphs = SHAPE(graph_mask)[0]                                                      |
|        pred = self.net(x, g)                                                                 2   ||  17         pred = self.net(                                                                     |
|                                                                                                  ||                 x,                                                                               |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
|        return self.net.loss(pred, label, g)                                                  3   ||  18         return self.net.loss(                                                                |
|                                                                                                  ||                 pred,                                                                            |
|                                                                                                  ||                 label,                                                                           |
|                                                                                                  ||                 src_idx,                                                                         |
|                                                                                                  ||                 dst_idx,                                                                         |
|                                                                                                  ||                 n_nodes,                                                                         |
|                                                                                                  ||                 n_edges,                                                                         |
|                                                                                                  ||                 ver_subgraph_idx,                                                                |
|                                                                                                  ||                 edge_subgraph_idx,                                                               |
|                                                                                                  ||                 graph_mask                                                                       |
|                                                                                                  ||             )                                                                                    |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[WARNING] ME(432265:139624987899712,MainProcess):2023-05-18-12:27:36.516.192 [mindspore/nn/layer/math.py:132] 'nn.Range' is deprecated from version 2.0 and will be removed in a future version,use 'ops.range' instead.
[WARNING] ME(432265:139624987899712,MainProcess):2023-05-18-12:27:36.873.343 [mindspore/common/_decorator.py:40] 'Norm' is deprecated from version 2.0 and will be removed in a future version, use 'ops.norm' instead.
[WARNING] ME(432265:139624987899712,MainProcess):2023-05-18-12:27:41.563.799 [mindspore/nn/layer/math.py:132] 'nn.Range' is deprecated from version 2.0 and will be removed in a future version,use 'ops.range' instead.
Epoch 0, Time 6.422 s, Train loss 1.7064886, Train acc 0.206, Val acc 0.233
Epoch 1, Time 0.481 s, Train loss 1.6917305, Train acc 0.217, Val acc 0.233
Early stop: 1/250, best_acc: 0.233
Epoch 2, Time 0.394 s, Train loss 1.6789702, Train acc 0.233, Val acc 0.267
Epoch 3, Time 0.521 s, Train loss 1.6713858, Train acc 0.240, Val acc 0.283
Epoch 4, Time 0.471 s, Train loss 1.6718845, Train acc 0.229, Val acc 0.267
Early stop: 1/250, best_acc: 0.283
Epoch 5, Time 0.447 s, Train loss 1.6641458, Train acc 0.275, Val acc 0.317
Epoch 6, Time 0.389 s, Train loss 1.6583868, Train acc 0.231, Val acc 0.250
Early stop: 1/250, best_acc: 0.317
Epoch 7, Time 0.495 s, Train loss 1.656835, Train acc 0.256, Val acc 0.317
Early stop: 2/250, best_acc: 0.317
Epoch 8, Time 0.478 s, Train loss 1.6510671, Train acc 0.210, Val acc 0.233
Early stop: 3/250, best_acc: 0.317
Epoch 9, Time 0.540 s, Train loss 1.66248, Train acc 0.242, Val acc 0.300
Early stop: 4/250, best_acc: 0.317
Epoch 10, Time 0.486 s, Train loss 1.6539736, Train acc 0.248, Val acc 0.317
Early stop: 5/250, best_acc: 0.317
Epoch 11, Time 0.483 s, Train loss 1.6435533, Train acc 0.256, Val acc 0.250
Early stop: 6/250, best_acc: 0.317
Epoch 12, Time 0.366 s, Train loss 1.6341974, Train acc 0.267, Val acc 0.333
Epoch 13, Time 0.474 s, Train loss 1.634097, Train acc 0.225, Val acc 0.250
Early stop: 1/250, best_acc: 0.333
Epoch 14, Time 0.445 s, Train loss 1.6255951, Train acc 0.285, Val acc 0.300
Early stop: 2/250, best_acc: 0.333
Epoch 15, Time 0.521 s, Train loss 1.6154679, Train acc 0.292, Val acc 0.350
Epoch 16, Time 0.384 s, Train loss 1.6074543, Train acc 0.290, Val acc 0.300
Early stop: 1/250, best_acc: 0.350
Epoch 17, Time 0.368 s, Train loss 1.61156, Train acc 0.219, Val acc 0.250
Early stop: 2/250, best_acc: 0.350
Epoch 18, Time 0.392 s, Train loss 1.6068789, Train acc 0.302, Val acc 0.333
Early stop: 3/250, best_acc: 0.350
Epoch 19, Time 0.449 s, Train loss 1.5910954, Train acc 0.258, Val acc 0.233
Early stop: 4/250, best_acc: 0.350
Epoch 20, Time 0.485 s, Train loss 1.5785607, Train acc 0.283, Val acc 0.367
Epoch 21, Time 0.446 s, Train loss 1.5662231, Train acc 0.285, Val acc 0.267
Early stop: 1/250, best_acc: 0.367
Epoch 22, Time 0.497 s, Train loss 1.5661079, Train acc 0.240, Val acc 0.233
Early stop: 2/250, best_acc: 0.367
Epoch 23, Time 0.385 s, Train loss 1.6008682, Train acc 0.319, Val acc 0.367
Early stop: 3/250, best_acc: 0.367
Epoch 24, Time 0.502 s, Train loss 1.576323, Train acc 0.317, Val acc 0.350
Early stop: 4/250, best_acc: 0.367
Epoch 25, Time 0.383 s, Train loss 1.5327238, Train acc 0.327, Val acc 0.333
Early stop: 5/250, best_acc: 0.367
Epoch 26, Time 0.368 s, Train loss 1.5316205, Train acc 0.360, Val acc 0.350
Early stop: 6/250, best_acc: 0.367
Epoch 27, Time 0.441 s, Train loss 1.4874936, Train acc 0.388, Val acc 0.417
Epoch 28, Time 0.456 s, Train loss 1.4507338, Train acc 0.417, Val acc 0.367
Early stop: 1/250, best_acc: 0.417
Epoch 29, Time 0.470 s, Train loss 1.4338444, Train acc 0.435, Val acc 0.433
Epoch 30, Time 0.396 s, Train loss 1.4172206, Train acc 0.460, Val acc 0.433
Early stop: 1/250, best_acc: 0.433
Epoch 31, Time 0.460 s, Train loss 1.41149, Train acc 0.442, Val acc 0.383
Early stop: 2/250, best_acc: 0.433
Epoch 32, Time 0.490 s, Train loss 1.4379822, Train acc 0.465, Val acc 0.417
Early stop: 3/250, best_acc: 0.433
Epoch 33, Time 0.426 s, Train loss 1.3974823, Train acc 0.438, Val acc 0.417
Early stop: 4/250, best_acc: 0.433
Epoch 34, Time 0.456 s, Train loss 1.3625757, Train acc 0.490, Val acc 0.383
Early stop: 5/250, best_acc: 0.433
Epoch 35, Time 0.368 s, Train loss 1.3566045, Train acc 0.492, Val acc 0.467
Epoch 36, Time 0.456 s, Train loss 1.340922, Train acc 0.506, Val acc 0.433
Early stop: 1/250, best_acc: 0.467
Epoch 37, Time 0.361 s, Train loss 1.2938114, Train acc 0.529, Val acc 0.433
Early stop: 2/250, best_acc: 0.467
Epoch 38, Time 0.459 s, Train loss 1.2826675, Train acc 0.537, Val acc 0.483
Epoch 39, Time 0.419 s, Train loss 1.2544947, Train acc 0.581, Val acc 0.517
Epoch 40, Time 0.432 s, Train loss 1.2169396, Train acc 0.517, Val acc 0.450
Early stop: 1/250, best_acc: 0.517
Epoch 41, Time 0.488 s, Train loss 1.1704968, Train acc 0.560, Val acc 0.467
Early stop: 2/250, best_acc: 0.517
Epoch 42, Time 0.442 s, Train loss 1.193745, Train acc 0.567, Val acc 0.483
Early stop: 3/250, best_acc: 0.517
Epoch 43, Time 0.362 s, Train loss 1.1681954, Train acc 0.571, Val acc 0.483
Early stop: 4/250, best_acc: 0.517
Epoch 44, Time 0.421 s, Train loss 1.1733896, Train acc 0.579, Val acc 0.483
Early stop: 5/250, best_acc: 0.517
Epoch 45, Time 0.460 s, Train loss 1.1438472, Train acc 0.652, Val acc 0.567
Epoch 46, Time 0.385 s, Train loss 1.0701123, Train acc 0.673, Val acc 0.550
Early stop: 1/250, best_acc: 0.567
Epoch 47, Time 0.411 s, Train loss 1.0106031, Train acc 0.671, Val acc 0.550
Early stop: 2/250, best_acc: 0.567
Epoch 48, Time 0.386 s, Train loss 1.0057458, Train acc 0.690, Val acc 0.600
Epoch 49, Time 0.428 s, Train loss 0.99255687, Train acc 0.665, Val acc 0.517
Early stop: 1/250, best_acc: 0.600
Epoch 50, Time 0.449 s, Train loss 0.9624395, Train acc 0.706, Val acc 0.567
Early stop: 2/250, best_acc: 0.600
Epoch 51, Time 0.443 s, Train loss 0.9536801, Train acc 0.723, Val acc 0.567
Early stop: 3/250, best_acc: 0.600
Epoch 52, Time 0.441 s, Train loss 0.97635716, Train acc 0.746, Val acc 0.567
Early stop: 4/250, best_acc: 0.600
Epoch 53, Time 0.488 s, Train loss 0.96464175, Train acc 0.721, Val acc 0.600
Early stop: 5/250, best_acc: 0.600
Epoch 54, Time 0.485 s, Train loss 0.94050926, Train acc 0.717, Val acc 0.550
Early stop: 6/250, best_acc: 0.600
Epoch 55, Time 0.445 s, Train loss 0.99274826, Train acc 0.685, Val acc 0.533
Early stop: 7/250, best_acc: 0.600
Epoch 56, Time 0.429 s, Train loss 0.85779, Train acc 0.771, Val acc 0.633
Epoch 57, Time 0.417 s, Train loss 0.8046947, Train acc 0.773, Val acc 0.617
Early stop: 1/250, best_acc: 0.633
Epoch 58, Time 0.420 s, Train loss 0.7757601, Train acc 0.754, Val acc 0.567
Early stop: 2/250, best_acc: 0.633
Epoch 59, Time 0.439 s, Train loss 0.814369, Train acc 0.769, Val acc 0.567
Early stop: 3/250, best_acc: 0.633
Epoch 60, Time 0.468 s, Train loss 0.87175035, Train acc 0.715, Val acc 0.550
Early stop: 4/250, best_acc: 0.633
Epoch 61, Time 0.412 s, Train loss 0.87673163, Train acc 0.698, Val acc 0.567
Early stop: 5/250, best_acc: 0.633
Epoch 62, Time 0.358 s, Train loss 0.7982448, Train acc 0.771, Val acc 0.600
Early stop: 6/250, best_acc: 0.633
Epoch 63, Time 0.421 s, Train loss 0.73642325, Train acc 0.800, Val acc 0.550
Early stop: 7/250, best_acc: 0.633
Epoch 64, Time 0.412 s, Train loss 0.7010088, Train acc 0.808, Val acc 0.483
Early stop: 8/250, best_acc: 0.633
Epoch 65, Time 0.394 s, Train loss 0.71504015, Train acc 0.840, Val acc 0.583
Early stop: 9/250, best_acc: 0.633
Epoch 66, Time 0.444 s, Train loss 0.67783755, Train acc 0.840, Val acc 0.583
Early stop: 10/250, best_acc: 0.633
Epoch 67, Time 0.454 s, Train loss 0.6550194, Train acc 0.806, Val acc 0.600
Early stop: 11/250, best_acc: 0.633
Epoch 68, Time 0.428 s, Train loss 0.6676238, Train acc 0.752, Val acc 0.567
Early stop: 12/250, best_acc: 0.633
Epoch 69, Time 0.522 s, Train loss 0.60199106, Train acc 0.835, Val acc 0.650
Epoch 70, Time 0.468 s, Train loss 0.54627097, Train acc 0.860, Val acc 0.633
Early stop: 1/250, best_acc: 0.650
Epoch 71, Time 0.456 s, Train loss 0.54646134, Train acc 0.869, Val acc 0.583
Early stop: 2/250, best_acc: 0.650
Epoch 72, Time 0.416 s, Train loss 0.53562737, Train acc 0.808, Val acc 0.600
Early stop: 3/250, best_acc: 0.650
Epoch 73, Time 0.378 s, Train loss 0.549462, Train acc 0.815, Val acc 0.550
Early stop: 4/250, best_acc: 0.650
Epoch 74, Time 0.396 s, Train loss 0.55073416, Train acc 0.817, Val acc 0.617
Early stop: 5/250, best_acc: 0.650
Epoch 75, Time 0.419 s, Train loss 0.5267834, Train acc 0.840, Val acc 0.617
Early stop: 6/250, best_acc: 0.650
Epoch 76, Time 0.453 s, Train loss 0.5649187, Train acc 0.852, Val acc 0.650
Early stop: 7/250, best_acc: 0.650
Epoch 77, Time 0.418 s, Train loss 0.53398067, Train acc 0.863, Val acc 0.617
Early stop: 8/250, best_acc: 0.650
Epoch 78, Time 0.390 s, Train loss 0.50380313, Train acc 0.844, Val acc 0.633
Early stop: 9/250, best_acc: 0.650
Epoch 79, Time 0.402 s, Train loss 0.49957582, Train acc 0.850, Val acc 0.583
Early stop: 10/250, best_acc: 0.650
Epoch 80, Time 0.427 s, Train loss 0.46956947, Train acc 0.815, Val acc 0.583
Early stop: 11/250, best_acc: 0.650
Epoch 81, Time 0.473 s, Train loss 0.44673976, Train acc 0.848, Val acc 0.550
Early stop: 12/250, best_acc: 0.650
Epoch 82, Time 0.444 s, Train loss 0.41646543, Train acc 0.860, Val acc 0.533
Early stop: 13/250, best_acc: 0.650
Epoch 83, Time 0.336 s, Train loss 0.39873838, Train acc 0.908, Val acc 0.600
Early stop: 14/250, best_acc: 0.650
Epoch 84, Time 0.437 s, Train loss 0.40928987, Train acc 0.919, Val acc 0.633
Early stop: 15/250, best_acc: 0.650
Epoch 85, Time 0.432 s, Train loss 0.4028697, Train acc 0.910, Val acc 0.617
Early stop: 16/250, best_acc: 0.650
Epoch 86, Time 0.400 s, Train loss 0.3883966, Train acc 0.917, Val acc 0.617
Early stop: 17/250, best_acc: 0.650
Epoch 87, Time 0.395 s, Train loss 0.34962785, Train acc 0.925, Val acc 0.633
Early stop: 18/250, best_acc: 0.650
Epoch 88, Time 0.414 s, Train loss 0.33028242, Train acc 0.912, Val acc 0.600
Early stop: 19/250, best_acc: 0.650
Epoch 89, Time 0.409 s, Train loss 0.36156556, Train acc 0.923, Val acc 0.650
Early stop: 20/250, best_acc: 0.650
Epoch 90, Time 0.413 s, Train loss 0.3391752, Train acc 0.931, Val acc 0.533
Early stop: 21/250, best_acc: 0.650
Epoch 91, Time 0.484 s, Train loss 0.3137073, Train acc 0.965, Val acc 0.567
Early stop: 22/250, best_acc: 0.650
Epoch 92, Time 0.356 s, Train loss 0.29543695, Train acc 0.938, Val acc 0.617
Early stop: 23/250, best_acc: 0.650
Epoch 93, Time 0.384 s, Train loss 0.24700405, Train acc 0.942, Val acc 0.600
Early stop: 24/250, best_acc: 0.650
Epoch 94, Time 0.387 s, Train loss 0.24677522, Train acc 0.950, Val acc 0.617
Early stop: 25/250, best_acc: 0.650
Epoch 95, Time 0.457 s, Train loss 0.30067888, Train acc 0.952, Val acc 0.667
Epoch 96, Time 0.451 s, Train loss 0.2510964, Train acc 0.908, Val acc 0.617
Early stop: 1/250, best_acc: 0.667
Epoch 97, Time 0.482 s, Train loss 0.29800838, Train acc 0.904, Val acc 0.617
Early stop: 2/250, best_acc: 0.667
Epoch 98, Time 0.443 s, Train loss 0.2544177, Train acc 0.902, Val acc 0.567
Early stop: 3/250, best_acc: 0.667
Epoch 99, Time 0.471 s, Train loss 0.28865027, Train acc 0.906, Val acc 0.633
Early stop: 4/250, best_acc: 0.667
Epoch 100, Time 0.390 s, Train loss 0.26789415, Train acc 0.925, Val acc 0.583
Early stop: 5/250, best_acc: 0.667
Epoch 101, Time 0.376 s, Train loss 0.20979248, Train acc 0.925, Val acc 0.567
Early stop: 6/250, best_acc: 0.667
Epoch 102, Time 0.478 s, Train loss 0.17252801, Train acc 0.963, Val acc 0.600
Early stop: 7/250, best_acc: 0.667
Epoch 103, Time 0.443 s, Train loss 0.16564254, Train acc 0.921, Val acc 0.583
Early stop: 8/250, best_acc: 0.667
Epoch 104, Time 0.429 s, Train loss 0.160008, Train acc 0.902, Val acc 0.550
Early stop: 9/250, best_acc: 0.667
Epoch 105, Time 0.427 s, Train loss 0.16334632, Train acc 0.923, Val acc 0.533
Early stop: 10/250, best_acc: 0.667
Epoch 106, Time 0.506 s, Train loss 0.15712732, Train acc 0.935, Val acc 0.583
Early stop: 11/250, best_acc: 0.667
Epoch 107, Time 0.446 s, Train loss 0.15501846, Train acc 0.923, Val acc 0.567
Early stop: 12/250, best_acc: 0.667
Epoch 108, Time 0.499 s, Train loss 0.18118753, Train acc 0.935, Val acc 0.600
Early stop: 13/250, best_acc: 0.667
Epoch 109, Time 0.473 s, Train loss 0.21216364, Train acc 0.892, Val acc 0.583
Early stop: 14/250, best_acc: 0.667
Epoch 110, Time 0.443 s, Train loss 0.25173578, Train acc 0.912, Val acc 0.567
Early stop: 15/250, best_acc: 0.667
Epoch 111, Time 0.458 s, Train loss 0.18857504, Train acc 0.900, Val acc 0.533
Early stop: 16/250, best_acc: 0.667
Epoch 112, Time 0.423 s, Train loss 0.2787898, Train acc 0.912, Val acc 0.517
Early stop: 17/250, best_acc: 0.667
Epoch 113, Time 0.350 s, Train loss 0.22484933, Train acc 0.950, Val acc 0.567
Early stop: 18/250, best_acc: 0.667
Epoch 114, Time 0.405 s, Train loss 0.18783577, Train acc 0.967, Val acc 0.633
Early stop: 19/250, best_acc: 0.667
Epoch 115, Time 0.345 s, Train loss 0.20939486, Train acc 0.898, Val acc 0.633
Early stop: 20/250, best_acc: 0.667
Epoch 116, Time 0.479 s, Train loss 0.21730827, Train acc 0.960, Val acc 0.633
Early stop: 21/250, best_acc: 0.667
Epoch 117, Time 0.425 s, Train loss 0.15713699, Train acc 0.975, Val acc 0.667
Early stop: 22/250, best_acc: 0.667
Epoch 118, Time 0.429 s, Train loss 0.1355639, Train acc 0.952, Val acc 0.667
Early stop: 23/250, best_acc: 0.667
Epoch 119, Time 0.423 s, Train loss 0.12293565, Train acc 0.944, Val acc 0.650
Early stop: 24/250, best_acc: 0.667
Epoch 120, Time 0.397 s, Train loss 0.13625096, Train acc 0.973, Val acc 0.633
Early stop: 25/250, best_acc: 0.667
Epoch 121, Time 0.408 s, Train loss 0.1587422, Train acc 0.963, Val acc 0.650
Early stop: 26/250, best_acc: 0.667
Epoch 122, Time 0.470 s, Train loss 0.1510946, Train acc 0.944, Val acc 0.633
Early stop: 27/250, best_acc: 0.667
Epoch 123, Time 0.501 s, Train loss 0.13755798, Train acc 0.938, Val acc 0.650
Early stop: 28/250, best_acc: 0.667
Epoch 124, Time 0.455 s, Train loss 0.1288264, Train acc 0.912, Val acc 0.617
Early stop: 29/250, best_acc: 0.667
Epoch 125, Time 0.475 s, Train loss 0.15512167, Train acc 0.915, Val acc 0.617
Early stop: 30/250, best_acc: 0.667
Epoch 126, Time 0.450 s, Train loss 0.16192411, Train acc 0.931, Val acc 0.617
Early stop: 31/250, best_acc: 0.667
Epoch 127, Time 0.449 s, Train loss 0.18230017, Train acc 0.912, Val acc 0.600
Early stop: 32/250, best_acc: 0.667
Epoch 128, Time 0.513 s, Train loss 0.17198002, Train acc 0.925, Val acc 0.600
Early stop: 33/250, best_acc: 0.667
Epoch 129, Time 0.466 s, Train loss 0.13033798, Train acc 0.873, Val acc 0.600
Early stop: 34/250, best_acc: 0.667
Epoch 130, Time 0.434 s, Train loss 0.14210929, Train acc 0.944, Val acc 0.533
Early stop: 35/250, best_acc: 0.667
Epoch 131, Time 0.443 s, Train loss 0.20320034, Train acc 0.908, Val acc 0.550
Early stop: 36/250, best_acc: 0.667
Epoch 132, Time 0.468 s, Train loss 0.18443727, Train acc 0.921, Val acc 0.533
Early stop: 37/250, best_acc: 0.667
Epoch 133, Time 0.473 s, Train loss 0.1809197, Train acc 0.910, Val acc 0.600
Early stop: 38/250, best_acc: 0.667
Epoch 134, Time 0.384 s, Train loss 0.1348609, Train acc 0.935, Val acc 0.550
Early stop: 39/250, best_acc: 0.667
Epoch 135, Time 0.459 s, Train loss 0.14584322, Train acc 0.873, Val acc 0.533
Early stop: 40/250, best_acc: 0.667
Epoch 136, Time 0.471 s, Train loss 0.14280766, Train acc 0.906, Val acc 0.617
Early stop: 41/250, best_acc: 0.667
Epoch 137, Time 0.480 s, Train loss 0.12956996, Train acc 0.879, Val acc 0.583
Early stop: 42/250, best_acc: 0.667
Epoch 138, Time 0.524 s, Train loss 0.27153617, Train acc 0.912, Val acc 0.567
Early stop: 43/250, best_acc: 0.667
Epoch 139, Time 0.410 s, Train loss 0.20232652, Train acc 0.940, Val acc 0.583
Early stop: 44/250, best_acc: 0.667
Epoch 140, Time 0.427 s, Train loss 0.16539143, Train acc 0.963, Val acc 0.583
Early stop: 45/250, best_acc: 0.667
Epoch 141, Time 0.412 s, Train loss 0.13425979, Train acc 0.979, Val acc 0.600
Early stop: 46/250, best_acc: 0.667
Epoch 142, Time 0.371 s, Train loss 0.10475817, Train acc 0.960, Val acc 0.583
Early stop: 47/250, best_acc: 0.667
Epoch 143, Time 0.506 s, Train loss 0.08289106, Train acc 0.988, Val acc 0.550
Early stop: 48/250, best_acc: 0.667
Epoch 144, Time 0.395 s, Train loss 0.0824121, Train acc 0.973, Val acc 0.600
Early stop: 49/250, best_acc: 0.667
Epoch 145, Time 0.367 s, Train loss 0.09631682, Train acc 0.969, Val acc 0.567
Early stop: 50/250, best_acc: 0.667
Epoch 146, Time 0.556 s, Train loss 0.091692455, Train acc 0.960, Val acc 0.550
Early stop: 51/250, best_acc: 0.667
Epoch 147, Time 0.462 s, Train loss 0.11979685, Train acc 0.981, Val acc 0.600
Early stop: 52/250, best_acc: 0.667
Epoch 148, Time 0.495 s, Train loss 0.107817955, Train acc 0.933, Val acc 0.617
Early stop: 53/250, best_acc: 0.667
Epoch 149, Time 0.496 s, Train loss 0.07426189, Train acc 0.971, Val acc 0.583
Early stop: 54/250, best_acc: 0.667
Epoch 150, Time 0.449 s, Train loss 0.05838417, Train acc 0.985, Val acc 0.567
Early stop: 55/250, best_acc: 0.667
Epoch 151, Time 0.492 s, Train loss 0.051390562, Train acc 0.985, Val acc 0.583
Early stop: 56/250, best_acc: 0.667
Epoch 152, Time 0.444 s, Train loss 0.048367947, Train acc 0.983, Val acc 0.617
Early stop: 57/250, best_acc: 0.667
Epoch 153, Time 0.390 s, Train loss 0.050382882, Train acc 0.988, Val acc 0.600
Early stop: 58/250, best_acc: 0.667
Epoch 154, Time 0.371 s, Train loss 0.04965042, Train acc 0.994, Val acc 0.633
Early stop: 59/250, best_acc: 0.667
Epoch 155, Time 0.472 s, Train loss 0.054516897, Train acc 0.985, Val acc 0.600
Early stop: 60/250, best_acc: 0.667
Epoch 156, Time 0.408 s, Train loss 0.050014555, Train acc 0.985, Val acc 0.617
Early stop: 61/250, best_acc: 0.667
Epoch 157, Time 0.445 s, Train loss 0.05796681, Train acc 0.950, Val acc 0.600
Early stop: 62/250, best_acc: 0.667
Epoch 158, Time 0.419 s, Train loss 0.08259619, Train acc 0.985, Val acc 0.617
Early stop: 63/250, best_acc: 0.667
Epoch 159, Time 0.495 s, Train loss 0.074711345, Train acc 0.969, Val acc 0.633
Early stop: 64/250, best_acc: 0.667
Epoch 160, Time 0.516 s, Train loss 0.057430387, Train acc 0.990, Val acc 0.600
Early stop: 65/250, best_acc: 0.667
Epoch 161, Time 0.467 s, Train loss 0.069561176, Train acc 0.963, Val acc 0.617
Early stop: 66/250, best_acc: 0.667
Epoch 162, Time 0.453 s, Train loss 0.0705011, Train acc 0.977, Val acc 0.583
Early stop: 67/250, best_acc: 0.667
Epoch 163, Time 0.417 s, Train loss 0.07167562, Train acc 0.985, Val acc 0.617
Early stop: 68/250, best_acc: 0.667
Epoch 164, Time 0.471 s, Train loss 0.07309025, Train acc 0.983, Val acc 0.633
Early stop: 69/250, best_acc: 0.667
Epoch 165, Time 0.398 s, Train loss 0.07655747, Train acc 0.973, Val acc 0.617
Early stop: 70/250, best_acc: 0.667
Epoch 166, Time 0.361 s, Train loss 0.08287124, Train acc 0.967, Val acc 0.650
Early stop: 71/250, best_acc: 0.667
Epoch 167, Time 0.455 s, Train loss 0.0943664, Train acc 0.990, Val acc 0.600
Early stop: 72/250, best_acc: 0.667
Epoch 168, Time 0.402 s, Train loss 0.08892169, Train acc 0.988, Val acc 0.617
Early stop: 73/250, best_acc: 0.667
Epoch 169, Time 0.420 s, Train loss 0.07717432, Train acc 0.985, Val acc 0.633
Early stop: 74/250, best_acc: 0.667
Epoch 170, Time 0.455 s, Train loss 0.07643742, Train acc 0.985, Val acc 0.633
Early stop: 75/250, best_acc: 0.667
Epoch 171, Time 0.469 s, Train loss 0.059742805, Train acc 0.988, Val acc 0.600
Early stop: 76/250, best_acc: 0.667
Epoch 172, Time 0.416 s, Train loss 0.062319826, Train acc 0.996, Val acc 0.633
Early stop: 77/250, best_acc: 0.667
Epoch 173, Time 0.482 s, Train loss 0.036422055, Train acc 0.994, Val acc 0.600
Early stop: 78/250, best_acc: 0.667
Epoch 174, Time 0.425 s, Train loss 0.037325233, Train acc 0.998, Val acc 0.650
Early stop: 79/250, best_acc: 0.667
Epoch 175, Time 0.389 s, Train loss 0.026700914, Train acc 0.998, Val acc 0.600
Early stop: 80/250, best_acc: 0.667
Epoch 176, Time 0.404 s, Train loss 0.02568832, Train acc 0.998, Val acc 0.700
Epoch 177, Time 0.460 s, Train loss 0.022029733, Train acc 0.998, Val acc 0.650
Early stop: 1/250, best_acc: 0.700
Epoch 178, Time 0.567 s, Train loss 0.020574419, Train acc 0.998, Val acc 0.667
Early stop: 2/250, best_acc: 0.700
Epoch 179, Time 1.590 s, Train loss 0.019690776, Train acc 0.998, Val acc 0.667
Early stop: 3/250, best_acc: 0.700
Epoch 180, Time 0.582 s, Train loss 0.01909261, Train acc 0.998, Val acc 0.683
Early stop: 4/250, best_acc: 0.700
Epoch 181, Time 0.498 s, Train loss 0.018634913, Train acc 0.998, Val acc 0.683
Early stop: 5/250, best_acc: 0.700
Epoch 182, Time 0.497 s, Train loss 0.018236376, Train acc 0.998, Val acc 0.683
Early stop: 6/250, best_acc: 0.700
Epoch 183, Time 0.484 s, Train loss 0.017710188, Train acc 0.998, Val acc 0.667
Early stop: 7/250, best_acc: 0.700
Epoch 184, Time 0.430 s, Train loss 0.017367357, Train acc 0.998, Val acc 0.667
Early stop: 8/250, best_acc: 0.700
Epoch 185, Time 0.505 s, Train loss 0.016884927, Train acc 0.998, Val acc 0.667
Early stop: 9/250, best_acc: 0.700
Epoch 186, Time 0.451 s, Train loss 0.0164474, Train acc 0.998, Val acc 0.683
Early stop: 10/250, best_acc: 0.700
Epoch 187, Time 0.443 s, Train loss 0.016078373, Train acc 0.998, Val acc 0.667
Early stop: 11/250, best_acc: 0.700
Epoch 188, Time 0.362 s, Train loss 0.015776258, Train acc 0.998, Val acc 0.667
Early stop: 12/250, best_acc: 0.700
Epoch 189, Time 0.510 s, Train loss 0.01552521, Train acc 0.998, Val acc 0.667
Early stop: 13/250, best_acc: 0.700
Epoch 190, Time 0.447 s, Train loss 0.015195705, Train acc 0.998, Val acc 0.667
Early stop: 14/250, best_acc: 0.700
Epoch 191, Time 0.437 s, Train loss 0.014874886, Train acc 0.998, Val acc 0.667
Early stop: 15/250, best_acc: 0.700
Epoch 192, Time 0.442 s, Train loss 0.014555245, Train acc 0.998, Val acc 0.667
Early stop: 16/250, best_acc: 0.700
Epoch 193, Time 0.476 s, Train loss 0.014296618, Train acc 0.998, Val acc 0.667
Early stop: 17/250, best_acc: 0.700
Epoch 194, Time 0.449 s, Train loss 0.013959006, Train acc 0.998, Val acc 0.667
Early stop: 18/250, best_acc: 0.700
Epoch 195, Time 0.480 s, Train loss 0.013738096, Train acc 0.998, Val acc 0.667
Early stop: 19/250, best_acc: 0.700
Epoch 196, Time 0.356 s, Train loss 0.013432686, Train acc 0.998, Val acc 0.667
Early stop: 20/250, best_acc: 0.700
Epoch 197, Time 0.562 s, Train loss 0.01319401, Train acc 0.998, Val acc 0.667
Early stop: 21/250, best_acc: 0.700
Epoch 198, Time 0.423 s, Train loss 0.012941019, Train acc 0.998, Val acc 0.667
Early stop: 22/250, best_acc: 0.700
Epoch 199, Time 0.448 s, Train loss 0.01277091, Train acc 0.998, Val acc 0.667
Early stop: 23/250, best_acc: 0.700
Epoch 200, Time 0.400 s, Train loss 0.012498942, Train acc 0.998, Val acc 0.667
Early stop: 24/250, best_acc: 0.700
Epoch 201, Time 0.465 s, Train loss 0.0122056, Train acc 0.998, Val acc 0.667
Early stop: 25/250, best_acc: 0.700
Epoch 202, Time 0.425 s, Train loss 0.0120049035, Train acc 0.998, Val acc 0.667
Early stop: 26/250, best_acc: 0.700
Epoch 203, Time 0.438 s, Train loss 0.011732702, Train acc 1.000, Val acc 0.667
Early stop: 27/250, best_acc: 0.700
Epoch 204, Time 0.453 s, Train loss 0.011497598, Train acc 1.000, Val acc 0.667
Early stop: 28/250, best_acc: 0.700
Epoch 205, Time 0.554 s, Train loss 0.011238384, Train acc 1.000, Val acc 0.667
Early stop: 29/250, best_acc: 0.700
Epoch 206, Time 0.549 s, Train loss 0.011011478, Train acc 1.000, Val acc 0.667
Early stop: 30/250, best_acc: 0.700
Epoch 207, Time 0.490 s, Train loss 0.010795026, Train acc 1.000, Val acc 0.667
Early stop: 31/250, best_acc: 0.700
Epoch 208, Time 0.473 s, Train loss 0.010626622, Train acc 1.000, Val acc 0.667
Early stop: 32/250, best_acc: 0.700
Epoch 209, Time 0.391 s, Train loss 0.0104058115, Train acc 1.000, Val acc 0.667
Early stop: 33/250, best_acc: 0.700
Epoch 210, Time 0.479 s, Train loss 0.01022911, Train acc 1.000, Val acc 0.667
Early stop: 34/250, best_acc: 0.700
Epoch 211, Time 0.429 s, Train loss 0.010020177, Train acc 1.000, Val acc 0.667
Early stop: 35/250, best_acc: 0.700
Epoch 212, Time 0.473 s, Train loss 0.009809411, Train acc 1.000, Val acc 0.667
Early stop: 36/250, best_acc: 0.700
Epoch 213, Time 0.469 s, Train loss 0.009606905, Train acc 1.000, Val acc 0.667
Early stop: 37/250, best_acc: 0.700
Epoch 214, Time 0.435 s, Train loss 0.009448944, Train acc 1.000, Val acc 0.667
Early stop: 38/250, best_acc: 0.700
Epoch 215, Time 0.476 s, Train loss 0.009237081, Train acc 1.000, Val acc 0.667
Early stop: 39/250, best_acc: 0.700
Epoch 216, Time 0.573 s, Train loss 0.009012331, Train acc 1.000, Val acc 0.667
Early stop: 40/250, best_acc: 0.700
Epoch 217, Time 0.498 s, Train loss 0.008849256, Train acc 1.000, Val acc 0.667
Early stop: 41/250, best_acc: 0.700
Epoch 218, Time 0.461 s, Train loss 0.008680132, Train acc 1.000, Val acc 0.650
Early stop: 42/250, best_acc: 0.700
Epoch 219, Time 0.488 s, Train loss 0.008516143, Train acc 1.000, Val acc 0.667
Early stop: 43/250, best_acc: 0.700
Epoch 220, Time 0.432 s, Train loss 0.008344542, Train acc 1.000, Val acc 0.667
Early stop: 44/250, best_acc: 0.700
Epoch 221, Time 0.457 s, Train loss 0.008182772, Train acc 1.000, Val acc 0.650
Early stop: 45/250, best_acc: 0.700
Epoch 222, Time 0.455 s, Train loss 0.008036084, Train acc 1.000, Val acc 0.667
Early stop: 46/250, best_acc: 0.700
Epoch 223, Time 0.484 s, Train loss 0.007829191, Train acc 1.000, Val acc 0.667
Early stop: 47/250, best_acc: 0.700
Epoch 224, Time 0.401 s, Train loss 0.007711843, Train acc 1.000, Val acc 0.667
Early stop: 48/250, best_acc: 0.700
Epoch 225, Time 0.527 s, Train loss 0.0075601432, Train acc 1.000, Val acc 0.667
Early stop: 49/250, best_acc: 0.700
Epoch 226, Time 0.408 s, Train loss 0.0074124634, Train acc 1.000, Val acc 0.667
Early stop: 50/250, best_acc: 0.700
Epoch 227, Time 0.415 s, Train loss 0.0072396244, Train acc 1.000, Val acc 0.650
Early stop: 51/250, best_acc: 0.700
Epoch 228, Time 0.490 s, Train loss 0.007126571, Train acc 1.000, Val acc 0.650
Early stop: 52/250, best_acc: 0.700
Epoch 229, Time 0.533 s, Train loss 0.006983183, Train acc 1.000, Val acc 0.650
Early stop: 53/250, best_acc: 0.700
Epoch 230, Time 0.506 s, Train loss 0.006865317, Train acc 1.000, Val acc 0.650
Early stop: 54/250, best_acc: 0.700
Epoch 231, Time 0.412 s, Train loss 0.0067660254, Train acc 1.000, Val acc 0.650
Early stop: 55/250, best_acc: 0.700
Epoch 232, Time 0.479 s, Train loss 0.006623872, Train acc 1.000, Val acc 0.650
Early stop: 56/250, best_acc: 0.700
Epoch 233, Time 0.498 s, Train loss 0.0065041757, Train acc 1.000, Val acc 0.650
Early stop: 57/250, best_acc: 0.700
Epoch 234, Time 0.433 s, Train loss 0.006386595, Train acc 1.000, Val acc 0.650
Early stop: 58/250, best_acc: 0.700
Epoch 235, Time 0.419 s, Train loss 0.00629307, Train acc 1.000, Val acc 0.650
Early stop: 59/250, best_acc: 0.700
Epoch 236, Time 0.371 s, Train loss 0.0061713383, Train acc 1.000, Val acc 0.650
Early stop: 60/250, best_acc: 0.700
Epoch 237, Time 0.453 s, Train loss 0.0060359705, Train acc 1.000, Val acc 0.650
Early stop: 61/250, best_acc: 0.700
Epoch 238, Time 0.479 s, Train loss 0.005904325, Train acc 1.000, Val acc 0.650
Early stop: 62/250, best_acc: 0.700
Epoch 239, Time 0.462 s, Train loss 0.0058056973, Train acc 1.000, Val acc 0.650
Early stop: 63/250, best_acc: 0.700
Epoch 240, Time 0.418 s, Train loss 0.005707497, Train acc 1.000, Val acc 0.650
Early stop: 64/250, best_acc: 0.700
Epoch 241, Time 0.382 s, Train loss 0.0056135566, Train acc 1.000, Val acc 0.650
Early stop: 65/250, best_acc: 0.700
Epoch 242, Time 0.368 s, Train loss 0.0055194255, Train acc 1.000, Val acc 0.650
Early stop: 66/250, best_acc: 0.700
Epoch 243, Time 0.460 s, Train loss 0.005417276, Train acc 1.000, Val acc 0.650
Early stop: 67/250, best_acc: 0.700
Epoch 244, Time 0.541 s, Train loss 0.0053373124, Train acc 1.000, Val acc 0.650
Early stop: 68/250, best_acc: 0.700
Epoch 245, Time 0.555 s, Train loss 0.005261449, Train acc 1.000, Val acc 0.650
Early stop: 69/250, best_acc: 0.700
Epoch 246, Time 0.491 s, Train loss 0.0051706643, Train acc 1.000, Val acc 0.650
Early stop: 70/250, best_acc: 0.700
Epoch 247, Time 0.500 s, Train loss 0.0050909277, Train acc 1.000, Val acc 0.650
Early stop: 71/250, best_acc: 0.700
Epoch 248, Time 0.398 s, Train loss 0.005022087, Train acc 1.000, Val acc 0.650
Early stop: 72/250, best_acc: 0.700
Epoch 249, Time 0.506 s, Train loss 0.0049565127, Train acc 1.000, Val acc 0.650
Early stop: 73/250, best_acc: 0.700
Epoch 250, Time 0.414 s, Train loss 0.0048789834, Train acc 1.000, Val acc 0.650
Early stop: 74/250, best_acc: 0.700
Epoch 251, Time 0.515 s, Train loss 0.004812721, Train acc 1.000, Val acc 0.650
Early stop: 75/250, best_acc: 0.700
Epoch 252, Time 0.474 s, Train loss 0.0047224537, Train acc 1.000, Val acc 0.650
Early stop: 76/250, best_acc: 0.700
Epoch 253, Time 0.521 s, Train loss 0.004660266, Train acc 1.000, Val acc 0.650
Early stop: 77/250, best_acc: 0.700
Epoch 254, Time 0.467 s, Train loss 0.0045897793, Train acc 1.000, Val acc 0.650
Early stop: 78/250, best_acc: 0.700
Epoch 255, Time 0.544 s, Train loss 0.004527989, Train acc 1.000, Val acc 0.650
Early stop: 79/250, best_acc: 0.700
Epoch 256, Time 0.415 s, Train loss 0.0044601373, Train acc 1.000, Val acc 0.650
Early stop: 80/250, best_acc: 0.700
Epoch 257, Time 0.437 s, Train loss 0.004400994, Train acc 1.000, Val acc 0.650
Early stop: 81/250, best_acc: 0.700
Epoch 258, Time 0.459 s, Train loss 0.0043371706, Train acc 1.000, Val acc 0.650
Early stop: 82/250, best_acc: 0.700
Epoch 259, Time 0.394 s, Train loss 0.004285587, Train acc 1.000, Val acc 0.650
Early stop: 83/250, best_acc: 0.700
Epoch 260, Time 0.390 s, Train loss 0.0042182603, Train acc 1.000, Val acc 0.650
Early stop: 84/250, best_acc: 0.700
Epoch 261, Time 0.452 s, Train loss 0.0041620624, Train acc 1.000, Val acc 0.650
Early stop: 85/250, best_acc: 0.700
Epoch 262, Time 0.434 s, Train loss 0.00409699, Train acc 1.000, Val acc 0.650
Early stop: 86/250, best_acc: 0.700
Epoch 263, Time 0.397 s, Train loss 0.004046612, Train acc 1.000, Val acc 0.650
Early stop: 87/250, best_acc: 0.700
Epoch 264, Time 0.402 s, Train loss 0.0039870483, Train acc 1.000, Val acc 0.650
Early stop: 88/250, best_acc: 0.700
Epoch 265, Time 0.470 s, Train loss 0.0039341473, Train acc 1.000, Val acc 0.650
Early stop: 89/250, best_acc: 0.700
Epoch 266, Time 0.468 s, Train loss 0.0038856594, Train acc 1.000, Val acc 0.650
Early stop: 90/250, best_acc: 0.700
Epoch 267, Time 0.436 s, Train loss 0.003829796, Train acc 1.000, Val acc 0.650
Early stop: 91/250, best_acc: 0.700
Epoch 268, Time 0.429 s, Train loss 0.0037954154, Train acc 1.000, Val acc 0.650
Early stop: 92/250, best_acc: 0.700
Epoch 269, Time 0.490 s, Train loss 0.0037262824, Train acc 1.000, Val acc 0.650
Early stop: 93/250, best_acc: 0.700
Epoch 270, Time 0.434 s, Train loss 0.0036794783, Train acc 1.000, Val acc 0.650
Early stop: 94/250, best_acc: 0.700
Epoch 271, Time 0.486 s, Train loss 0.0036213314, Train acc 1.000, Val acc 0.650
Early stop: 95/250, best_acc: 0.700
Epoch 272, Time 0.504 s, Train loss 0.0035795223, Train acc 1.000, Val acc 0.650
Early stop: 96/250, best_acc: 0.700
Epoch 273, Time 0.519 s, Train loss 0.0035248247, Train acc 1.000, Val acc 0.650
Early stop: 97/250, best_acc: 0.700
Epoch 274, Time 0.416 s, Train loss 0.0034876112, Train acc 1.000, Val acc 0.650
Early stop: 98/250, best_acc: 0.700
Epoch 275, Time 0.399 s, Train loss 0.0034317093, Train acc 1.000, Val acc 0.650
Early stop: 99/250, best_acc: 0.700
Epoch 276, Time 0.518 s, Train loss 0.0033929376, Train acc 1.000, Val acc 0.650
Early stop: 100/250, best_acc: 0.700
Epoch 277, Time 0.573 s, Train loss 0.0033423684, Train acc 1.000, Val acc 0.650
Early stop: 101/250, best_acc: 0.700
Epoch 278, Time 0.519 s, Train loss 0.003301292, Train acc 1.000, Val acc 0.650
Early stop: 102/250, best_acc: 0.700
Epoch 279, Time 0.484 s, Train loss 0.003261026, Train acc 1.000, Val acc 0.650
Early stop: 103/250, best_acc: 0.700
Epoch 280, Time 0.460 s, Train loss 0.003216929, Train acc 1.000, Val acc 0.650
Early stop: 104/250, best_acc: 0.700
Epoch 281, Time 0.461 s, Train loss 0.0031778135, Train acc 1.000, Val acc 0.650
Early stop: 105/250, best_acc: 0.700
Epoch 282, Time 0.475 s, Train loss 0.0031328704, Train acc 1.000, Val acc 0.650
Early stop: 106/250, best_acc: 0.700
Epoch 283, Time 0.422 s, Train loss 0.0030928731, Train acc 1.000, Val acc 0.650
Early stop: 107/250, best_acc: 0.700
Epoch 284, Time 0.474 s, Train loss 0.003052411, Train acc 1.000, Val acc 0.650
Early stop: 108/250, best_acc: 0.700
Epoch 285, Time 0.490 s, Train loss 0.0030130472, Train acc 1.000, Val acc 0.650
Early stop: 109/250, best_acc: 0.700
Epoch 286, Time 0.406 s, Train loss 0.0029879622, Train acc 1.000, Val acc 0.650
Early stop: 110/250, best_acc: 0.700
Epoch 287, Time 0.463 s, Train loss 0.002943135, Train acc 1.000, Val acc 0.650
Early stop: 111/250, best_acc: 0.700
Epoch 288, Time 0.418 s, Train loss 0.0029066373, Train acc 1.000, Val acc 0.650
Early stop: 112/250, best_acc: 0.700
Epoch 289, Time 0.560 s, Train loss 0.0028646514, Train acc 1.000, Val acc 0.650
Early stop: 113/250, best_acc: 0.700
Epoch 290, Time 0.524 s, Train loss 0.00283499, Train acc 1.000, Val acc 0.650
Early stop: 114/250, best_acc: 0.700
Epoch 291, Time 0.519 s, Train loss 0.0027942366, Train acc 1.000, Val acc 0.650
Early stop: 115/250, best_acc: 0.700
Epoch 292, Time 0.495 s, Train loss 0.0027622913, Train acc 1.000, Val acc 0.650
Early stop: 116/250, best_acc: 0.700
Epoch 293, Time 0.413 s, Train loss 0.0027235637, Train acc 1.000, Val acc 0.650
Early stop: 117/250, best_acc: 0.700
Epoch 294, Time 0.518 s, Train loss 0.002692718, Train acc 1.000, Val acc 0.650
Early stop: 118/250, best_acc: 0.700
Epoch 295, Time 0.413 s, Train loss 0.0026575185, Train acc 1.000, Val acc 0.650
Early stop: 119/250, best_acc: 0.700
Epoch 296, Time 0.509 s, Train loss 0.0026240489, Train acc 1.000, Val acc 0.650
Early stop: 120/250, best_acc: 0.700
Epoch 297, Time 0.528 s, Train loss 0.002592683, Train acc 1.000, Val acc 0.650
Early stop: 121/250, best_acc: 0.700
Epoch 298, Time 0.501 s, Train loss 0.0025590973, Train acc 1.000, Val acc 0.650
Early stop: 122/250, best_acc: 0.700
Epoch 299, Time 0.459 s, Train loss 0.0025302249, Train acc 1.000, Val acc 0.650
Early stop: 123/250, best_acc: 0.700
Epoch 300, Time 0.460 s, Train loss 0.0024978924, Train acc 1.000, Val acc 0.650
Early stop: 124/250, best_acc: 0.700
Epoch 301, Time 0.476 s, Train loss 0.0024676884, Train acc 1.000, Val acc 0.650
Early stop: 125/250, best_acc: 0.700
Epoch 302, Time 0.428 s, Train loss 0.002437773, Train acc 1.000, Val acc 0.650
Early stop: 126/250, best_acc: 0.700
Epoch 303, Time 0.432 s, Train loss 0.002407946, Train acc 1.000, Val acc 0.650
Early stop: 127/250, best_acc: 0.700
Epoch 304, Time 0.500 s, Train loss 0.0023790116, Train acc 1.000, Val acc 0.650
Early stop: 128/250, best_acc: 0.700
Epoch 305, Time 0.507 s, Train loss 0.0023505527, Train acc 1.000, Val acc 0.650
Early stop: 129/250, best_acc: 0.700
Epoch 306, Time 0.517 s, Train loss 0.0023214521, Train acc 1.000, Val acc 0.650
Early stop: 130/250, best_acc: 0.700
Epoch 307, Time 0.544 s, Train loss 0.0022939832, Train acc 1.000, Val acc 0.650
Early stop: 131/250, best_acc: 0.700
Epoch 308, Time 0.529 s, Train loss 0.0022661656, Train acc 1.000, Val acc 0.650
Early stop: 132/250, best_acc: 0.700
Epoch 309, Time 0.510 s, Train loss 0.0022384946, Train acc 1.000, Val acc 0.650
Early stop: 133/250, best_acc: 0.700
Epoch 310, Time 0.426 s, Train loss 0.0022119582, Train acc 1.000, Val acc 0.650
Early stop: 134/250, best_acc: 0.700
Epoch 311, Time 0.429 s, Train loss 0.002186575, Train acc 1.000, Val acc 0.650
Early stop: 135/250, best_acc: 0.700
Epoch 312, Time 0.477 s, Train loss 0.0021592162, Train acc 1.000, Val acc 0.650
Early stop: 136/250, best_acc: 0.700
Epoch 313, Time 0.513 s, Train loss 0.002132119, Train acc 1.000, Val acc 0.650
Early stop: 137/250, best_acc: 0.700
Epoch 314, Time 0.476 s, Train loss 0.0021065245, Train acc 1.000, Val acc 0.650
Early stop: 138/250, best_acc: 0.700
Epoch 315, Time 0.439 s, Train loss 0.0020811583, Train acc 1.000, Val acc 0.650
Early stop: 139/250, best_acc: 0.700
Epoch 316, Time 0.412 s, Train loss 0.0020567281, Train acc 1.000, Val acc 0.650
Early stop: 140/250, best_acc: 0.700
Epoch 317, Time 0.459 s, Train loss 0.002030554, Train acc 1.000, Val acc 0.650
Early stop: 141/250, best_acc: 0.700
Epoch 318, Time 0.496 s, Train loss 0.0020065028, Train acc 1.000, Val acc 0.650
Early stop: 142/250, best_acc: 0.700
Epoch 319, Time 0.544 s, Train loss 0.0019822156, Train acc 1.000, Val acc 0.650
Early stop: 143/250, best_acc: 0.700
Epoch 320, Time 0.504 s, Train loss 0.001959175, Train acc 1.000, Val acc 0.650
Early stop: 144/250, best_acc: 0.700
Epoch 321, Time 0.512 s, Train loss 0.0019363916, Train acc 1.000, Val acc 0.650
Early stop: 145/250, best_acc: 0.700
Epoch 322, Time 0.402 s, Train loss 0.0019132663, Train acc 1.000, Val acc 0.650
Early stop: 146/250, best_acc: 0.700
Epoch 323, Time 0.495 s, Train loss 0.0018907436, Train acc 1.000, Val acc 0.650
Early stop: 147/250, best_acc: 0.700
Epoch 324, Time 0.386 s, Train loss 0.0018701704, Train acc 1.000, Val acc 0.650
Early stop: 148/250, best_acc: 0.700
Epoch 325, Time 0.444 s, Train loss 0.0018470414, Train acc 1.000, Val acc 0.650
Early stop: 149/250, best_acc: 0.700
Epoch 326, Time 0.543 s, Train loss 0.0018262207, Train acc 1.000, Val acc 0.650
Early stop: 150/250, best_acc: 0.700
Epoch 327, Time 0.456 s, Train loss 0.0018042196, Train acc 1.000, Val acc 0.650
Early stop: 151/250, best_acc: 0.700
Epoch 328, Time 0.494 s, Train loss 0.001783349, Train acc 1.000, Val acc 0.650
Early stop: 152/250, best_acc: 0.700
Epoch 329, Time 0.504 s, Train loss 0.0017633556, Train acc 1.000, Val acc 0.650
Early stop: 153/250, best_acc: 0.700
Epoch 330, Time 0.481 s, Train loss 0.0017425042, Train acc 1.000, Val acc 0.650
Early stop: 154/250, best_acc: 0.700
Epoch 331, Time 0.481 s, Train loss 0.0017238525, Train acc 1.000, Val acc 0.650
Early stop: 155/250, best_acc: 0.700
Epoch 332, Time 0.428 s, Train loss 0.001703361, Train acc 1.000, Val acc 0.650
Early stop: 156/250, best_acc: 0.700
Epoch 333, Time 0.517 s, Train loss 0.0016837387, Train acc 1.000, Val acc 0.650
Early stop: 157/250, best_acc: 0.700
Epoch 334, Time 0.434 s, Train loss 0.0016640314, Train acc 1.000, Val acc 0.650
Early stop: 158/250, best_acc: 0.700
Epoch 335, Time 0.447 s, Train loss 0.0016452976, Train acc 1.000, Val acc 0.650
Early stop: 159/250, best_acc: 0.700
Epoch 336, Time 0.493 s, Train loss 0.0016260847, Train acc 1.000, Val acc 0.650
Early stop: 160/250, best_acc: 0.700
Epoch 337, Time 0.398 s, Train loss 0.0016082326, Train acc 1.000, Val acc 0.650
Early stop: 161/250, best_acc: 0.700
Epoch 338, Time 0.468 s, Train loss 0.0015899701, Train acc 1.000, Val acc 0.650
Early stop: 162/250, best_acc: 0.700
Epoch 339, Time 0.488 s, Train loss 0.0015728011, Train acc 1.000, Val acc 0.650
Early stop: 163/250, best_acc: 0.700
Epoch 340, Time 0.395 s, Train loss 0.0015551735, Train acc 1.000, Val acc 0.650
Early stop: 164/250, best_acc: 0.700
Epoch 341, Time 0.460 s, Train loss 0.0015367278, Train acc 1.000, Val acc 0.650
Early stop: 165/250, best_acc: 0.700
Epoch 342, Time 0.480 s, Train loss 0.0015196254, Train acc 1.000, Val acc 0.650
Early stop: 166/250, best_acc: 0.700
Epoch 343, Time 0.451 s, Train loss 0.0015028259, Train acc 1.000, Val acc 0.650
Early stop: 167/250, best_acc: 0.700
Epoch 344, Time 0.412 s, Train loss 0.0014859053, Train acc 1.000, Val acc 0.650
Early stop: 168/250, best_acc: 0.700
Epoch 345, Time 0.510 s, Train loss 0.0014698872, Train acc 1.000, Val acc 0.650
Early stop: 169/250, best_acc: 0.700
Epoch 346, Time 0.531 s, Train loss 0.0014525208, Train acc 1.000, Val acc 0.650
Early stop: 170/250, best_acc: 0.700
Epoch 347, Time 0.444 s, Train loss 0.0014354134, Train acc 1.000, Val acc 0.650
Early stop: 171/250, best_acc: 0.700
Epoch 348, Time 0.474 s, Train loss 0.0014210086, Train acc 1.000, Val acc 0.650
Early stop: 172/250, best_acc: 0.700
Epoch 349, Time 0.434 s, Train loss 0.0014044397, Train acc 1.000, Val acc 0.650
Early stop: 173/250, best_acc: 0.700
Epoch 350, Time 0.523 s, Train loss 0.0013889294, Train acc 1.000, Val acc 0.650
Early stop: 174/250, best_acc: 0.700
Epoch 351, Time 0.477 s, Train loss 0.0013734986, Train acc 1.000, Val acc 0.650
Early stop: 175/250, best_acc: 0.700
Epoch 352, Time 0.387 s, Train loss 0.0013576957, Train acc 1.000, Val acc 0.650
Early stop: 176/250, best_acc: 0.700
Epoch 353, Time 0.502 s, Train loss 0.0013429964, Train acc 1.000, Val acc 0.650
Early stop: 177/250, best_acc: 0.700
Epoch 354, Time 0.525 s, Train loss 0.0013304517, Train acc 1.000, Val acc 0.650
Early stop: 178/250, best_acc: 0.700
Epoch 355, Time 0.577 s, Train loss 0.0013134499, Train acc 1.000, Val acc 0.650
Early stop: 179/250, best_acc: 0.700
Epoch 356, Time 0.443 s, Train loss 0.0012987355, Train acc 1.000, Val acc 0.650
Early stop: 180/250, best_acc: 0.700
Epoch 357, Time 0.450 s, Train loss 0.0012830388, Train acc 1.000, Val acc 0.650
Early stop: 181/250, best_acc: 0.700
Epoch 358, Time 0.541 s, Train loss 0.0012687013, Train acc 1.000, Val acc 0.650
Early stop: 182/250, best_acc: 0.700
Epoch 359, Time 0.522 s, Train loss 0.0012549628, Train acc 1.000, Val acc 0.650
Early stop: 183/250, best_acc: 0.700
Epoch 360, Time 0.454 s, Train loss 0.0012409346, Train acc 1.000, Val acc 0.650
Early stop: 184/250, best_acc: 0.700
Epoch 361, Time 0.466 s, Train loss 0.0012269075, Train acc 1.000, Val acc 0.650
Early stop: 185/250, best_acc: 0.700
Epoch 362, Time 0.485 s, Train loss 0.0012140591, Train acc 1.000, Val acc 0.650
Early stop: 186/250, best_acc: 0.700
Epoch 363, Time 0.403 s, Train loss 0.001200567, Train acc 1.000, Val acc 0.650
Early stop: 187/250, best_acc: 0.700
Epoch 364, Time 0.479 s, Train loss 0.0011875076, Train acc 1.000, Val acc 0.650
Early stop: 188/250, best_acc: 0.700
Epoch 365, Time 0.473 s, Train loss 0.0011746249, Train acc 1.000, Val acc 0.650
Early stop: 189/250, best_acc: 0.700
Epoch 366, Time 0.437 s, Train loss 0.0011618413, Train acc 1.000, Val acc 0.650
Early stop: 190/250, best_acc: 0.700
Epoch 367, Time 0.400 s, Train loss 0.001149184, Train acc 1.000, Val acc 0.650
Early stop: 191/250, best_acc: 0.700
Epoch 368, Time 0.649 s, Train loss 0.0011360978, Train acc 1.000, Val acc 0.650
Early stop: 192/250, best_acc: 0.700
Epoch 369, Time 0.507 s, Train loss 0.0011240572, Train acc 1.000, Val acc 0.650
Early stop: 193/250, best_acc: 0.700
Epoch 370, Time 0.468 s, Train loss 0.0011120125, Train acc 1.000, Val acc 0.650
Early stop: 194/250, best_acc: 0.700
Epoch 371, Time 0.455 s, Train loss 0.0010997426, Train acc 1.000, Val acc 0.650
Early stop: 195/250, best_acc: 0.700
Epoch 372, Time 0.476 s, Train loss 0.0010877469, Train acc 1.000, Val acc 0.650
Early stop: 196/250, best_acc: 0.700
Epoch 373, Time 0.428 s, Train loss 0.0010764294, Train acc 1.000, Val acc 0.650
Early stop: 197/250, best_acc: 0.700
Epoch 374, Time 0.441 s, Train loss 0.0010642502, Train acc 1.000, Val acc 0.650
Early stop: 198/250, best_acc: 0.700
Epoch 375, Time 0.396 s, Train loss 0.0010531273, Train acc 1.000, Val acc 0.650
Early stop: 199/250, best_acc: 0.700
Epoch 376, Time 0.505 s, Train loss 0.0010419948, Train acc 1.000, Val acc 0.650
Early stop: 200/250, best_acc: 0.700
Epoch 377, Time 0.483 s, Train loss 0.0010305009, Train acc 1.000, Val acc 0.650
Early stop: 201/250, best_acc: 0.700
Epoch 378, Time 0.476 s, Train loss 0.0010194418, Train acc 1.000, Val acc 0.650
Early stop: 202/250, best_acc: 0.700
Epoch 379, Time 0.455 s, Train loss 0.0010090617, Train acc 1.000, Val acc 0.650
Early stop: 203/250, best_acc: 0.700
Epoch 380, Time 0.442 s, Train loss 0.0009978554, Train acc 1.000, Val acc 0.650
Early stop: 204/250, best_acc: 0.700
Epoch 381, Time 0.399 s, Train loss 0.0009873973, Train acc 1.000, Val acc 0.650
Early stop: 205/250, best_acc: 0.700
Epoch 382, Time 0.479 s, Train loss 0.0009766555, Train acc 1.000, Val acc 0.650
Early stop: 206/250, best_acc: 0.700
Epoch 383, Time 0.490 s, Train loss 0.0009684251, Train acc 1.000, Val acc 0.650
Early stop: 207/250, best_acc: 0.700
Epoch 384, Time 0.455 s, Train loss 0.00095539325, Train acc 1.000, Val acc 0.650
Early stop: 208/250, best_acc: 0.700
Epoch 385, Time 0.472 s, Train loss 0.0009443324, Train acc 1.000, Val acc 0.650
Early stop: 209/250, best_acc: 0.700
Epoch 386, Time 0.437 s, Train loss 0.0009340558, Train acc 1.000, Val acc 0.650
Early stop: 210/250, best_acc: 0.700
Epoch 387, Time 0.445 s, Train loss 0.0009240494, Train acc 1.000, Val acc 0.650
Early stop: 211/250, best_acc: 0.700
Epoch 388, Time 0.471 s, Train loss 0.00091401, Train acc 1.000, Val acc 0.650
Early stop: 212/250, best_acc: 0.700
Epoch 389, Time 0.466 s, Train loss 0.00090472004, Train acc 1.000, Val acc 0.650
Early stop: 213/250, best_acc: 0.700
Epoch 390, Time 0.546 s, Train loss 0.00089628267, Train acc 1.000, Val acc 0.650
Early stop: 214/250, best_acc: 0.700
Epoch 391, Time 0.468 s, Train loss 0.00088586594, Train acc 1.000, Val acc 0.650
Early stop: 215/250, best_acc: 0.700
Epoch 392, Time 0.396 s, Train loss 0.00087670266, Train acc 1.000, Val acc 0.650
Early stop: 216/250, best_acc: 0.700
Epoch 393, Time 0.489 s, Train loss 0.00086664903, Train acc 1.000, Val acc 0.650
Early stop: 217/250, best_acc: 0.700
Epoch 394, Time 0.527 s, Train loss 0.0008576791, Train acc 1.000, Val acc 0.650
Early stop: 218/250, best_acc: 0.700
Epoch 395, Time 0.479 s, Train loss 0.0008484356, Train acc 1.000, Val acc 0.650
Early stop: 219/250, best_acc: 0.700
Epoch 396, Time 0.453 s, Train loss 0.0008396956, Train acc 1.000, Val acc 0.650
Early stop: 220/250, best_acc: 0.700
Epoch 397, Time 0.517 s, Train loss 0.00083096203, Train acc 1.000, Val acc 0.650
Early stop: 221/250, best_acc: 0.700
Epoch 398, Time 0.537 s, Train loss 0.00082240364, Train acc 1.000, Val acc 0.650
Early stop: 222/250, best_acc: 0.700
Epoch 399, Time 0.475 s, Train loss 0.0008137794, Train acc 1.000, Val acc 0.650
Early stop: 223/250, best_acc: 0.700
Epoch 400, Time 0.493 s, Train loss 0.00080539467, Train acc 1.000, Val acc 0.650
Early stop: 224/250, best_acc: 0.700
Epoch 401, Time 0.576 s, Train loss 0.00079721137, Train acc 1.000, Val acc 0.650
Early stop: 225/250, best_acc: 0.700
Epoch 402, Time 0.536 s, Train loss 0.00078945054, Train acc 1.000, Val acc 0.650
Early stop: 226/250, best_acc: 0.700
Epoch 403, Time 0.507 s, Train loss 0.00078088517, Train acc 1.000, Val acc 0.650
Early stop: 227/250, best_acc: 0.700
Epoch 404, Time 0.497 s, Train loss 0.00077294017, Train acc 1.000, Val acc 0.650
Early stop: 228/250, best_acc: 0.700
Epoch 405, Time 0.442 s, Train loss 0.00076504186, Train acc 1.000, Val acc 0.650
Early stop: 229/250, best_acc: 0.700
Epoch 406, Time 0.437 s, Train loss 0.000757363, Train acc 1.000, Val acc 0.650
Early stop: 230/250, best_acc: 0.700
Epoch 407, Time 0.461 s, Train loss 0.0007495063, Train acc 1.000, Val acc 0.650
Early stop: 231/250, best_acc: 0.700
Epoch 408, Time 0.439 s, Train loss 0.0007421226, Train acc 1.000, Val acc 0.650
Early stop: 232/250, best_acc: 0.700
Epoch 409, Time 0.488 s, Train loss 0.00073483685, Train acc 1.000, Val acc 0.650
Early stop: 233/250, best_acc: 0.700
Epoch 410, Time 0.453 s, Train loss 0.00072724116, Train acc 1.000, Val acc 0.650
Early stop: 234/250, best_acc: 0.700
Epoch 411, Time 0.440 s, Train loss 0.0007201417, Train acc 1.000, Val acc 0.650
Early stop: 235/250, best_acc: 0.700
Epoch 412, Time 0.521 s, Train loss 0.0007130946, Train acc 1.000, Val acc 0.650
Early stop: 236/250, best_acc: 0.700
Epoch 413, Time 0.534 s, Train loss 0.0007057858, Train acc 1.000, Val acc 0.650
Early stop: 237/250, best_acc: 0.700
Epoch 414, Time 0.581 s, Train loss 0.00069864193, Train acc 1.000, Val acc 0.650
Early stop: 238/250, best_acc: 0.700
Epoch 415, Time 0.553 s, Train loss 0.0006914244, Train acc 1.000, Val acc 0.650
Early stop: 239/250, best_acc: 0.700
Epoch 416, Time 0.522 s, Train loss 0.0006849554, Train acc 1.000, Val acc 0.650
Early stop: 240/250, best_acc: 0.700
Epoch 417, Time 0.449 s, Train loss 0.00067824824, Train acc 1.000, Val acc 0.650
Early stop: 241/250, best_acc: 0.700
Epoch 418, Time 0.519 s, Train loss 0.0006718963, Train acc 1.000, Val acc 0.650
Early stop: 242/250, best_acc: 0.700
Epoch 419, Time 0.576 s, Train loss 0.00066471513, Train acc 1.000, Val acc 0.650
Early stop: 243/250, best_acc: 0.700
Epoch 420, Time 0.583 s, Train loss 0.0006578341, Train acc 1.000, Val acc 0.650
Early stop: 244/250, best_acc: 0.700
Epoch 421, Time 0.483 s, Train loss 0.0006515351, Train acc 1.000, Val acc 0.650
Early stop: 245/250, best_acc: 0.700
Epoch 422, Time 0.470 s, Train loss 0.0006447732, Train acc 1.000, Val acc 0.650
Early stop: 246/250, best_acc: 0.700
Epoch 423, Time 0.475 s, Train loss 0.00063862215, Train acc 1.000, Val acc 0.650
Early stop: 247/250, best_acc: 0.700
Epoch 424, Time 0.484 s, Train loss 0.00063218846, Train acc 1.000, Val acc 0.650
Early stop: 248/250, best_acc: 0.700
Epoch 425, Time 0.355 s, Train loss 0.0006258628, Train acc 1.000, Val acc 0.650
Early stop: 249/250, best_acc: 0.700
Epoch 426, Time 0.446 s, Train loss 0.0006198035, Train acc 1.000, Val acc 0.650
Early stop: 250/250, best_acc: 0.700
Test acc: 0.683

Process finished with exit code 0
")