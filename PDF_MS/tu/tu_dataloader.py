import mindspore as ms
from mindspore import Tensor, ops
from mindspore_gl import BatchedGraphField
from mindspore_gl.dataloader import RandomBatchSampler, split_data
from tqdm import tqdm

from utils.basis_transform import basis_transform


class graphTmp():
    def __init__(self, graph, feat, bases, label):
        self.g = graph
        self.feat = Tensor(feat)
        self.bases = bases
        self.label = Tensor(label).view((1)).astype(ms.int32)


class batchedGraphTmp():
    def __init__(self, batchedgraph, feats, bases, labels):
        self.bg = batchedgraph
        self.feats = feats.astype(ms.float32)
        self.bases = bases.astype(ms.float32)
        self.labels = Tensor(labels)


class TUDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 basis,
                 epsilon,
                 power,
                 collate_fn=None,
                 seed=0,
                 shuffle=True,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.label_dim = 6
        self.kwargs = {'pin_memory': True} if 'GPU' == device else {}

        new_dataset = []
        for i in tqdm(range(dataset.graph_count - 1), desc="Pre-process"):
            g = dataset[i]
            feat = dataset.graph_node_feat(i)
            new_g, feat, bases = basis_transform(g, feat=feat,
                                                 basis=basis,
                                                 epsilon=epsilon,
                                                 power=power,
                                                 degs=[],
                                                 edgehop=None,
                                                 basis_norm=False)
            tmp = graphTmp(new_g, feat, bases, dataset.graph_label[i])
            new_dataset.append(tmp)

        train_dataset, valid_dataset = self._split_rand(new_dataset)
        train_sampler = RandomBatchSampler(data_source=train_dataset, batch_size=batch_size)
        valid_sampler = RandomBatchSampler(data_source=valid_dataset, batch_size=batch_size)

        self.train_loader = self._get_bacthed_loader(train_sampler)
        self.valid_loader = self._get_bacthed_loader(valid_sampler)


    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        pass

    def _split_rand(self, dataset, split_ratio=0.7, seed=0, shuffle=True):
        train_len = int(len(dataset) * split_ratio) + 1
        train_dataset = dataset[0:train_len]
        val_dataset = dataset[train_len:]
        return train_dataset, val_dataset

    def _get_bacthed_loader(self, sampler):
        batchedLoader = []
        for s in sampler:
            src_idx = Tensor([], dtype=ms.int64)
            dst_idx = Tensor([], dtype=ms.int64)
            n_node = 0
            n_edge = 0
            ver_subgraph_idx = Tensor([], dtype=ms.int64)
            edge_subgraph_idx = Tensor([], dtype=ms.int64)

            feat = None
            bases = None
            label = None

            for i, g in enumerate(s):
                graph = g.g.get_graph()
                src_idx = ops.cat([src_idx, Tensor(graph[0])])
                dst_idx = ops.cat([dst_idx, graph[1]])
                n_node += graph[2]
                n_edge += graph[3]
                ver_subgraph_idx = ops.cat([ver_subgraph_idx, ops.full((graph[2],), i, dtype=ms.int64)])
                edge_subgraph_idx = ops.cat([edge_subgraph_idx, ops.full((graph[3],), i, dtype=ms.int64)])
                if feat == None:
                    feat = g.feat
                else:
                    feat = ops.cat([feat, g.feat])
                if bases == None:
                    bases = g.bases
                else:
                    bases = ops.cat([bases, g.bases])

                labeltmp = ops.zeros((1, self.label_dim))
                labeltmp[0][g.label] = 1
                if label == None:
                    label = labeltmp
                else:
                    label = ops.cat([label, labeltmp], axis=0)
            graph_mask = ops.full((len(s),), 1, dtype=ms.int64)
            batchedGraphField = BatchedGraphField(src_idx, dst_idx,
                                                  n_node, n_edge,
                                                  ver_subgraph_idx,
                                                  edge_subgraph_idx,
                                                  graph_mask)
            bgt = batchedGraphTmp(batchedGraphField,
                                  feats=feat, bases=bases, labels=label)
            batchedLoader.append(bgt)
        return batchedLoader
