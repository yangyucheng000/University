import mindspore
from mindspore_gl.dataset.enzymes import Enzymes


def get_one_node_feat(dataset: Enzymes):
    feats = dataset.node_feat
    for i in list(range(dataset.graph_count)):
        feat = dataset.graph_node_feat(i)
        print(feat)


if __name__ == '__main__':
    # root = "/home/mzx/Datasets/ENZYMES"
    # dataset = Enzymes(root)
    # get_one_node_feat(dataset)
    pass