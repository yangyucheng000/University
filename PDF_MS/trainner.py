import random

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore_gl import Enzymes, BatchedGraph
from mindspore_gl.nn.gnn_cell import GNNCell

from tu.model import Net
from tu.tu_dataloader import TUDataLoader
from utils.basis_transform import basis_transform
from utils.config import get_config_from_json
import os
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class LossNet(GNNCell):
    """ LossNet definition """
    def __init__(self, net, criterion):
        super().__init__()
        self.net = net
        self.loss_fn = criterion

    def construct(self, node_feat, edge_weight, target, g: BatchedGraph):
        predict = self.net(node_feat, edge_weight, g).view((32, 6)).astype(ms.float32)
        target = ops.Squeeze()(target).astype(ms.float32)
        loss = self.loss_fn(predict, target).view((1))
        loss = ops.ReduceSum()(loss * g.graph_mask) / (32)
        return loss


def show(data, idx, title="ENZYMES loss"):
    plt.figure(figsize=(20, 10))
    plt.plot(data, idx)
    plt.title(title)
    plt.show()


def train(model, dataloader, train_net):
    train_loss = []
    for batch in dataloader:
        graphs, feats, bases, labels = batch.bg, batch.feats, batch.bases, batch.labels
        loss = train_net(feats, bases, labels, *graphs.get_batched_graph())
        train_loss.append(loss)
        print(loss)

    return train_loss


def train2(model, dataloader, criterion, optimizer=None, lossNet = None):
    grad = C.GradOperation(get_by_list=True, sens_param=True)
    grad_no_sens = C.GradOperation(get_by_list=True)
    grad_reducer = nn.Identity()
    weights = optimizer.parameters

    for batch in dataloader:
        graphs, feats, bases, labels = batch.bg, batch.feats, batch.bases, batch.labels
        loss = lossNet(feats, bases, labels, *graphs.get_batched_graph())
        sens = F.fill(loss.dtype, loss.shape, 1.0)
        grads = grad(lossNet, weights)(feats, bases, labels, *graphs.get_batched_graph(), sens)
        # grads = grad_no_sens(lossNet, weights)(feats, bases, labels, *graphs.get_batched_graph())
        grads = grad_reducer(grads)
        loss = F.depend(loss, optimizer(grads))


def test(model, dataloader, criterion):
    for batch in dataloader:
        graphs, feats, bases, labels = batch.bg, batch.feats, batch.bases, batch.labels
        predict = model(feats, bases, *graphs.get_batched_graph()).view((32, 6)).astype(ms.float32)
        target = ops.Squeeze()(labels).astype(ms.float32)
        loss = criterion(predict, target)
        predict = ops.softmax(predict)
        print(predict)


def main():
    root = "/home/mzx/Datasets/ENZYMES"
    dataset = Enzymes(root)
    config = get_config_from_json("/home/mzx/Codes/pdf_ms/configs/ENZYMES.json")
    econ = config.architecture

    data = dataset[0]
    feat = dataset.graph_node_feat(0)

    basis = "DEN"
    epsilon = [-0.2, -0.25, -0.3, -0.35]
    power = [[1, 6], [1, 6], [1, 6], [1, 6]]
    new_g, feat, bases = basis_transform(data, feat=feat, basis=basis, epsilon=epsilon, power=power,
                           degs=[],
                           edgehop=None,
                           basis_norm=False)

    train_loader, valid_loader = TUDataLoader(
        dataset, basis=basis, epsilon=epsilon, power=power,
        batch_size=32, device="GPU", seed=None, shuffle=True,
        split_name='fold10', fold_idx=0).train_valid_loader()

    model = Net(input_dim=feat.shape[1],
              output_dim=dataset.label_dim,
              num_basis=bases.shape[1],
              config=econ)

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=config.hyperparams.learning_rate,
                                   weight_decay=config.hyperparams.get('weight_decay', 0))
    loss = LossNet(model, criterion)
    train_net = nn.TrainOneStepCell(loss, optimizer, sens=1.0)

    train_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(1, config.hyperparams.epochs):
        model.set_train(True)
        train_loss = train(loss, train_loader, train_net)
        train_losses += train_loss
    num = len(train_losses)
    t2 = train_losses[:int(num / 2)]
    t4 = train_losses[:int(num / 4)]

    show(list(range(num)), train_losses)
    show(list(range(int(num / 2))), t2)
    show(list(range(int(num / 4))), t4)


    for epoch in range(1, config.hyperparams.epochs):
        test(model, valid_loader, criterion)


if __name__ == '__main__':
    main()
