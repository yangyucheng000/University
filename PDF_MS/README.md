# PDF model for MindSpore
## Paper
Link: [Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering](https://arxiv.org/abs/2305.06102)

## Quick Start
### Dataset
```python
root = "/your/path/to/Datasets/ENZYMES"
dataset = Enzymes(root)
```
### Init Params
```python
config = get_config_from_json("/your/path/to/pdf_ms/configs/ENZYMES.json")
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
```
### Dataloader
```python
train_loader, valid_loader = TUDataLoader(
        dataset, basis=basis, epsilon=epsilon, power=power,
        batch_size=32, device="GPU", seed=None, shuffle=True,
        split_name='fold10', fold_idx=0).train_valid_loader()
```
### Modules Defined
```
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
```
