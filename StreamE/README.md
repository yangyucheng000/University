# pytorch implementation of StreamE

Paper: "StreamE: Learning to Update Representations for Temporal Knowledge Graphs in Streaming Scenarios"

In this paper, we propose a lightweight framework called StreamE for the efficient generation of TKG representations in streaming scenarios. To reduce the parameter size, entity representations in StreamE are decoupled from the model training to
serve as the memory module to store the historical information of entities. To achieve efficient update and generation, the process of generating representations is decoupled as two functions
in StreamE. An update function is learned to incrementally update entity representations based on the newly-arrived knowledge and
a read function is learned to predict the future semantics of entity
representations. The Mindspore version implementation of our code is in https://github.com/zjs123/StreamE_MindSpore.

<p align="center"><img src="StreamE.PNG"/></p>

If you make use of this code in your work, please cite the following paper:

```
@inproceedings{DBLP:conf/sigir/ZhangS023,
  author       = {Jiasheng Zhang, Jie Shao, and, Bin Cui},
  title        = {StreamE: Learning to Update Representations for Temporal Knowledge
                  Graphs in Streaming Scenarios},
  booktitle    = {SIGIR 2023},
  pages        = {622--631},
  year         = {2023}
}
```

## Contents
1. [Installation](#installation)
2. [Train_and_Test](#Train_and_Test)
3. [Datasets](#Datasets)
4. [Baselines](#Baselines)
5. [Contact](#contact)

## Installation

Install the following packages:

```
pip install torch
pip install numpy
```

Install CUDA and cudnn. Then run:

```
pip install cutorch
pip install cunn
pip install cudnn
```

Then clone the repository::

```
git clone https://github.com/zjs123/StreamE.git
```

We use Python3 for data processing and our code is also written in Python3. 

## Train_and_Test

The user can use the following command to reproduce the reported results of our models, in which the train and the evaluation processes are performed automatically.
```
python Main.py -dataset ICEWS14
```
Some of the important available options include:
```
        '-hidden', default = 100, type = int, help ='dimension of the learned embedding'
	'-lr',  default = 0.001, type = float, help = 'Learning rate'
	'-ns', 	 default = 10,   	type=int, 	help='negative samples for training'
	'-dataset', 	 default = "ICEWS14",   	choice=["ICEWS14","ICEWS05","GDELT"], 	help='dataset used to train'
	'-numOfEpoch', 	default=300,	type=int	help='Train Epoches'
   ```

## Datasets

There are four datasets used in our experiment:ICEWS14, ICEWS05-15, ICEWS18, and GDELT. facts of each datases are formed as "[subject entity, relation, object entity, time]". Each data folder has four files: 

**-train.txt, test.txt, valid.txt:** the first column is index of subject entity, second column is index of relation, third column is index of object entity, fourth column is the happened time of fact.

**-stat.txt:** num of entites and num of relations

The detailed statistic of each dataset
| Datasets   | Num of Entity | Num of Relation | Num of Time | Train | Valid | Test |
|------------|---------------|-----------------|-------------|-------|-------|------|
| ICEWS14 ([Alberto et al., 2018](https://www.aclweb.org/anthology/D18-1516.pdf))    | 7,128         | 230             | 365         | 72,826| 8,941 | 8,963 |
| ICEWS05-15 ([Alberto et al., 2018](https://www.aclweb.org/anthology/D18-1516.pdf))  | 10,488        | 251             | 4,071       | 38,6962| 46,275| 46,092|
| ICEWS18 ([Zhen Han et al., 2020](https://arxiv.org/abs/2012.15537v4))  | 23,033       | 256             | 304       | 373,018| 45,995| 49,545|
|GDELT ([Goel et al., 2018](https://arxiv.org/pdf/1907.03143.pdf))     | 500           | 20              | 366         | 2,735,685| 341,961| 341,961 |

## Baselines

We use following public codes for baseline experiments. 

| Baselines   | Code                                                                      | Embedding size | Batch num |
|-------------|---------------------------------------------------------------------------|----------------|------------|
| TransE ([Bordes et al., 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data))      | [Link](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/openke) | 100, 200       | 100, 200       |
| TTransE ([Leblay et al., 2018](https://dl.acm.org/doi/fullHtml/10.1145/3184558.3191639))    | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)                                  | 50, 100, 200   | 100, 200       |
| TA-TransE ([Alberto et al., 2018](https://www.aclweb.org/anthology/D18-1516.pdf))      | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)     | 100, 200            | Default    |
| HyTE ([Dasgupta et al., 2018](http://talukdar.net/papers/emnlp2018_HyTE.pdf))        | [Link](https://github.com/malllabiisc/HyTE)                               | Default            | Default    |
| DE-DistMult ([Goel et al., 2020](https://arxiv.org/pdf/1907.03143.pdf))        | [Link](https://github.com/BorealisAI/de-simple)                               | Default            | Default    |
| TNTComplEX ([Timothee et al., 2020](https://openreview.net/pdf?id=rke2P1BFwS))        | [Link](https://github.com/facebookresearch/tkbc)                               | Default            | Default    |
| ATiSE ([Chenjin et al., 2020](https://arxiv.org/pdf/1911.07893.pdf))        | [Link](https://github.com/soledad921/ATISE)                               | Default            | Default    |

## Contact

For any questions or suggestions you can use the issues section or contact us at zjss12358@gmail.com.
