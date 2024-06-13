import os
from config import Config 

import random
import time
import numpy as np

import utils.dir_utils as utils
from data_RGB import get_training_data
from models.MPRNet import MPRNet
import models.losses as losses
from tqdm import tqdm
from models.kpn_simple import Reblur_Model
from models.warp import GetBackwarp
from flowfunction import mask_gene

import mindspore
import mindspore.nn as nn
import mindspore.ops as O
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import CheckpointConfig,ModelCheckpoint,LossMonitor


opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

device_nums = len(gpus)

######### Set Seeds ###########
random.seed(1234)
mindspore.set_seed(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = GeneratorDataset(train_dataset, column_names=['img', 'gt'], shuffle=True, num_parallel_workers=8).batch(opt.OPTIM.BATCH_SIZE)

#val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
#val_loader = GeneratorDataset(val_dataset, shuffle=False, num_parallel_workers=8).batch(16)


######### Scheduler ###########
step_pre_epoch=train_loader.get_dataset_size()
warmup_epochs = 3
warmup_scheduler = nn.warmup_lr(opt.OPTIM.LR_INITIAL,step_pre_epoch*warmup_epochs,step_pre_epoch,warmup_epochs)
scheduler_cosine = nn.cosine_decay_lr(opt.OPTIM.LR_MIN,opt.OPTIM.LR_INITIAL,step_pre_epoch*(opt.OPTIM.NUM_EPOCHS-warmup_epochs),step_pre_epoch,opt.OPTIM.NUM_EPOCHS-warmup_epochs)
scheduler = warmup_scheduler+scheduler_cosine
start_epoch = 0

######### Model ###########
model_restoration = MPRNet()
reblur_model = Reblur_Model()

######### Resume ###########
if opt.TRAINING.RESUME is not None:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    ckpt = mindspore.load_checkpoint(path_chk_rest)
    mindspore.load_param_into_net(model_restoration, ckpt)
    start_epoch = opt.TRAINING.RESUME

    new_lr = scheduler[start_epoch]
    scheduler = scheduler[start_epoch:]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

optimizer = nn.Adam(model_restoration.trainable_params(), learning_rate=scheduler, beta1=0.9, beta2=0.999, eps=1e-8)

if device_nums>1:
    print("multiple GPUs are not supported right now")

######### Loss ###########
net = losses.NetWithLoss(model_restoration, GetBackwarp(), reblur_model, mask_gene)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

model = mindspore.Model(network=net, optimizer=optimizer)

ckpt_config = CheckpointConfig(save_checkpoint_steps=step_pre_epoch*10, keep_checkpoint_max=10)
ckpt_callback = ModelCheckpoint(prefix="mprnet", directory=model_dir, config=ckpt_config)
loss_monitor = LossMonitor(step_pre_epoch)

model.train(opt.OPTIM.NUM_EPOCHS+1-start_epoch, train_loader, callbacks=[loss_monitor,ckpt_callback])

'''

data_iterator = train_loader.create_tuple_iterator(num_epochs=opt.OPTIM.NUM_EPOCHS-start_epoch+1)

def forward_fn(input_, target):
    restored = model_restoration(input_)

    z_tensor = mindspore.Tensor(0, dtype=restored[0].dtype)
    o_tensor = mindspore.Tensor(1, dtype=restored[0].dtype)
    output = O.clip_by_value(restored[0], z_tensor, o_tensor)

    warp_gt, flow = get_backwarp(output, target, pwcnet)
    warp_out, flow_ = get_backwarp(target, output, pwcnet)

    reblur_out_0 = reblur_model(restored[0], input_)
    reblur_out_1 = reblur_model(restored[1], input_)
    reblur_out_2 = reblur_model(restored[2], input_)

    mask = mask_gene(flow, 0.4)
    mask_ = mask_gene(flow_, 0.4)

    # Compute loss at each stage
    loss_char = np.sum([criterion_char(restored[j]*mask, warp_gt*mask) for j in range(len(restored))])
    loss_edge = np.sum([criterion_edge(restored[j]*mask, warp_gt*mask) for j in range(len(restored))])
    loss_reblur = criterion_char(reblur_out_0, input_)+criterion_char(reblur_out_1, input_)+criterion_char(reblur_out_2, input_)
    # loss_reblur = criterion_char(reblur_out_0, input_)
    loss_out = criterion_char(warp_out*mask_, target*mask_)
    loss = (loss_char) + loss_out + 0.5*loss_reblur + (0.05*loss_edge)

    return loss, restored

grad_fn = value_and_grad(forward_fn, None, optimizer.trainable_params(), has_aux=True)
step = 0
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.set_train()
    for i, data in enumerate(tqdm(data_iterator)):

        target = mindspore.Tensor(data[0])
        input_ = mindspore.Tensor(data[1])

        (loss, _), grads = grad_fn(input_, target)
        loss = O.depend(loss, optimizer(grads))

        epoch_loss +=loss.asnumpy()
        step+=1

    #### Evaluation ####
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler[step]))
    print("------------------------------------------------------------------")

    mindspore.save_checkpoint(model_restoration, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))
'''