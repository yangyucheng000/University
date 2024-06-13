

import os
from core.models import predict
from core.layers.Discriminator import Discriminatorr
from x2ms_adapter.optimizers import optim_register
import mindspore
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_cell
import x2ms_adapter.loss as loss_wrapper
import mindspore as ms


class Model(mindspore.nn.Cell):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        
        
        #导入模型结构
        networks_map = {
            'convlstm':predict.ConvLSTM,
            'predrnn':predict.PredRNN,
            'predrnn_plus': predict.PredRNN_Plus,
            'sac_lstm': predict.SAC_LSTM,    
        }

        if configs.model_name in networks_map:

            Network = networks_map[configs.model_name]
            
            # self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            self.network = x2ms_adapter.to(Network(self.num_layers, self.num_hidden, configs), configs.device)
            
            # self.Discriminator = Discriminatorr(self.patch_height, self.patch_width, self.patch_channel,
            #                                self.configs.D_num_hidden).to(configs.device)
            self.network = x2ms_nn.DataParallel(self.network)
            # amp_level = "O3"
            # mindspore.amp.auto_mixed_precision(self.network, amp_level)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        if self.configs.is_parallel:
            self.network = x2ms_nn.DataParallel(self.network)
        # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        self.optimizer = optim_register.adam(x2ms_adapter.parameters(self.network), lr=configs.lr)
        # self.optimizer_D = Adam(self.Discriminator.parameters(), lr=configs.lr_d)
        
        self.MSE_criterion = x2ms_nn.MSELoss(size_average=False)
        # self.D_criterion = nn.BCELoss()
        self.L1_loss = loss_wrapper.L1Loss(size_average=False)


    def save(self,ite = None):
        stats = {}
        stats['net_param'] = x2ms_adapter.state_dict(self.network)
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_pm.ckpt')
        x2ms_adapter.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)
        
        # stats = {}
        # stats['net_param'] = self.Discriminator.state_dict()
        # checkpoint_path = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        # torch.save(stats, checkpoint_path)
        # print("save discriminator model to %s" % checkpoint_path)

    def load(self):
        print('model has been loaded:')
        checkpoint_path = os.path.join(self.configs.save_dir, 'convert_mindspore_pytorch_2.ckpt')
        # checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        stats = x2ms_adapter.load(checkpoint_path)
        x2ms_adapter.load_state_dict(self.network, stats['net_param'],False)

        
        # print('load discriminator model:')
        # checkpoint_path_d = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        # stats = torch.load(checkpoint_path_d)
        # self.Discriminator.load_state_dict(stats['net_param'])
    
    def load_mindspore(self):
        print('model has been loaded:')
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_pm.ckpt')
        # checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        stats = x2ms_adapter.load(checkpoint_path)
        x2ms_adapter.load_state_dict(self.network, stats['net_param'],False)
        
        ms_dict = self.network.parameters_dict()
        # for pt_key in ms_dict:  
        #     print(pt_key)

            
        for param in self.network.get_parameters():
            name=param.name
            value=param.data.asnumpy()
            print(name,value.shape)
        
        
        # print('load discriminator model:')
        # checkpoint_path_d = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        # stats = torch.load(checkpoint_path_d)
        # self.Discriminator.load_state_dict(stats['net_param'])
        
    def load_pytorch(self):
        "no-gan"
        print('model has been loaded:')
        checkpoint_path_pm = os.path.join(self.configs.save_dir, 'model.ckpt')
        stats = torch.load(checkpoint_path_pm)
        self.network.load_state_dict(stats['net_param'],False)
        
        pt_dict = self.network.state_dict()
        for pt_key in pt_dict:  
            print(pt_key)
        
    
    
    
    def train(self, frames, mask):
        return self(frames, mask)

    def construct(self, frames, mask):
        
        # frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        frames_tensor = frames
        mask_tensor = mask
        # print("frames_tensor",frames_tensor.shape)
        # self.optimizer.zero_grad()

        # print("frames_tensor",frames_tensor.shape)
        # print("mask_tensor",mask_tensor.shape)
        next_frames = self.network(frames_tensor, mask_tensor)
        # print("frames_tensor",frames_tensor.shape)
        # print("next_frames",next_frames.shape)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])+\
               self.L1_loss(next_frames, frames_tensor[:, 1:])
               # 0.02*self.SSIM_criterion(next_frames, frames_tensor[:, 1:])
        return loss
        
        
        
        
        # # print("mask_tensor",mask_tensor.shape)
        # next_frames = self.network(frames_tensor, mask_tensor)
        # # print("next_frames",next_frames.shape)
        # ground_truth = frames_tensor[:, 1:]
        
        # next_frames = next_frames.permute(0, 1, 4, 2, 3)
        # ground_truth = ground_truth.permute(0, 1, 4, 2, 3)
        
        # batch_size = next_frames.shape[0]
        # zeros_label = torch.zeros(batch_size).cuda()
        # ones_label = torch.ones(batch_size).cuda()
        
        # # train D
        # self.Discriminator.zero_grad()
        # d_gen, _ = self.Discriminator(next_frames.detach())
        # d_gt, _ = self.Discriminator(ground_truth)
        # D_loss = self.D_criterion(d_gen, zeros_label) + self.D_criterion(d_gt, ones_label)
        # D_loss.backward(retain_graph=True)
        # self.optimizer_D.step()
        
        # self.optimizer.zero_grad()
        # d_gen_pre, features_gen = self.Discriminator(next_frames)
        # _, features_gt = self.Discriminator(ground_truth)
        
        # loss_l1 = self.L1_loss(next_frames, ground_truth)
        # loss_l2 = self.MSE_criterion(next_frames, ground_truth)
        # gen_D_loss = self.D_criterion(d_gen_pre, ones_label)
        # loss_features = self.MSE_criterion(features_gen, features_gt)
        # loss = loss_l1 + loss_l2 + 0.001*loss_features + 0.0001*gen_D_loss
        # loss.backward()
        # self.optimizer.step()
        # return loss.detach().cpu().numpy()

    def test(self, frames, mask):

        frames_tensor = x2ms_adapter.FloatTensor(frames)
        mask_tensor = x2ms_adapter.FloatTensor(mask)
        # print("frames_tensor_test",frames_tensor.shape)
        # print("mask_tensor_test",mask_tensor.shape)
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) +\
               self.L1_loss(next_frames,frames_tensor[:,1:])
               # + 0.02 * self.SSIM_criterion(next_frames, frames_tensor[:, 1:])

        return x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(next_frames)),x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(loss))

class Model_convert_pytorch_to_mindspore(mindspore.nn.Cell):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in x2ms_adapter.tensor_api.split(configs.num_hidden, ',')]
        self.num_layers = len(self.num_hidden)
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        
        
        #导入模型结构
        networks_map = {
            'convlstm':predict.ConvLSTM,
            'predrnn':predict.PredRNN,
            'predrnn_plus': predict.PredRNN_Plus,
            'sac_lstm': predict.SAC_LSTM,    
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            # Network_torch = networks_map[configs.model_name_torch]
            self.network = x2ms_adapter.to(Network(self.num_layers, self.num_hidden, configs), configs.device)
            # self.network_torch = Network_torch(self.num_layers, self.num_hidden, configs)
            # self.network = x2ms_nn.DataParallel(self.network)

        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        
        self.MSE_criterion = x2ms_nn.MSELoss(size_average=False)
        self.L1_loss = loss_wrapper.L1Loss(size_average=False)


    def save(self,ite = None):
        stats = {}
        stats['net_param'] = x2ms_adapter.state_dict(self.network)
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_pm.ckpt')
        x2ms_adapter.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)
        
        # stats = {}
        # stats['net_param'] = self.Discriminator.state_dict()
        # checkpoint_path = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        # torch.save(stats, checkpoint_path)
        # print("save discriminator model to %s" % checkpoint_path)

    def load(self):
        print('model has been loaded:')
        checkpoint_path = os.path.join(self.configs.save_dir, 'convert_mindspore_pytorch_new.ckpt')
        # checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        stats = x2ms_adapter.load(checkpoint_path)
        x2ms_adapter.load_state_dict(self.network, stats['net_param'],False)

        
        # print('load discriminator model:')
        # checkpoint_path_d = os.path.join(self.configs.save_dir, 'model_d.ckpt')
        # stats = torch.load(checkpoint_path_d)
        # self.Discriminator.load_state_dict(stats['net_param'])
    
    def load_mindspore_custom(self):
        "mindspore"
        self.network.set_train(False)
        checkpoint_path = os.path.join(self.configs.save_dir, 'convert_mindspore_pytorch_new.ckpt')
        # param_dict=ms.load_checkpoint(checkpoint_path)
        # ms.load_param_into_net(self.network,param_dict)
        ms.load_checkpoint(checkpoint_path, self.network)
        print("========= ms_resnet conv1.weight ==========")
        # ms_dict = self.network.parameters_dict()
        # for pt_key in ms_dict:  
        #     pass
        #     # print(pt_key)
        "可以查看权重的每层值"
        # for index,i in enumerate(self.network.get_parameters()):
        #     # print(i.name)
        #     # if index < 2:
        #     # print(i.data)
        #     print(i.data.asnumpy().reshape((-1,))[:10])
        
        """torch"""
        # self.network_torch.eval()
        # pth_path = "checkpoints_debug/pytorch/pytorch_model.ckpt"
        # # self.network_torch.load_state_dict(torch.load(pth_path), map_location=torch.device('cpu') )
        # self.network_torch.load_state_dict(torch.load(pth_path), map_location=torch.device('npu:0') )
        # print(self.network)
        # print(self.network.cell_list.value_conv.weight.data.asnumpy().reshape((-1,))[:10])
    
    def train(self, frames, mask):
        return self(frames, mask)

    def construct(self, frames, mask):
        

        frames_tensor = frames
        mask_tensor = mask
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])+\
               self.L1_loss(next_frames, frames_tensor[:, 1:])
               # 0.02*self.SSIM_criterion(next_frames, frames_tensor[:, 1:])
        return loss
        
        
        # # print("mask_tensor",mask_tensor.shape)
        # next_frames = self.network(frames_tensor, mask_tensor)
        # # print("next_frames",next_frames.shape)
        # ground_truth = frames_tensor[:, 1:]
        
        # next_frames = next_frames.permute(0, 1, 4, 2, 3)
        # ground_truth = ground_truth.permute(0, 1, 4, 2, 3)
        
        # batch_size = next_frames.shape[0]
        # zeros_label = torch.zeros(batch_size).cuda()
        # ones_label = torch.ones(batch_size).cuda()
        
        # # train D
        # self.Discriminator.zero_grad()
        # d_gen, _ = self.Discriminator(next_frames.detach())
        # d_gt, _ = self.Discriminator(ground_truth)
        # D_loss = self.D_criterion(d_gen, zeros_label) + self.D_criterion(d_gt, ones_label)
        # D_loss.backward(retain_graph=True)
        # self.optimizer_D.step()
        
        # self.optimizer.zero_grad()
        # d_gen_pre, features_gen = self.Discriminator(next_frames)
        # _, features_gt = self.Discriminator(ground_truth)
        
        # loss_l1 = self.L1_loss(next_frames, ground_truth)
        # loss_l2 = self.MSE_criterion(next_frames, ground_truth)
        # gen_D_loss = self.D_criterion(d_gen_pre, ones_label)
        # loss_features = self.MSE_criterion(features_gen, features_gt)
        # loss = loss_l1 + loss_l2 + 0.001*loss_features + 0.0001*gen_D_loss
        # loss.backward()
        # self.optimizer.step()
        # return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        self.network.set_train(False)
        frames_tensor = x2ms_adapter.FloatTensor(frames)
        mask_tensor = x2ms_adapter.FloatTensor(mask)
        print("frames_tensor_test",frames_tensor.shape)
        print("mask_tensor_test",mask_tensor.shape)
        next_frames = self.network(frames_tensor, mask_tensor)
        # print("next_frames", next_frames[0,0,0])
        # print("next_frames", next_frames)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) +\
               self.L1_loss(next_frames,frames_tensor[:,1:])
               # + 0.02 * self.SSIM_criterion(next_frames, frames_tensor[:, 1:])

        return x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(next_frames)),x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(loss))
