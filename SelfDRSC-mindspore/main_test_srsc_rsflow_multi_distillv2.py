import os.path
import argparse
import logging
from mindspore.dataset import GeneratorDataset as DataLoader

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from models.select_model import define_Model

from data.dataset_rsgopro_self import RSGOPRO as D

import mindspore as ms
import mindspore.dataset as ds
ds.config.set_enable_autotune(True)
#ms.set_context(mode=ms.GRAPH_MODE)

def main(json_path='options/test_srsc_rsflow_multi_distillv2_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # # ----------------------------------------
    # # update opt
    # # ----------------------------------------

    border = opt['scale']

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'test_110000'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # # ----------------------------------------
    # # seed
    # # ----------------------------------------
    # seed = opt['train']['manual_seed']
    # if seed is None:
    #     seed = random.randint(1, 10000)
    # print('Random seed: {}'.format(seed))
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            # test_set = define_Dataset(dataset_opt)
            path = os.path.join(dataset_opt['data_root'], 'test')  #'valid') #
            # flow = True
            # if not 'EPE' in para.loss:
            flow = False
            test_set = D(path, dataset_opt['future_frames'], dataset_opt['past_frames'], dataset_opt['frames'], None, dataset_opt['centralize'],
                          dataset_opt['normalize'], flow, False, True)
            test_loader = DataLoader(test_set,shuffle=False, num_parallel_workers=1,
                                     column_names=["rs_imgs", "gs_imgs", "fl_imgs", "prior_imgs", "time_rsc", "all_time_rsc", 
                                                    "out_paths", "input_path"]).batch(batch_size=1, drop_remainder=False)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0

    for test_data in test_loader:
        idx += 1
        video_name = test_data[7].numpy()[0][0].split('/')[-3]

        img_dir = os.path.join(opt['path']['inference_results'], video_name)
        util.mkdir(img_dir)
        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint_list(visuals['E'])
        H_img = util.tensor2uint_list(visuals['H'])

        current_psnr = 0
        current_ssim = 0
        for save_idx in range(len(E_img)):
            image_name_ext = os.path.basename(test_data[7].numpy()[0][save_idx])
            img_name, ext = os.path.splitext(image_name_ext)
            save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
            util.imsave(E_img[save_idx], save_img_path)
            current_psnr += util.calculate_psnr(E_img[save_idx], H_img[save_idx])
            current_ssim += util.calculate_ssim(E_img[save_idx], H_img[save_idx])
        # -----------------------
        # calculate PSNR
        # -----------------------

        current_psnr = current_psnr / len(E_img)
        current_ssim = current_ssim / len(E_img)

        logger.info('{:->4d}--> {:>10s} | {:<4.3f}dB  {:.4f}'.format(idx, image_name_ext, current_psnr, current_ssim))

        avg_psnr += current_psnr
        avg_ssim += current_ssim

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # testing log
    logger.info('Average PSNR : {:<.3f}dB\n'.format(avg_psnr))
    logger.info('Average SSIM : {:<.4f}\n'.format(avg_ssim))

if __name__ == '__main__':
    main()
