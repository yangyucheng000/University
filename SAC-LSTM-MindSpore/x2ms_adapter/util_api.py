#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import bisect
import logging

import mindspore
import numpy as np

from .context import x2ms_context


def pair(data):
    if isinstance(data, (tuple, list)):
        return data
    return data, data


class SummaryWriter(object):
    def __init__(self, logdir=None, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', **kwargs):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        pass

    def add_graph(self, model, input_to_model=None, verbose=False):
        pass

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        pass

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        pass

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        pass

    def close(self):
        pass


def amp_initialize(models, optimizers=None, enabled=True, opt_level="O1", cast_model_type=None,
                   patch_torch_functions=None, keep_batchnorm_fp32=None, master_weights=None, loss_scale=None,
                   cast_model_outputs=None, num_losses=1, verbosity=1, min_loss_scale=None, max_loss_scale=2. ** 24):
    if opt_level == "O1":
        logger.warning("MindSpore does not support O1, use O2 instead.")
        x2ms_context.amp_opt_level = "O2"
    else:
        x2ms_context.amp_opt_level = opt_level
    x2ms_context.loss_scale = loss_scale
    if optimizers is None:
        return models
    return models, optimizers


def amp_state_dict(destination=None):
    return {}


def clip_grad_norm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    x2ms_context.clip_grad_norm = max_norm
    return 0.0


def amp_master_params(optimizer):
    return optimizer.trainable_params(True)


def checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)


class GradScaler(object):
    def __init__(self, init_scale=2. ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._enabled = enabled
        if enabled:
            self._init_scale = init_scale
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._init_growth_tracker = 0
            self._growth_tracker = None
            x2ms_context.amp_opt_level = "O2"
            x2ms_context.loss_scale = init_scale

    def scale(self, outputs):
        class _ScaleResultStub:
            def backward(self, *args, **kwargs):
                pass

        return _ScaleResultStub()

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, *args, **kwargs):
        pass

    def update(self, new_scale=None):
        pass

    def get_scale(self):
        if self._enabled:
            return self._init_scale if self._scale is None else 1.0
        else:
            return 1.0

    def get_growth_factor(self):
        return self._growth_factor

    def set_growth_factor(self, new_factor):
        self._growth_factor = new_factor

    def get_backoff_factor(self):
        return self._backoff_factor

    def set_backoff_factor(self, new_factor):
        self._backoff_factor = new_factor

    def get_growth_interval(self):
        return self._growth_interval

    def set_growth_interval(self, new_interval):
        self._growth_interval = new_interval

    def is_enabled(self):
        return self._enabled

    def state_dict(self):
        if self._enabled:
            return {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
            }
        else:
            return {}

    def _get_growth_tracker(self):
        if self._enabled:
            return self._init_growth_tracker if self._growth_tracker is None else self._growth_tracker.item()
        else:
            return 0


class SingleProcessDataLoaderIter:
    class DatasetFetcher:
        def __init__(self, loader):
            self.loader = iter(loader)

        def fetch(self, index):
            return next(self.loader)

    def __init__(self, loader):
        self._num_yielded = 0
        self.loader = loader
        self._pin_memory = False
        self._index = 0
        self._dataset_fetcher = SingleProcessDataLoaderIter.DatasetFetcher(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self):
        return len(self.loader)

    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        return data

    def _next_index(self):
        start_index = self._index
        self._index += self.loader.batch_size
        return list(range(start_index, self._index))


class Generator:
    def __init__(self, device='cpu'):
        self.seed = 0

    def manual_seed(self, seed):
        self.seed = seed


class MixupStub:
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.num_classes = num_classes
        self.one_hot = mindspore.ops.OneHot()

    def __call__(self, x, target):
        target = self.one_hot(target.astype(mindspore.int64), self.num_classes,
                              mindspore.Tensor(1.0, dtype=mindspore.float32),
                              mindspore.Tensor(0.0, dtype=mindspore.float32))
        return x, target


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    logger.warning(f"MindSpore does not supported download {url} to {dst}, please download it by yourself.")


class ConcatDataset:
    def __init__(self, datasets):
        if len(datasets) <= 0:
            raise ValueError('Input datasets should not be empty.')
        self.datasets = list(datasets)
        for one_dataset in self.datasets:
            if not (hasattr(one_dataset, '__len__') and hasattr(one_dataset, '__getitem__')):
                raise TypeError("The datasets should have implemented '__len__' and '__getitem__' "
                                "method to be mindspore dataset")
        self.cumulative_index = self.index_generator(self.datasets)

    def __len__(self):
        return self.cumulative_index[-1]

    def __getitem__(self, item):
        if abs(item) > len(self):
            raise ValueError("Index out of dataset length range.")
        if item < 0:
            item += len(self)

        dataset_index = bisect.bisect_right(self.cumulative_index, item) - 1
        sample_index = item - self.cumulative_index[dataset_index]

        return self.datasets[dataset_index][sample_index]

    @staticmethod
    def index_generator(dataset_list):
        index_list = [0]
        for i, one_dataset in enumerate(dataset_list):
            index_list.append(len(one_dataset) + index_list[i])
        return index_list


def get_num_threads():
    return 1


def float_tensor_2_bool_tensor(data):
    if isinstance(data, mindspore.Tensor) and data.dtype != mindspore.bool_:
        _data = data != 0
    else:
        _data = data
    return _data


def out_adaptor(result, out):
    if out is not None:
        return out.assign_value(result)
    return result


def inplace_adaptor(original_tensor, result, inplace):
    if inplace:
        return original_tensor.assign_value(result)
    return result


_NP_TO_MS_TYPE_DICT = {
    np.float16: mindspore.float16,
    np.float32: mindspore.float32,
    np.float64: mindspore.float64,
    np.int: mindspore.int32,
    np.long: mindspore.int64,
    np.bool_: mindspore.bool_,
    np.int8: mindspore.int8,
    np.int16: mindspore.int16,
    np.int32: mindspore.int32,
    np.int64: mindspore.int64,
    np.uint8: mindspore.uint8,
    np.uint16: mindspore.uint16,
    np.uint32: mindspore.uint32,
    np.uint64: mindspore.uint64,
}


def np_to_tensor(array: np.ndarray):
    if array.size == 0 and array.ndim != 1:
        return mindspore.ops.Zeros()(array.shape, _NP_TO_MS_TYPE_DICT.get(array.dtype, mindspore.float32))
    # if array.dtype is not in supported type list (such as "|S82"), convert its dtype to str.
    if type(array.dtype).type == np.str_:
        array = mindspore.dataset.text.to_str(array)
    return mindspore.Tensor(array)


def check_input_dtype(input_tensor, support_dtype):
    trans_flag = False
    origin_type = None
    if input_tensor.dtype not in support_dtype:
        trans_flag = True
        origin_type = input_tensor.dtype
        input_tensor = input_tensor.astype(mindspore.float32)
    return trans_flag, input_tensor, origin_type


class ModelStats:
    def __init__(self, model, input_shape):
        pass

    def to_html(self, buf=None):
        pass

    def to_csv(self, path_or_buf=None):
        pass

    def iloc(self):
        pass


def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=True, verbose=False):
    pass


class Stream:
    def __init__(self):
        pass


class Logger(logging.Logger):

    def __init__(self, log_level=logging.INFO,
                 log_format='%(asctime)s [%(levelname)s] %(message)s',
                 datefmt='%Y-%m-%d %H:%M:%S'):
        super().__init__('', log_level)
        self._formatter = logging.Formatter(log_format, datefmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._formatter)
        self.addHandler(console_handler)


logger = Logger()
