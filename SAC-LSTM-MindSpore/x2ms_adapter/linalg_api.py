#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore

from .util_api import out_adaptor


def inv(A, *, out=None):
    result = mindspore.ops.MatrixInverse()(A)
    return out_adaptor(result, out)
