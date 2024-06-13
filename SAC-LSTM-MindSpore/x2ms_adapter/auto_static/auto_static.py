#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


def g_list(pack):
    new_list = []
    for ele in pack:
        new_list.append(ele)
    return new_list


def g_extend(origin_list, pack):
    for ele in pack:
        origin_list.append(ele)
    return origin_list


def g_dict(iterator):
    dct = {}
    for key, val in iterator:
        if not isinstance(key, str):
            raise ValueError("Graph mode only support string type key.")
        dct[key] = val

    return dct
