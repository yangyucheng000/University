import mindspore as ms
import numpy as np
import torch

def ms2torch(t):
    if isinstance(t,ms.Tensor):
        return torch.tensor(t.numpy()).cuda()
    return t

def torch2ms(t):
    if isinstance(t,torch.Tensor):
        return ms.tensor(t.detach().cpu().numpy())
    return t

def ck_list(l,func):
    new_l=[]
    for idx,i in enumerate(l):
        if isinstance(i,list):
            new_l.append(ck_list(i,func))
        elif isinstance(i,tuple):
            new_l.append(ck_tuple(i,func))
        elif isinstance(i,dict):
            new_l.append(ck_dict(i,func))
        else:
            new_l.append(func(i))
    return new_l

def ck_tuple(l,func):
    new_tuple=()
    for idx,i in enumerate(l):
        if isinstance(i,list):
            new_tuple=(new_tuple,ck_list(i,func))
        elif isinstance(i,tuple):
           new_tuple=(new_tuple,ck_tuple(i,func))
        elif isinstance(i,dict):
            new_tuple=(new_tuple,ck_dict(i,func))
        else:
            new_tuple=(new_tuple,func(i))
    return new_tuple

def ck_dict(l,func):
    new_d={}
    for k in l:
        i=l[k]
        if isinstance(i,list):
            new_d[k]=ck_list(i,func)
        elif isinstance(i,tuple):
            new_d[k]=ck_tuple(i,func)
        elif isinstance(i,dict):
            new_d[k]=ck_dict(i,func)
        else:
            new_d[k]=func(i)
    return new_d

def m_t_tensor_convert(func):
    def wrapper(*args,**kwargs):
        args=ck_list(args,ms2torch)
        kwargs=ck_dict(kwargs,ms2torch)
        ret=func(*args,**kwargs)
        if isinstance(ret,list):
            ret=ck_list(ret,torch2ms)
        elif isinstance(ret,tuple):
            ret=ck_tuple(ret,torch2ms)
        elif isinstance(ret,dict):
            ret=ck_dict(ret,torch2ms)
        else:
            ret=torch2ms(ret)
        return ret
    return wrapper

def t_m_tensor_convert(func):
    def wrapper(*args,**kwargs):
        args=ck_list(args,torch2ms)
        kwargs=ck_dict(kwargs,torch2ms)
        ret=func(*args,**kwargs)
        if isinstance(ret,list):
            ret=ck_list(ret,ms2torch)
        elif isinstance(ret,tuple):
            ret=ck_tuple(ret,ms2torch)
        elif isinstance(ret,dict):
            ret=ck_dict(ret,ms2torch)
        else:
            ret=ms2torch(ret)
        return ret
    return wrapper