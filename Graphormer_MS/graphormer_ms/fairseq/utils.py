from typing import Callable, Optional, List
import warnings

from mindspore import nn, Tensor
import mindspore as ms


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        sm = nn.Softmax(axis=dim)
        return sm(x)
    else:
        sm = nn.Softmax(axis=dim)
        return sm(x)


def relu_squared(x: Tensor):
    relu = nn.ReLU()
    return relu(x).pow(2)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    # from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        relu = nn.ReLU()
        return relu
    elif activation == "relu_squared":
        return relu_squared
        '''elif activation == "gelu":
            return gelu
        elif activation == "gelu_fast":
            deprecation_warning(
                "--activation-fn=gelu_fast has been renamed to gelu_accurate"
            )
            return gelu_accurate
        elif activation == "gelu_accurate":
            return gelu_accurate'''
    elif activation == "tanh":
        tanh = nn.Tanh()
        return tanh
    elif activation == "linear":
        linear = lambda x: x
        return linear
    elif activation == "swish":
        silu = nn.SiLU()
        return silu
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]


def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None
