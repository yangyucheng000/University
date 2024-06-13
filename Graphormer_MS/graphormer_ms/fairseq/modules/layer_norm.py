from mindspore import nn


'''try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False'''


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if type(normalized_shape) == int:
        normalized_shape = [normalized_shape]
    ln = nn.LayerNorm(normalized_shape=normalized_shape, epsilon=eps)
    return ln
    '''if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)'''