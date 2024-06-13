import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O
from .ms_tensor_utils import m_t_tensor_convert

backwarp_tenGrid = {}

def warp(tenInput, tenFlow, device='cuda'):
    k = (str(tenFlow.shape))
    if k not in backwarp_tenGrid:
        tenHorizontal = O.linspace(-1.0, 1.0, tenFlow.shape[3]).view(
            1, 1, 1, tenFlow.shape[3]).broadcast_to((tenFlow.shape[0], -1, tenFlow.shape[2], -1))
        tenVertical = O.linspace(-1.0, 1.0, tenFlow.shape[2]).view(
            1, 1, tenFlow.shape[2], 1).broadcast_to((tenFlow.shape[0], -1, -1, tenFlow.shape[3]))
        backwarp_tenGrid[k] = O.cat([tenHorizontal, tenVertical], 1)

    tenFlow = O.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return O.grid_sample(input=tenInput, grid=O.clamp(g, -1, 1), mode='bilinear',
                         padding_mode='zeros', align_corners=True)

def multi_warp(img, flows, device='cuda'):
    return multi_backward_warp(img, flows, device)

def multi_backward_warp(img, flows, device='cuda'):
    num_flows = int(flows.shape[1] // 2)
    warped_imgs = []
    for i in range(num_flows):
        warped_imgs.append(warp(img, flows[:, 2 * i:2 * (i + 1)], device))
    warped_imgs = O.cat(warped_imgs, axis=1)
    return warped_imgs
