import mindspore.ops as O
import math
import numpy as np

def generate_2D_grid(H, W):
    x = np.arange(0, W, 1).astype(np.float32)
    y = np.arange(0, H, 1).astype(np.float32)

    xx = np.tile(x,(H, 1))
    yy = np.tile(y.reshape(H, 1),(1, W))

    grid = np.stack([xx, yy], axis=0)

    return grid  # (2,H,W)


def distortion_map(h, w, ref_row, reverse=False):
    grid_row = generate_2D_grid(h, w)[1].astype(np.float32)
    mask = grid_row / (h - 1)
    if reverse:
        mask *= -1.
        ref_row_floor = math.floor(h - 1 - ref_row)
        mask = mask - mask[int(ref_row_floor)] + (h - 1 - ref_row - ref_row_floor) * (1. / (h - 1))
    else:
        ref_row_floor = math.floor(ref_row)
        mask = mask - mask[int(ref_row_floor)] - (ref_row - ref_row_floor) * (1. / (h - 1))

    return mask

def time_map(h, w, ref_row, reverse=False):
    grid_row = generate_2D_grid(h, w)[1].astype(np.float32)
    mask = grid_row / ref_row
    formask = mask.copy()
    backmask = mask.copy()
    warpmask = mask.copy()

    warpmask[warpmask <= 1] = 1
    warpmask[warpmask > 1] = 0

    if reverse:
        formask[formask > 1] = 1
        formask = np.flip(formask, [0])

        backmask[backmask < 1] = -1
        backmask[backmask >= 1] -= 1
        backmask = backmask / backmask.max()
        backmask[backmask < 0] = 0
        backmask = np.flip(backmask, [0])
        warpmask = np.flip(warpmask, [0])
    else:
        formask[formask > 1] = 1
        '''
        0,0,0,
        1/ref_row, 1/ref_row, 1/ref_row
        ...
        1,1,1
        ...
        1,1,1
        '''
        backmask[backmask < 1] = 0
        backmask[backmask >= 1] -= 1
        backmask = backmask / backmask.max()
        '''
            0,0,0,
            ...
            0,0,0
            1/(h-ref_row), 1/(h-ref_row), 1/(h-ref_row)
            ...
            1,1,1
        '''

    return np.stack([formask[..., np.newaxis], backmask[..., np.newaxis], warpmask[..., np.newaxis]], axis=0)

if __name__ == '__main__':
    mask = distortion_map(h=256, w=256, ref_row=0)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255 / 2)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255)
    print('mask', mask)
    ## reverse: scanning from bottom to up
    mask = distortion_map(h=256, w=256, ref_row=0, reverse=True)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255 / 2, reverse=True)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255, reverse=True)
    print('mask', mask)
