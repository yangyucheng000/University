from fairseq.utils import softmax
import mindspore as ms

x = ms.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ms.float32)
sm = softmax(x, -1, True)
print(sm)