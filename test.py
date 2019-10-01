import time

import torch
import torch.nn.functional as F

from carafe_layer import carafe_layer_function


def test1(input, kernel_map, kernel_size, up_factor=2):
    tic = time.time()
    output = carafe_layer_function(
        input, kernel_map, kernel_size, up_factor)
    toc = time.time()
    print("test1 cost time {}".format(toc - tic))
    return output


def test2(input, kernel_map, kernel_size, up_factor=2):
    b, c, h, w = input.size()
    _, kernel_area, uph, upw = kernel_map.size()
    tic = time.time()
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    x_unfold = F.unfold(input,
                        kernel_size=kernel_size,
                        padding=padding)
    x_unfold = x_unfold.reshape(b, c, kernel_area, h, w)
    out = F.interpolate(x_unfold,
                        scale_factor=(1, up_factor, up_factor),
                        mode='nearest')
    out = out * kernel_map.unsqueeze(1)
    out = out.sum(dim=2)
    toc = time.time()
    print("test2 cost time {}".format(toc - tic))
    return out


if __name__ == '__main__':
    input = torch.rand((2, 256, 100, 100)).cuda()
    kernel_map = torch.ones(2, 49, 200, 200).cuda()
    kernel_size = (7, 7)
    output1 = test1(input, kernel_map, kernel_size)
    output2 = test2(input, kernel_map, kernel_size)
    loss = torch.abs(output2 - output1).mean()
    print("loss: {}".format(loss))
