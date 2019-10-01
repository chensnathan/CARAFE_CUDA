from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn
from torch.nn.modules.utils import _pair

from . import carafe_layer_cuda


class CarafeLayer(Function):

    @staticmethod
    def forward(ctx, input, kernel_map, kernel_size, up_factor=2):
        assert input.dim() == 4
        assert kernel_map.dim() == 4
        assert input.size(0) == kernel_map.size(0)
        assert input.size(2) * up_factor == kernel_map.size(2)
        assert input.size(3) * up_factor == kernel_map.size(3)
        kernel_h, kernel_w = _pair(kernel_size)
        assert kernel_map.size(1) == kernel_h * kernel_w

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.up_factor = up_factor
        ctx.save_for_backward(input, kernel_map)

        if not input.is_cuda:
            raise NotImplementedError

        batch_size, in_channels, height, width = input.size()
        _, _, up_height, up_width = kernel_map.size()

        outputs = input.new_zeros(
            batch_size, in_channels, up_height, up_width)
        carafe_layer_cuda.carafe_layer_forward(
            input, kernel_map, outputs, up_factor, kernel_h, kernel_w)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_outputs):
        input, kernel_map = ctx.saved_tensors

        grad_input, grad_kernel_map = None, None

        if not grad_outputs.is_cuda:
            raise NotImplementedError

        if ctx.needs_input_grad[0]:
            grad_input = input.new_zeros(input.size())
            carafe_layer_cuda.carafe_layer_input_backward(
                kernel_map, grad_outputs, grad_input, ctx.up_factor,
                ctx.kernel_h, ctx.kernel_w)

        if ctx.needs_input_grad[1]:
            grad_kernel_map = kernel_map.new_zeros(kernel_map.size())
            carafe_layer_cuda.carafe_layer_kernel_map_backward(
                input, grad_outputs, grad_kernel_map, ctx.up_factor,
                ctx.kernel_h, ctx.kernel_w)
        return grad_input, grad_kernel_map, None, None


carafe_layer_function = CarafeLayer.apply


class CarafeLayer(nn.Module):
    """CARAFE Layer with cuda implementation."""

    def __init__(self, kernel_size, up_factor=2):
        super(CarafeLayer, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor

    def forward(self, input, kernel_map):
        return carafe_layer_function(
            input, kernel_map, self.kernel_size, self.up_factor)
