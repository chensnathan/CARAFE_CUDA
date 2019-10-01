// author: Qiang Chen, Weihan Chen

#include <torch/extension.h>

#include <cmath>
#include <vector>

// CUDA forward Laucher declaration for CARAFE

int CarafeLayerIm2colForwardLaucher(
    const torch::Tensor input,
    const torch::Tensor kernel_map,
    torch::Tensor output,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w
);

// CUDA input backward Laucher declaration for CARAFE

int CarafeLayerIm2colInputBackwardLaucher(
    const torch::Tensor kernel_map,
    const torch::Tensor grad_output,
    torch::Tensor grad_input,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w
);

// CUDA kernel_map backward Laucher declaration for CARAFE

int CarafeLayerIm2colKernelmapBackwardLaucher(
    const torch::Tensor input,
    const torch::Tensor grad_output,
    torch::Tensor grad_kernel_map,
    const int num_channels,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w
);

// Check utils

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), "x must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), "x must be contiguous")
#define CHECK_INPUT(X) CHECK_CUDA(X); CHECK_CONTIGUOUS(X)


int carafe_layer_forward_cuda(
    const torch::Tensor input,
    const torch::Tensor kernel_map,
    torch::Tensor output,
    const int up_factor,
    const int kernel_h,
    const int kernel_w) {

    CHECK_INPUT(input);
    CHECK_INPUT(kernel_map);
    CHECK_INPUT(output);
    const int orig_height = input.size(2);
    const int orig_width = input.size(3);
    if (orig_height < 1 || orig_width < 1) AT_ERROR("Input size is zero.");
    // TODO: make padding customizable
    const int pad_h = kernel_h / 2;
    const int pad_w = kernel_w / 2;
    // reshape output
    const int batch_size = output.size(0);
    const int num_channels = output.size(1);
    const int up_height = output.size(2);
    const int up_width = output.size(3);
    if (up_height != orig_height * up_factor || up_width != orig_width * up_factor) {
        AT_ERROR("Input shape with up_factor and output shape wont match: (%d x %d, %d vs %d x %d).",
             orig_height, orig_width, up_factor, up_height, up_width);
    }

    output = output.view({batch_size * num_channels, up_height * up_width});

    CarafeLayerIm2colForwardLaucher(
        input, kernel_map, output, orig_height, orig_width,
        up_factor, kernel_h, kernel_w, pad_h, pad_w
    );

    // reshape output back
    output = output.view({batch_size, num_channels, up_height, up_width});

    return 1;
}


int carafe_layer_input_backward_cuda(
    const torch::Tensor kernel_map,
    const torch::Tensor grad_output,
    torch::Tensor grad_input,
    const int up_factor,
    const int kernel_h,
    const int kernel_w) {

    CHECK_INPUT(kernel_map);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_input);
    const int orig_height = grad_output.size(2) / up_factor;
    const int orig_width = grad_output.size(3) / up_factor;
    // TODO: make padding customizable
    const int pad_h = kernel_h / 2;
    const int pad_w = kernel_w / 2;
    // reshpe grad_input
    const int batch_size = grad_input.size(0);
    const int num_channels = grad_input.size(1);
    const int height = grad_input.size(2);
    const int width = grad_input.size(3);
    if (height != orig_height || width != orig_width) {
        AT_ERROR("Input shape and output shape divide up_factor wont match: (%d x %d vs %d x %d).",
             height, width, orig_height, orig_width);
    }

    grad_input = grad_input.view({batch_size * num_channels, height * width});

    CarafeLayerIm2colInputBackwardLaucher(
        kernel_map, grad_output, grad_input, orig_height, orig_width,
        up_factor, kernel_h, kernel_w, pad_h, pad_w
    );

    // reshape grad_input back
    grad_input = grad_input.view({batch_size, num_channels, height, width});

    return 1;
}


int carafe_layer_kernel_map_backward_cuda(
    const torch::Tensor input,
    const torch::Tensor grad_output,
    torch::Tensor grad_kernel_map,
    const int up_factor,
    const int kernel_h,
    const int kernel_w) {

    CHECK_INPUT(input);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_kernel_map);
    const int num_channels = input.size(1);
    const int orig_height = input.size(2);
    const int orig_width = input.size(3);
    // TODO: make padding customizable
    const int pad_h = kernel_h / 2;
    const int pad_w = kernel_w / 2;
    // reshape grad_kernel_map
    const int batch_size = grad_kernel_map.size(0);
    const int kernel_area = grad_kernel_map.size(1);
    const int up_height = grad_kernel_map.size(2);
    const int up_width = grad_kernel_map.size(3);
    if (up_height != orig_height * up_factor || up_width != orig_width * up_factor) {
        AT_ERROR("Input shape with up_factor and output shape wont match: (%d x %d, %d vs %d x %d).",
             orig_height, orig_width, up_factor, up_height, up_width);
    }

    grad_kernel_map = grad_kernel_map.view({batch_size * kernel_area, up_height * up_width});

    CarafeLayerIm2colKernelmapBackwardLaucher(
        input, grad_output, grad_kernel_map, num_channels, orig_height,
        orig_width, up_factor, kernel_h, kernel_w, pad_h, pad_w
    );

    // reshape grad_kernel_map back
    grad_kernel_map = grad_kernel_map.view({batch_size, kernel_area, up_height, up_width});

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("carafe_layer_forward", &carafe_layer_forward_cuda, "carafe_layer_forward (CUDA)");
    m.def("carafe_layer_input_backward", &carafe_layer_input_backward_cuda, "carafe_layer_input_backward (CUDA)");
    m.def("carafe_layer_kernel_map_backward", &carafe_layer_kernel_map_backward_cuda, "carafe_layer_kernel_map_backward (CUDA)");
}