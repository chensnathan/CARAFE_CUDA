// author: Qiang Chen, Weihan Chen

// #include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>

#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}


template <typename scalar_t>
__global__ void CarafeLayerIm2colForward(
    const int n,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> kernel_map,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w) {
    // num_channels
    const int num_channels = input.size(1);
    // output column: k_upH * k_upW
    const int total_column = output.size(1);
    // kernel
    CUDA_1D_KERNEL_LOOP (index, n) {
        // batch channel idx
        const int bc = index / total_column;
        const int column_index = index % total_column;
        const int b = bc / num_channels;
        const int c = bc % num_channels;
        // index_h, index_w in the output feature map
        const int index_h = column_index / (orig_width * up_factor);
        const int index_w = column_index % (orig_width * up_factor);
        // the corresponding source location
        const int orig_index_h = index_h / up_factor;
        const int orig_index_w = index_w / up_factor;
        // the left-top point in the corresponding patch
        const int h_offset = orig_index_h - pad_h;
        const int w_offset = orig_index_w - pad_w;
        for (int i = 0; i < kernel_h; ++i) {
            int h_im = h_offset + i;
            for (int j = 0; j < kernel_w; ++j) {
                int w_im = w_offset + j;
                if (h_im >= 0 && w_im >= 0 && h_im < orig_height && w_im < orig_width) {
                    output[bc][column_index] += input[b][c][h_im][w_im] * kernel_map[b][i * kernel_w + j][index_h][index_w];
                }
            }
        }
    }
}


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
    const int pad_w) {
        const int N = output.size(0) * output.size(1);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.type(), "CarafeLayerIm2colLaucherForward", ([&] {
            CarafeLayerIm2colForward<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK>>>(
                N,
                input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                kernel_map.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                orig_height, orig_width, up_factor, kernel_h, kernel_w, pad_h, pad_w);
        }));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("error in CarafeLayerIm2colForwardLaucher: %s\n", cudaGetErrorString(err));
        }
        return 1;
  }


template <typename scalar_t>
__global__ void CarafeLayerIm2colInputBackward(
    const int n,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> kernel_map,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_input,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w) {
    // num_channels
    const int num_channels = grad_output.size(1);
    // grad_input column: H * W
    const int total_column = grad_input.size(1);
    // kernel
    CUDA_1D_KERNEL_LOOP (index, n) {
         // batch channel idx
         const int bc = index / total_column;
         const int column_index = index % total_column;
         const int b = bc / num_channels;
         const int c = bc % num_channels;
        // index_h, index_w in the input feature map
        const int index_h = column_index / orig_width;
        const int index_w = column_index % orig_width;
        // the corresponding target location
        const int out_index_h = index_h * up_factor;
        const int out_index_w = index_w * up_factor;
        // the right-bottom point in the corresponding patch
        const int h_offset = out_index_h + (pad_h + 1) * up_factor - 1;
        const int w_offset = out_index_w + (pad_w + 1) * up_factor - 1;
        for (int i = 0; i < kernel_h * up_factor; ++i) {
            int h_im = h_offset - i;
            for (int j = 0; j < kernel_w * up_factor; ++j) {
                int w_im = w_offset - j;
                if (h_im >= 0 && w_im >= 0 && h_im < orig_height * up_factor && w_im < orig_width * up_factor && h_im <= h_offset && w_im <= w_offset) {
                    grad_input[bc][column_index] += grad_output[b][c][h_im][w_im] * kernel_map[b][(i / up_factor) * kernel_w + j / up_factor][h_im][w_im];
                }
            }
        }
    }
}

int CarafeLayerIm2colInputBackwardLaucher(
    const torch::Tensor kernel_map,
    const torch::Tensor grad_output,
    const torch::Tensor grad_input,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w) {
        const int N = grad_input.size(0) * grad_input.size(1);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.type(), "CarafeLayerIm2colInputLaucherBackward", ([&] {
                CarafeLayerIm2colInputBackward<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK>>>(
                N,
                kernel_map.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                grad_output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                grad_input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                orig_height, orig_width, up_factor, kernel_h, kernel_w, pad_h, pad_w);
        }));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("error in CarafeLayerIm2colInputBackwardLaucher: %s\n", cudaGetErrorString(err));
        }
        return 1;
}


template <typename scalar_t>
__global__ void CarafeLayerIm2colKernelmapBackward(
    const int n,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_kernel_map,
    const int num_channels,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w) {
    // grad_kernel_map column: k_upH * k_upW
    const int total_column = grad_kernel_map.size(1);
    // kernel
    CUDA_1D_KERNEL_LOOP (index, n) {
        // batch channel idx
        const int bc = index / total_column;
        const int column_index = index % total_column;

        const int b = bc / (kernel_h * kernel_w);
        const int kernel_index = bc % (kernel_h * kernel_w);
        const int cur_kernel_h = kernel_index / kernel_w;
        const int cur_kernel_w = kernel_index % kernel_w;
        // column idx, column: 2H * 2W
        // index_h, index_w in the output feature map
        const int index_h = column_index / (orig_width * up_factor);
        const int index_w = column_index % (orig_width * up_factor);
        // the corresponding source location
        const int orig_index_h = index_h / up_factor;
        const int orig_index_w = index_w / up_factor;
        // the left-top point in the corresponding patch
        const int h_offset = orig_index_h - pad_h;
        const int w_offset = orig_index_w - pad_w;
        const int h_im = h_offset + cur_kernel_h;
        const int w_im = w_offset + cur_kernel_w;
        if (h_im >= 0 && w_im >= 0 && h_im < orig_height && w_im < orig_width) {
            for (int c = 0; c < num_channels; ++c) {
                grad_kernel_map[bc][column_index] += grad_output[b][c][index_h][index_w] * input[b][c][h_im][w_im];
            }
        }
    }
}

int CarafeLayerIm2colKernelmapBackwardLaucher(
    const torch::Tensor input,
    const torch::Tensor grad_output,
    const torch::Tensor grad_kernel_map,
    const int num_channels,
    const int orig_height,
    const int orig_width,
    const int up_factor,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w) {
        const int N = grad_kernel_map.size(0) * grad_kernel_map.size(1);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.type(), "CarafeLayerIm2colKernelmapLaucherBackward", ([&] {
                CarafeLayerIm2colKernelmapBackward<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK>>>(
                N,
                input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                grad_output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                grad_kernel_map.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                num_channels, orig_height, orig_width, up_factor, kernel_h, kernel_w, pad_h, pad_w);
        }));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("error in CarafeLayerIm2colKernelmapBackwardLaucher: %s\n", cudaGetErrorString(err));
        }
        return 1;
}