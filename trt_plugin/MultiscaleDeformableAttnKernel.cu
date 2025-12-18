#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t* input, const int height, const int width,
    const int num_heads, const int channels,
    const scalar_t h, const scalar_t w, const int m, const int c) {
    
    if (h <= -1 || width <= h || w <= -1 || height <= w) {
        return 0;
    }

    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    scalar_t hh = 1 - lh;
    scalar_t hw = 1 - lw;

    const int w_stride = num_heads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = input[ptr1];
    }
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = input[ptr2];
    }
    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = input[ptr3];
    }
    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = input[ptr4];
    }

    scalar_t val = (hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4);
    return val;
}

template <typename scalar_t>
__global__ void ms_deformable_attn_cuda_kernel(
    const int n, const scalar_t* value, const scalar_t* spatial_shapes,
    const scalar_t* level_start_index, const scalar_t* sampling_loc,
    const scalar_t* attn_weight, scalar_t* output,
    const int batch, const int num_queries, const int num_heads,
    const int channels, const int num_levels, const int num_points) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_queries;
    _temp /= num_queries;
    const int b_col = _temp;

    scalar_t col = 0;

    const scalar_t* data_value_ptr = value + b_col * num_heads * channels;
    const int data_weight_ptr = sampling_index * num_levels * num_points;
    const int data_loc_w_ptr = data_weight_ptr << 1;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
        const int level_start_id = level_start_index[l_col];
        const int spatial_h_ptr = l_col << 1;
        const int spatial_h = spatial_shapes[spatial_h_ptr];
        const int spatial_w = spatial_shapes[spatial_h_ptr + 1];
        const scalar_t* value_ptr = data_value_ptr + (level_start_id * spatial_h * spatial_w * num_heads * channels);

        for (int p_col = 0; p_col < num_points; ++p_col) {
            const int loc_w_ptr = data_loc_w_ptr + l_col * num_points * 2 + p_col * 2;
            const scalar_t loc_w = sampling_loc[loc_w_ptr];
            const scalar_t loc_h = sampling_loc[loc_w_ptr + 1];
            const int weight_ptr = data_weight_ptr + l_col * num_points + p_col;
            const scalar_t weight = attn_weight[weight_ptr];

            const scalar_t h_im = loc_h * spatial_h - 0.5;
            const scalar_t w_im = loc_w * spatial_w - 0.5;

            if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                col += ms_deform_attn_im2col_bilinear(
                    value_ptr, spatial_h, spatial_w, num_heads, channels,
                    h_im, w_im, m_col, c_col) * weight;
            }
        }
    }
    output[index] = col;
}

void multiscale_deformable_attn_cuda_forward(
    const float* value, const float* spatial_shapes, const float* level_start_index,
    const float* sampling_loc, const float* attn_weight, float* output,
    int batch, int num_queries, int num_heads, int channels, int num_levels, int num_points,
    cudaStream_t stream) {
    
    const int num_kernels = batch * num_queries * num_heads * channels;
    const int num_threads = 256;
    const int num_blocks = (num_kernels + num_threads - 1) / num_threads;

    ms_deformable_attn_cuda_kernel<float><<<num_blocks, num_threads, 0, stream>>>(
        num_kernels, value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output,
        batch, num_queries, num_heads, channels, num_levels, num_points);
}
