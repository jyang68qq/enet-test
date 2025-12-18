# MultiscaleDeformableAttnPlugin for TensorRT

This directory contains a TensorRT plugin implementation for Multiscale Deformable Attention mechanism, commonly used in transformer-based vision models like Deformable DETR.

## Overview

The Multiscale Deformable Attention mechanism is a key component in modern vision transformers that enables efficient attention computation across multiple feature scales. This plugin provides an optimized CUDA implementation for TensorRT inference.

## Features

- **Efficient CUDA kernels**: Optimized for NVIDIA GPUs
- **TensorRT integration**: Seamless integration with TensorRT inference pipelines
- **Dynamic shapes support**: Handles variable batch sizes and input dimensions
- **FP32 precision**: Support for float32 operations

## Architecture

The plugin consists of three main components:

1. **MultiscaleDeformableAttnPlugin.h**: Plugin interface definition
2. **MultiscaleDeformableAttnPlugin.cpp**: Plugin implementation and registration
3. **MultiscaleDeformableAttnKernel.cu**: CUDA kernels for deformable attention computation

## Input Tensors

The plugin expects 5 input tensors:

1. **value**: `[batch, num_value, num_heads, channels]` - Feature values
2. **spatial_shapes**: `[num_levels, 2]` - Spatial dimensions for each level
3. **level_start_index**: `[num_levels]` - Starting indices for each level
4. **sampling_loc**: `[batch, num_queries, num_heads, num_levels, num_points, 2]` - Sampling locations
5. **attn_weight**: `[batch, num_queries, num_heads, num_levels, num_points]` - Attention weights

## Output Tensor

The plugin produces 1 output tensor:

- **output**: `[batch, num_queries, num_heads * channels]` - Aggregated features

## Parameters

- **num_heads** (int): Number of attention heads
- **num_levels** (int): Number of feature pyramid levels
- **num_points** (int): Number of sampling points per level

## Building

### Prerequisites

- CUDA Toolkit (10.2 or later)
- TensorRT (7.0 or later)
- CMake (3.10 or later)
- C++14 compatible compiler

### Build Instructions

```bash
cd trt_plugin
mkdir build
cd build
cmake -DTENSORRT_ROOT=/path/to/TensorRT ..
make -j$(nproc)
```

The build system will automatically detect CUDA and attempt to locate TensorRT. If TensorRT is not found automatically, specify the path using the `TENSORRT_ROOT` CMake variable.

### Build Outputs

- `lib/libMultiscaleDeformableAttnPlugin.so`: The plugin shared library
- `bin/plugin_example` (optional): Example program demonstrating plugin usage

## Installation

After building, install the plugin:

```bash
sudo make install
```

This will install:
- Plugin library to `/usr/local/lib`
- Header files to `/usr/local/include`

## Usage

### In C++

```cpp
#include "MultiscaleDeformableAttnPlugin.h"

// Create plugin creator
auto creator = new nvinfer1::plugin::MultiscaleDeformableAttnPluginCreator();

// Set plugin parameters
int num_heads = 8;
int num_levels = 4;
int num_points = 4;

std::vector<nvinfer1::PluginField> pluginData;
pluginData.emplace_back("num_heads", &num_heads, nvinfer1::PluginFieldType::kINT32, 1);
pluginData.emplace_back("num_levels", &num_levels, nvinfer1::PluginFieldType::kINT32, 1);
pluginData.emplace_back("num_points", &num_points, nvinfer1::PluginFieldType::kINT32, 1);

nvinfer1::PluginFieldCollection fc;
fc.nbFields = pluginData.size();
fc.fields = pluginData.data();

// Create plugin instance
auto plugin = creator->createPlugin("MultiscaleDeformableAttn", &fc);
```

### In ONNX Model

The plugin can be integrated into ONNX models by adding a custom operator node:

```python
import onnx
from onnx import helper, TensorProto

# Create custom op node
node = helper.make_node(
    'MultiscaleDeformableAttnPlugin',
    inputs=['value', 'spatial_shapes', 'level_start_index', 'sampling_loc', 'attn_weight'],
    outputs=['output'],
    domain='',
    num_heads=8,
    num_levels=4,
    num_points=4
)
```

## Algorithm

The multiscale deformable attention computes:

```
output[b,q,m,c] = Σ_l Σ_p weight[b,q,m,l,p] × value[b, Φ(q,m,l,p), m, c]
```

Where:
- `Φ(q,m,l,p)` represents bilinear sampling at learned offsets
- The summation is over all levels `l` and sampling points `p`
- Bilinear interpolation is used for sub-pixel sampling

## Performance

The CUDA kernel uses:
- Bilinear interpolation for sampling
- Coalesced memory access patterns
- Optimized thread block configuration (256 threads per block)

## Limitations

- Currently supports FP32 precision only
- Requires CUDA-capable GPU
- All input tensors must use linear (NCHW-like) format

## References

- [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

## License

See the repository LICENSE file for license information.

## Contributing

Contributions are welcome! Please ensure all code follows the existing style and includes appropriate tests.
