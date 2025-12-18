#include "MultiscaleDeformableAttnPlugin.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr const char* MULTISCALE_DEFORMABLE_ATTN_PLUGIN_VERSION{"1"};
constexpr const char* MULTISCALE_DEFORMABLE_ATTN_PLUGIN_NAME{"MultiscaleDeformableAttnPlugin"};
}

// Forward declaration of CUDA kernel
void multiscale_deformable_attn_cuda_forward(
    const float* value, const float* spatial_shapes, const float* level_start_index,
    const float* sampling_loc, const float* attn_weight, float* output,
    int batch, int num_queries, int num_heads, int channels, int num_levels, int num_points,
    cudaStream_t stream);

// Plugin implementation
MultiscaleDeformableAttnPlugin::MultiscaleDeformableAttnPlugin(
    int num_heads, int num_levels, int num_points)
    : num_heads_(num_heads), num_levels_(num_levels), num_points_(num_points) {}

MultiscaleDeformableAttnPlugin::MultiscaleDeformableAttnPlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    const char* const a = d;

    num_heads_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    num_levels_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    num_points_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    assert(d == a + length);
}

MultiscaleDeformableAttnPlugin::~MultiscaleDeformableAttnPlugin() {
    terminate();
}

IPluginV2DynamicExt* MultiscaleDeformableAttnPlugin::clone() const noexcept {
    auto* plugin = new MultiscaleDeformableAttnPlugin(num_heads_, num_levels_, num_points_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

DimsExprs MultiscaleDeformableAttnPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    // Output shape: [batch, num_queries, num_heads * channels]
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0]; // batch
    output.d[1] = inputs[0].d[1]; // num_queries
    output.d[2] = inputs[0].d[2]; // num_heads * channels
    return output;
}

bool MultiscaleDeformableAttnPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    // All inputs and outputs must be FP32 and linear format
    return (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR);
}

void MultiscaleDeformableAttnPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    // Configuration is done at enqueue time
}

size_t MultiscaleDeformableAttnPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0; // No additional workspace required
}

int MultiscaleDeformableAttnPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept {
    // Input tensors:
    // 0: value - [batch, num_value, num_heads, channels]
    // 1: spatial_shapes - [num_levels, 2]
    // 2: level_start_index - [num_levels]
    // 3: sampling_loc - [batch, num_queries, num_heads, num_levels, num_points, 2]
    // 4: attn_weight - [batch, num_queries, num_heads, num_levels, num_points]

    const float* value = static_cast<const float*>(inputs[0]);
    const float* spatial_shapes = static_cast<const float*>(inputs[1]);
    const float* level_start_index = static_cast<const float*>(inputs[2]);
    const float* sampling_loc = static_cast<const float*>(inputs[3]);
    const float* attn_weight = static_cast<const float*>(inputs[4]);
    float* output = static_cast<float*>(outputs[0]);

    int batch = inputDesc[0].dims.d[0];
    int num_queries = inputDesc[3].dims.d[1];
    int channels = inputDesc[0].dims.d[3];

    multiscale_deformable_attn_cuda_forward(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output,
        batch, num_queries, num_heads_, channels, num_levels_, num_points_, stream);

    return 0;
}

DataType MultiscaleDeformableAttnPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept {
    return DataType::kFLOAT;
}

const char* MultiscaleDeformableAttnPlugin::getPluginType() const noexcept {
    return MULTISCALE_DEFORMABLE_ATTN_PLUGIN_NAME;
}

const char* MultiscaleDeformableAttnPlugin::getPluginVersion() const noexcept {
    return MULTISCALE_DEFORMABLE_ATTN_PLUGIN_VERSION;
}

int MultiscaleDeformableAttnPlugin::getNbOutputs() const noexcept {
    return 1;
}

int MultiscaleDeformableAttnPlugin::initialize() noexcept {
    return 0;
}

void MultiscaleDeformableAttnPlugin::terminate() noexcept {}

size_t MultiscaleDeformableAttnPlugin::getSerializationSize() const noexcept {
    return sizeof(int) * 3; // num_heads, num_levels, num_points
}

void MultiscaleDeformableAttnPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = num_heads_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = num_levels_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = num_points_;
    d += sizeof(int);
}

void MultiscaleDeformableAttnPlugin::destroy() noexcept {
    delete this;
}

void MultiscaleDeformableAttnPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* MultiscaleDeformableAttnPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// Plugin Creator implementation
MultiscaleDeformableAttnPluginCreator::MultiscaleDeformableAttnPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_levels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_points", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

MultiscaleDeformableAttnPluginCreator::~MultiscaleDeformableAttnPluginCreator() {}

const char* MultiscaleDeformableAttnPluginCreator::getPluginName() const noexcept {
    return MULTISCALE_DEFORMABLE_ATTN_PLUGIN_NAME;
}

const char* MultiscaleDeformableAttnPluginCreator::getPluginVersion() const noexcept {
    return MULTISCALE_DEFORMABLE_ATTN_PLUGIN_VERSION;
}

const PluginFieldCollection* MultiscaleDeformableAttnPluginCreator::getFieldNames() noexcept {
    return &mFieldCollection;
}

IPluginV2* MultiscaleDeformableAttnPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept {
    int num_heads = 8;
    int num_levels = 4;
    int num_points = 4;

    for (int i = 0; i < fc->nbFields; ++i) {
        std::string field_name(fc->fields[i].name);
        if (field_name == "num_heads") {
            num_heads = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "num_levels") {
            num_levels = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "num_points") {
            num_points = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    auto* plugin = new MultiscaleDeformableAttnPlugin(num_heads, num_levels, num_points);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

IPluginV2* MultiscaleDeformableAttnPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
    auto* plugin = new MultiscaleDeformableAttnPlugin(serialData, serialLength);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

void MultiscaleDeformableAttnPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* MultiscaleDeformableAttnPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// Register the plugin creator
REGISTER_TENSORRT_PLUGIN(MultiscaleDeformableAttnPluginCreator);
