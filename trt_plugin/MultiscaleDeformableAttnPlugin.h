#ifndef MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H
#define MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H

#include <NvInfer.h>
#include <string>
#include <vector>

namespace nvinfer1 {
namespace plugin {

class MultiscaleDeformableAttnPlugin : public IPluginV2DynamicExt {
public:
    MultiscaleDeformableAttnPlugin(int num_heads, int num_levels, int num_points);
    MultiscaleDeformableAttnPlugin(const void* data, size_t length);
    ~MultiscaleDeformableAttnPlugin() override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                   IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                    int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                         const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                            const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    int num_heads_;
    int num_levels_;
    int num_points_;
    std::string namespace_;
};

class MultiscaleDeformableAttnPluginCreator : public IPluginCreator {
public:
    MultiscaleDeformableAttnPluginCreator();
    ~MultiscaleDeformableAttnPluginCreator() override;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    PluginFieldCollection mFieldCollection;
    std::vector<PluginField> mPluginAttributes;
    std::string namespace_;
};

} // namespace plugin
} // namespace nvinfer1

#endif // MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H
