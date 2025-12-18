#include <iostream>
#include <vector>
#include <NvInfer.h>
#include "MultiscaleDeformableAttnPlugin.h"

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

int main() {
    std::cout << "=== MultiscaleDeformableAttnPlugin Example ===" << std::endl;

    // Plugin parameters
    int num_heads = 8;
    int num_levels = 4;
    int num_points = 4;

    // Create plugin creator
    auto creator = new plugin::MultiscaleDeformableAttnPluginCreator();
    
    std::cout << "Plugin Name: " << creator->getPluginName() << std::endl;
    std::cout << "Plugin Version: " << creator->getPluginVersion() << std::endl;
    
    // Prepare plugin fields
    std::vector<PluginField> pluginData;
    pluginData.emplace_back(PluginField("num_heads", &num_heads, PluginFieldType::kINT32, 1));
    pluginData.emplace_back(PluginField("num_levels", &num_levels, PluginFieldType::kINT32, 1));
    pluginData.emplace_back(PluginField("num_points", &num_points, PluginFieldType::kINT32, 1));
    
    PluginFieldCollection fc;
    fc.nbFields = pluginData.size();
    fc.fields = pluginData.data();
    
    // Create plugin instance
    IPluginV2* plugin = creator->createPlugin("MultiscaleDeformableAttn", &fc);
    
    if (plugin) {
        std::cout << "Plugin created successfully!" << std::endl;
        std::cout << "Plugin Type: " << plugin->getPluginType() << std::endl;
        std::cout << "Number of Outputs: " << plugin->getNbOutputs() << std::endl;
        
        // Clone plugin
        auto* dynamicPlugin = dynamic_cast<IPluginV2DynamicExt*>(plugin);
        if (dynamicPlugin) {
            IPluginV2DynamicExt* clonedPlugin = dynamicPlugin->clone();
            std::cout << "Plugin cloned successfully!" << std::endl;
            clonedPlugin->destroy();
        }
        
        plugin->destroy();
        std::cout << "Plugin destroyed successfully!" << std::endl;
    } else {
        std::cerr << "Failed to create plugin!" << std::endl;
        return 1;
    }
    
    delete creator;
    
    std::cout << "\n=== Example completed successfully ===" << std::endl;
    return 0;
}
