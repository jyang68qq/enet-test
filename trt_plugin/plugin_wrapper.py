"""
Python wrapper for MultiscaleDeformableAttnPlugin TensorRT plugin.

This module provides a Python interface for loading and using the 
MultiscaleDeformableAttnPlugin in TensorRT inference pipelines.
"""

import ctypes
import os
import tensorrt as trt


def load_plugin_library(lib_path=None):
    """
    Load the MultiscaleDeformableAttnPlugin shared library.
    
    Args:
        lib_path: Path to the plugin library. If None, searches in common locations.
    
    Returns:
        Handle to the loaded library
    
    Raises:
        FileNotFoundError: If the library cannot be found
    """
    if lib_path is None:
        # Search in common locations
        search_paths = [
            "./lib/libMultiscaleDeformableAttnPlugin.so",
            "../lib/libMultiscaleDeformableAttnPlugin.so",
            "/usr/local/lib/libMultiscaleDeformableAttnPlugin.so",
            "./build/lib/libMultiscaleDeformableAttnPlugin.so",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                lib_path = path
                break
        
        if lib_path is None:
            raise FileNotFoundError(
                "Could not find libMultiscaleDeformableAttnPlugin.so. "
                "Please specify the path explicitly or build the plugin first."
            )
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Plugin library not found at: {lib_path}")
    
    # Load the library
    handle = ctypes.CDLL(lib_path)
    print(f"Successfully loaded plugin library from: {lib_path}")
    return handle


def get_plugin_creator(plugin_registry, plugin_name="MultiscaleDeformableAttnPlugin", 
                       plugin_version="1", plugin_namespace=""):
    """
    Get the plugin creator from TensorRT plugin registry.
    
    Args:
        plugin_registry: TensorRT plugin registry
        plugin_name: Name of the plugin
        plugin_version: Version of the plugin
        plugin_namespace: Namespace of the plugin
    
    Returns:
        Plugin creator object
    """
    creator = plugin_registry.get_plugin_creator(
        plugin_name, plugin_version, plugin_namespace
    )
    
    if creator is None:
        raise RuntimeError(
            f"Could not find plugin creator for {plugin_name} v{plugin_version}"
        )
    
    return creator


def create_plugin(num_heads=8, num_levels=4, num_points=4, 
                  plugin_name="MultiscaleDeformableAttn"):
    """
    Create an instance of MultiscaleDeformableAttnPlugin.
    
    Args:
        num_heads: Number of attention heads
        num_levels: Number of feature pyramid levels
        num_points: Number of sampling points per level
        plugin_name: Name for the plugin instance
    
    Returns:
        Plugin instance
    """
    # Get plugin registry
    plugin_registry = trt.get_plugin_registry()
    
    # Get plugin creator
    creator = get_plugin_creator(plugin_registry)
    
    # Create field collection
    field_collection = []
    
    field_collection.append(
        trt.PluginField("num_heads", 
                       [num_heads],
                       trt.PluginFieldType.INT32)
    )
    field_collection.append(
        trt.PluginField("num_levels",
                       [num_levels],
                       trt.PluginFieldType.INT32)
    )
    field_collection.append(
        trt.PluginField("num_points",
                       [num_points],
                       trt.PluginFieldType.INT32)
    )
    
    # Create plugin
    plugin = creator.create_plugin(plugin_name, trt.PluginFieldCollection(field_collection))
    
    if plugin is None:
        raise RuntimeError(f"Failed to create plugin: {plugin_name}")
    
    return plugin


def add_plugin_to_network(network, inputs, num_heads=8, num_levels=4, num_points=4,
                          layer_name="multiscale_deformable_attn"):
    """
    Add MultiscaleDeformableAttnPlugin layer to a TensorRT network.
    
    Args:
        network: TensorRT network
        inputs: List of input tensors [value, spatial_shapes, level_start_index, 
                                       sampling_loc, attn_weight]
        num_heads: Number of attention heads
        num_levels: Number of feature pyramid levels
        num_points: Number of sampling points per level
        layer_name: Name for the layer
    
    Returns:
        Output tensor from the plugin layer
    """
    # Create plugin instance
    plugin = create_plugin(num_heads, num_levels, num_points)
    
    # Add plugin layer to network
    layer = network.add_plugin_v2(inputs, plugin)
    layer.name = layer_name
    
    return layer.get_output(0)


# Example usage
if __name__ == "__main__":
    import sys
    
    print("MultiscaleDeformableAttnPlugin Python Wrapper")
    print("=" * 50)
    
    try:
        # Load the plugin library
        lib_path = sys.argv[1] if len(sys.argv) > 1 else None
        handle = load_plugin_library(lib_path)
        
        print("\n✓ Plugin library loaded successfully!")
        print("\nTo use this plugin in your TensorRT pipeline:")
        print("  1. Import this module: from plugin_wrapper import add_plugin_to_network")
        print("  2. Load the plugin library: load_plugin_library()")
        print("  3. Add to network: add_plugin_to_network(network, inputs, ...)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nUsage: python plugin_wrapper.py [path/to/libMultiscaleDeformableAttnPlugin.so]")
        sys.exit(1)
