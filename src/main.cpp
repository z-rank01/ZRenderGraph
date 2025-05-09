#include "vulkan_engine.h"
#include "utility/config_reader.h"
#include "utility/logger.h"
#include <iostream>

int main()
{
    std::cout << "Hello, World!" << '\n';
    std::cout << "This is a Vulkan Engine" << '\n';

    // window config
    SWindowConfig window_config;
    const auto window_width = 800;
    const auto window_height = 600;
    const auto* window_name = "Vulkan Engine";

    window_config.width = window_width;
    window_config.height = window_height;
    window_config.title = window_name;

    // general config
    ConfigReader config_reader("E:\\Projects\\ZRenderGraph\\config\\win64\\app_config.json");
    SGeneralConfig general_config;
    if (!config_reader.TryParseGeneralConfig(general_config))
    {
        Logger::LogError("Failed to parse general config");
        return -1;
    }

    // engine config
    SEngineConfig config;
    config.window_config = window_config;
    config.general_config = general_config;
    config.frame_count = 2;
    config.use_validation_layers = true;
    

    // main loop
    VulkanEngine engine(config);
    engine.Run();

    std::cout << "Goodbye" << '\n';

    return 0;
}