#pragma once

#include "initialization/vulkan_instance.h"
#include "initialization/vulkan_device.h"
#include "initialization/vulkan_queue.h"
#include "initialization/vulkan_window.h"

#include "pipeline/vulkan_shader.h"
#include "pipeline/vulkan_pipeline.h"
#include "pipeline/vulkan_renderpass.h"

#include "source/vulkan_commandbuffer.h"
#include "source/vulkan_framebuffer.h"

#include "synchronization/vulkan_synchronization.h"
#include "utility/config_reader.h"
#include "utility/logger.h"

#include "vulkan_resource_allocator/vra.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <memory>
#include <VkBootstrap.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

enum class EWindowState : std::uint8_t
{
    Initialized, // Engine is initialized but not running
    Running,     // Engine is running
    Stopped      // Engine is stopped
};

enum class ERenderState : std::uint8_t
{
    True,   // Render is enabled 
    False   // Render is disabled
};

struct SWindowConfig
{
    int width;
    int height;
    std::string title;

    [[nodiscard]] constexpr auto Validate() const -> bool
    {
        return width > 0 && height > 0;
    }
};

struct SEngineConfig
{
    SWindowConfig window_config;
    SGeneralConfig general_config;
    uint8_t frame_count;
    bool use_validation_layers;
};

struct SOutputFrame
{
    uint32_t image_index;
    std::string queue_id;
    std::string command_buffer_id;
    std::string image_available_sempaphore_id;
    std::string render_finished_sempaphore_id;
    std::string fence_id;
};

struct SMvpMatrix
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

struct SCamera
{
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;
    float yaw;
    float pitch;
    float movement_speed;
    float mouse_sensitivity;
    float zoom;
    
    // 聚焦点相关
    glm::vec3 focus_point;
    bool has_focus_point;
    float focus_distance;
    float min_focus_distance;
    float max_focus_distance;

    SCamera(glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
            float initial_yaw = -90.0f,
            float initial_pitch = 0.0f)
        : position(pos), world_up(up), yaw(initial_yaw), pitch(initial_pitch),
          movement_speed(2.5f), mouse_sensitivity(0.1f), zoom(45.0f)
    {
        UpdateCameraVectors();
    }

    void UpdateCameraVectors()
    {
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        // 在Vulkan坐标系中计算相机方向：+X向右，+Y向下，+Z向屏幕外
        glm::vec3 new_front;
        new_front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        new_front.y = sin(glm::radians(pitch)); // Y轴向下
        new_front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(new_front);
        
        // 计算右向量和上向量
        right = glm::normalize(glm::cross(front, world_up));
        up = glm::normalize(glm::cross(right, front));
    }
};

class VulkanEngine
{
public:
    VulkanEngine() = delete;
    VulkanEngine(const SEngineConfig& config);
    ~VulkanEngine();
    
    void Run();
    void Draw();

    static VulkanEngine& GetInstance();

private:
#define FRAME_INDEX_TO_UNIFORM_BUFFER_ID(frame_index) (frame_index + 4)
    // engine members
    uint8_t frame_index_ = 0;
    bool resize_request_ = false;
    EWindowState engine_state_;
    ERenderState render_state_;
    SEngineConfig engine_config_;
    SCamera camera_;
    std::vector<SOutputFrame> output_frames_;

    // vulkan bootstrap members
    vkb::Instance vkb_instance_;
    vkb::PhysicalDevice vkb_physical_device_;
    vkb::Device vkb_device_;
    vkb::Swapchain vkb_swapchain_;

    SVulkanSwapChainConfig swapchain_config_;

    // vra and vma members
    VmaAllocator vma_allocator_;
    VmaAllocation local_buffer_allocation_;
    VmaAllocation staging_buffer_allocation_;
    VmaAllocation uniform_buffer_allocation_;
    VmaAllocationInfo local_buffer_allocation_info_;
    VmaAllocationInfo staging_buffer_allocation_info_;
    VmaAllocationInfo uniform_buffer_allocation_info_;
    std::unique_ptr<vra::VraDataBatcher> vra_data_batcher_;

    // vra resource id
    std::vector<vra::ResourceId> vertex_index_data_id_;
    std::vector<vra::ResourceId> staging_data_id_;
    std::vector<vra::ResourceId> uniform_buffer_id_;

    // vra resource type name
    std::string vertex_index_data_type_name_ = "vertex_index_data";
    std::string staging_data_type_name_ = "staging_data";
    std::string uniform_buffer_type_name_ = "uniform_buffer";

    // vra data batcher output
    const vra::VraDataBatcher::VraBatchHandle *vertex_index_data_batch_;
    const vra::VraDataBatcher::VraBatchHandle *staging_data_batch_;
    const vra::VraDataBatcher::VraBatchHandle *uniform_buffer_batch_;

    // vulkan native members
    VkBuffer local_buffer_;
    VkBuffer staging_buffer_;
    VkBuffer uniform_buffer_;
    VkDescriptorPool descriptor_pool_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkDescriptorSet descriptor_set_;
    VkVertexInputBindingDescription vertex_input_binding_description_;
    VkVertexInputAttributeDescription vertex_input_attribute_position_;
    VkVertexInputAttributeDescription vertex_input_attribute_color_;

    // vulkan helper members
    std::unique_ptr<VulkanSDLWindowHelper> vkWindowHelper_;
    std::unique_ptr<VulkanShaderHelper> vkShaderHelper_;
    std::unique_ptr<VulkanRenderpassHelper> vkRenderpassHelper_;
    std::unique_ptr<VulkanPipelineHelper> vkPipelineHelper_;
    std::unique_ptr<VulkanCommandBufferHelper> vkCommandBufferHelper_;
    std::unique_ptr<VulkanFrameBufferHelper> vkFrameBufferHelper_;
    std::unique_ptr<VulkanSynchronizationHelper> vkSynchronizationHelper_;
    
    // uniform data
    std::vector<SMvpMatrix> mvp_matrices_;
    void * uniform_buffer_mapped_data_;
    
    // Input handling members
    float last_x_ = 0.0f;
    float last_y_ = 0.0f;
    bool camera_rotation_mode_ = false;  // 相机旋转模式标志（右键）
    bool camera_pan_mode_ = false;       // 相机平移模式标志（中键）
    float orbit_distance_ = 0.0f;      // 轨道旋转时与中心的距离

    void InitializeSDL();
    void InitializeVulkan();
    void InitializeCamera();

    // --- Vulkan Initialization Steps ---
    void GenerateFrameStructs();
    bool CreateInstance();
    bool CreateSurface();
    bool CreatePhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateSwapChain();
    bool CreatePipeline();
    bool CreateFrameBuffer();
    bool CreateCommandPool();
    bool CreateAndWriteDescriptorRelatives();
    bool CreateVertexInputBuffers();
    bool CreateUniformBuffers();
    bool AllocatePerFrameCommandBuffer();
    bool CreateSynchronizationObjects();
    // ------------------------------------

    // --- Vulkan Draw Steps ---
    void DrawFrame();
    void ResizeSwapChain();
    bool RecordCommand(uint32_t image_index, std::string command_buffer_id);
    void UpdateUniformBuffer(uint32_t current_frame_index);
    // -------------------------

    // --- camera control ---
    void ProcessInput(SDL_Event& event);
    void ProcessKeyboardInput(float delta_time);
    void ProcessMouseScroll(float yoffset);
    void FocusOnObject(const glm::vec3& object_position, float target_distance);
};
