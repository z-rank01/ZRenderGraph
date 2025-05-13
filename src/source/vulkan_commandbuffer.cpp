#include "vulkan_commandbuffer.h"

VulkanCommandBufferHelper::VulkanCommandBufferHelper()
{
}

VulkanCommandBufferHelper::~VulkanCommandBufferHelper()
{
    // destroy command buffers
    for (auto command_buffer : command_buffer_map_)
    {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer.second);
    }
    command_buffer_map_.clear();

    // destroy command pool
    if (command_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
}

VkCommandBuffer VulkanCommandBufferHelper::GetCommandBuffer(std::string id) const
{
    if (command_buffer_map_.find(id) != command_buffer_map_.end())
    {
        return command_buffer_map_.at(id);
    }
    return VK_NULL_HANDLE;
}

bool VulkanCommandBufferHelper::CreateCommandPool(VkDevice device, uint32_t queue_family_index)
{
    device_ = device;

    // create command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    return Logger::LogWithVkResult(
        vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
        "Failed to create command pool",
        "Succeeded in creating command pool");
}

bool VulkanCommandBufferHelper::AllocateCommandBuffer(const SVulkanCommandBufferAllocationConfig& config, std::string id)
{
    // check if command buffer already exists
    if (command_buffer_map_.count(id))
    {
        Logger::LogError("Command buffer with ID " + id + " already exists. Allocation failed.");
        // 返回 false 表示因为 ID 重复分配失败
        return false;
    }

    // allocate command buffer
    VkCommandBuffer command_buffer;
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = config.command_buffer_level;
    alloc_info.commandBufferCount = config.command_buffer_count;

    if (!Logger::LogWithVkResult(
        vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer),
        "Failed to allocate command buffer",
        "Succeeded in allocating command buffer"))
    {
        return false; // 分配失败才返回 false
    }
    command_buffer_map_[id] = command_buffer;
    return true; // 分配成功返回 true
}

bool VulkanCommandBufferHelper::BeginCommandBufferRecording(std::string id, VkCommandBufferUsageFlags usage_flags)
{
    // begin command buffer recording
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = usage_flags;
    begin_info.pInheritanceInfo = nullptr;
    VkCommandBuffer command_buffer = command_buffer_map_.at(id);

    return Logger::LogWithVkResult(
        vkBeginCommandBuffer(command_buffer, &begin_info),
        "Failed to begin command buffer recording",
        "Succeeded in beginning command buffer recording");
}

bool VulkanCommandBufferHelper::EndCommandBufferRecording(std::string id)
{
    // end command buffer recording
    return Logger::LogWithVkResult(
        vkEndCommandBuffer(command_buffer_map_.at(id)),
        "Failed to end command buffer recording",
        "Succeeded in ending command buffer recording");
}

bool VulkanCommandBufferHelper::ResetCommandBuffer(std::string id)
{
    // reset command buffer
    return Logger::LogWithVkResult(
        vkResetCommandBuffer(command_buffer_map_.at(id), 0),
        "Failed to reset command buffer",
        "Succeeded in resetting command buffer");
}
