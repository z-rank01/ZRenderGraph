#include "vulkan_framebuffer.h"

VulkanFrameBufferHelper::~VulkanFrameBufferHelper()
{
    if (!framebuffers_.empty())
    {
        for (auto framebuffer : framebuffers_)
        {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
        framebuffers_.clear();
    }
}

bool VulkanFrameBufferHelper::CreateFrameBuffer(VkDevice device, VkRenderPass renderpass)
{
    device_ = device;

    // create framebuffers
    framebuffers_.resize(config_.image_views->size());
    for (size_t i = 0; i < config_.image_views->size(); i++)
    {
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderpass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &(*config_.image_views)[i];
        framebufferInfo.width = config_.extent.width;
        framebufferInfo.height = config_.extent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &framebuffers_[i]) != VK_SUCCESS)
        {
            Logger::LogError("Failed to create framebuffer");
            return false;
        }
    }

    return true;
}