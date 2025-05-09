#pragma once

#include "pool.h"
#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <limits>
#include <functional> // Required for std::function
#include <string>     // Required for std::string
#include <map>        // Required for std::map
#include <iostream>

namespace vra
{
    // - Resource ID is used to identify the resource
    // - actual type: uint64_t
    using ResourceId = uint64_t;

    // - Batch ID is used to identify the batch
    // - actual type: char*
    using BatchId = std::string;
    class VraBuiltInBatchIds
    {
    public:
        static constexpr BatchId GPU_Only = "GPU_Only";
        static constexpr BatchId CPU_GPU = "CPU_GPU";
    };

    // --- Forward Declarations ---
    class VraDataDesc;
    struct VraRawData;

    // --- Data Memory Pattern ---
    enum class VraDataMemoryPattern
    {
        // Default
        Default,

        // - Device Local, no CPU access
        GPU_Only,

        // - CPU and GPU sequential access, e.g. UBO update
        CPU_GPU,

        // - CPU and GPU random access; 
        // - Always has Host-Cached
        GPU_CPU,

        // - CPU and GPU access with unified memory architecture
        SOC,

        // - ring buffer mode, e.g. indirect draw command; 
        // - Always has Host-Cached
        Stream_Ring,
    };

    // --- Data Update Rate ---
    enum class VraDataUpdateRate
    {
        // Default
        Default,

        // - Update frequently, e.g. UBO update;
        // - Always has Host-Coherent
        Frequent,

        // - Update rarely or never, e.g. static and uniform data;
        // - Require to flush memory before using if there is a transfer
        RarelyOrNever
    };

    struct VraRawData
    {
        const void *pData_ = nullptr;
        size_t size_ = 0;
    };

    class VraDataDesc
    {
    public:
        VraDataDesc() = default;
        VraDataDesc(VraDataMemoryPattern pattern,
                    VraDataUpdateRate update_rate,
                    VkBufferCreateInfo buffer_create_info = VkBufferCreateInfo{})
            : data_pattern_(pattern), data_update_rate_(update_rate), buffer_create_info_(buffer_create_info)
        {
        }
        ~VraDataDesc() = default;

        // --- Copy and Move ---
        VraDataDesc(const VraDataDesc &other) = default;
        VraDataDesc &operator=(const VraDataDesc &other) = default;
        VraDataDesc(VraDataDesc &&other) noexcept = default;
        VraDataDesc &operator=(VraDataDesc &&other) noexcept = default;

        // --- Getter Method ---
        VraDataMemoryPattern GetMemoryPattern() const { return data_pattern_; }
        VraDataUpdateRate GetUpdateRate() const { return data_update_rate_; }
        const VkBufferCreateInfo& GetBufferCreateInfo() const { return buffer_create_info_; }
        VkBufferCreateInfo& GetBufferCreateInfo() { return buffer_create_info_; }

    private:
        // --- main members ---
        VraDataMemoryPattern data_pattern_;
        VraDataUpdateRate data_update_rate_;
        VkBufferCreateInfo buffer_create_info_;
    };

    class VraDataBatcher
    {
        /// @brief VraBatchHandle is a struct that contains a vector of uint8_t, an unordered_map of ResourceId and size_t, and a VraDataDesc.
        /// @brief VraBatchHandle is used to store batch information like consolidated data, offsets, and data description.
        struct VraBatchHandle
        {
            bool initialized = false;
            std::vector<uint8_t> consolidated_data;
            std::unordered_map<ResourceId, size_t> offsets;
            VraDataDesc data_desc;

            void Clear()
            {
                initialized = false;
                consolidated_data.clear();
                offsets.clear();
                data_desc = VraDataDesc{};
            }
        };

        /// @brief VraBatcher is a struct that contains a batch id, a predicate, and a batch method.
        /// @brief VraBatcher is used to batch buffer data by memory pattern and update rate according to the predicate with the batch method
        struct VraBatcher
        {
            const BatchId batch_id;
            std::function<bool(const VraDataDesc &)> predicate;
            std::function<void(ResourceId id,
                               VraBatchHandle &batch,
                               const VraDataDesc &data_desc,
                               const VraRawData &data)>
                batch_method;
            VraBatchHandle batch_handle; // Each strategy instance owns its data
        };

    public:
        VraDataBatcher() = delete;
        VraDataBatcher(VkPhysicalDevice physical_device) : physical_device_handle_(physical_device) 
        {
            vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties_);
            RegisterDefaultBatcher();
        }
        ~VraDataBatcher();
        
        // --- Data Process ---

        /// @brief Collects buffer data description and raw data pointer.
        /// @param data_desc Description of the buffer data (usage, memory pattern, etc.).
        /// @param data Raw data pointer and size. The pointer pData_ must remain valid until Execute is called.
        /// @return True if collected successfully, false if ID already exists or max count reached.
        bool Collect(VraDataDesc data_desc, VraRawData data, ResourceId& id);

        /// @brief processes all collected buffer data, grouping them by memory pattern
        void Batch();

        /// @brief clear all collected data and grouped data
        void Clear();

        // --- Data Access ---

        /// @brief register buffer batch strategy
        /// @param batch_id register batch id
        /// @param predicate predicate before grouping
        /// @param batch_method action when grouping
        void RegisterBatcher(
            const BatchId batch_id,
            std::function<bool(const VraDataDesc &)> predicate,
            std::function<void(ResourceId id, VraBatchHandle &batch, const VraDataDesc &data_desc, const VraRawData &data)> batch_method);

        /// @brief get batch index
        /// @param batch_id batch id
        /// @return batch index
        size_t GetBatchIndex(const BatchId& batch_id) const
        {
            auto it = batch_id_to_index_map_.find(batch_id);
            if (it != batch_id_to_index_map_.end())
            {
                return it->second;
            }
            return std::numeric_limits<size_t>::max();
        }

        /// @brief get batch data
        /// @param batch_id batch id
        /// @return batch data
        const VraBatchHandle* GetBatch(const BatchId& batch_id) const
        {
            auto it = batch_id_to_index_map_.find(batch_id);
            if (it != batch_id_to_index_map_.end())
            {
                if (it->second < registered_batchers_.size())
                { // Boundary check
                    return &registered_batchers_[it->second].batch_handle;
                }
            }
            return nullptr;
        }
        
        /// @brief get all batch ids
        /// @return all batch ids
        std::vector<BatchId> GetAllBatchIDs() const 
        { 
            std::vector<BatchId> ids;
            for (const auto& batch : registered_batchers_) {
                ids.push_back(batch.batch_id);
            }
            return ids;
        }

        size_t GetResourceOffset(BatchId batch_id, ResourceId id) const
        {
            return GetBatch(batch_id)->offsets.at(id);
        }
        
        /// @brief get suggest memory flags for vulkan
        /// @param batch_id batch id
        /// @return suggest memory flags
        VkMemoryPropertyFlags GetSuggestMemoryFlags(BatchId batch_id);

        /// @brief get suggest vma memory flags for vulkan
        /// @param batch_id batch id
        /// @return suggest vma memory flags
        VmaAllocationCreateFlags GetSuggestVmaMemoryFlags(BatchId batch_id);

    private :
        // --- Vulkan Native Objects Cache ---
        
        VkPhysicalDevice physical_device_handle_;
        VkPhysicalDeviceProperties physical_device_properties_;

        // --- Buffer specific storage ---
        
        std::unordered_map<ResourceId, VraDataDesc> buffer_desc_map_;   // TODO: Optimize to use vector
        std::unordered_map<ResourceId, VraRawData> buffer_data_map_;    // TODO: Optimize to use vector

        // --- Limits ---
        
        static constexpr size_t MAX_BUFFER_COUNT = 4096;

        // --- Registered Group Strategies ---

        std::vector<VraBatcher> registered_batchers_;
        std::map<BatchId, size_t> batch_id_to_index_map_;

        /// @brief clear grouped buffer data
        void ClearBatch();

        /// @brief register default grouping strategies
        void RegisterDefaultBatcher();
    };

    class VraDescriptorAllocator
    {
        VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;

        // constructor
        VraDescriptorAllocator(VkDevice device)
            : device_(device)
        {
        }

        // --- Helper methods ---

        // --- Descriptor methods ---
        VkDescriptorSet AllocateDescriptorSet(VkDescriptorSetLayout layout)
        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptor_pool_;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &layout;

            VkDescriptorSet descriptor_set;
            VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set);
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate descriptor set!");
            }

            return descriptor_set;
        }
    };

    class VRA
    {
    public:
        VRA();
        ~VRA();
    };
}