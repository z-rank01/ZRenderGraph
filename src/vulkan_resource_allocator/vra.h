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
#include <atomic>
#include <memory>

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
        static const BatchId GPU_Only;
        static const BatchId CPU_GPU_Rarely;
        static const BatchId CPU_GPU_Frequently;
        static const BatchId GPU_CPU_Rarely;
        static const BatchId GPU_CPU_Frequently;
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

    class ResourceIDGenerator
    {
        using AtomicResourceId = std::atomic<ResourceId>;
    public:
        ResourceIDGenerator() = default;
        ~ResourceIDGenerator() = default;

        /// @brief generate resource id
        /// @param type_name
        /// @return resource id
        ResourceId GenerateID(const std::string& type_name)
        {
            auto it = type_id_table_.find(type_name);
            if (it != type_id_table_.end())
            {
                // Atomically get the current value and then increment through the unique_ptr
                return it->second->fetch_add(1);
            }
            else
            {
                // Type not registered, handle error (e.g., return invalid ID or assert)
                std::cerr << "Error: Resource type '" << type_name << "' not registered in ResourceIDGenerator." << std::endl;
                return std::numeric_limits<ResourceId>::max(); // Indicate invalid ID
            }
        }

        /// @brief register resource type
        /// @param type_name
        void RegisterType(const std::string& type_name)
        {
            // create new atomic resource id using unique_ptr
            if (type_id_table_.find(type_name) == type_id_table_.end())
            {
                type_id_table_.emplace(type_name, std::make_unique<AtomicResourceId>(0));
            }
            else
            {
                // Type already registered, do nothing or log a warning
                std::cerr << "Warning: Resource type '" << type_name << "' already registered in ResourceIDGenerator." << std::endl;
            }
        }

    private:
        // Use unique_ptr to store atomic counters to avoid non-copyable/non-movable issues
        std::unordered_map<std::string, std::unique_ptr<AtomicResourceId>> type_id_table_;
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
    public:
        VraDataBatcher() = delete;
        VraDataBatcher(VkPhysicalDevice physical_device) : physical_device_handle_(physical_device)
        {
            vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties_);
            RegisterDefaultBatcher();
        }
        ~VraDataBatcher();

        // --- Batch Handle ---

        /// @brief VraBatchHandle is a struct that contains a vector of uint8_t, an unordered_map of ResourceId and size_t, and a VraDataDesc.
        /// @brief VraBatchHandle is used to store batch information like consolidated data, offsets, and data description.
        struct VraBatchHandle
        {
            bool initialized = false;
            std::vector<uint8_t> consolidated_data;
            std::vector<ResourceId> resource_ids;
            std::unordered_map<ResourceId, size_t> offsets;
            VraDataDesc data_desc;

            void Clear()
            {
                initialized = false;
                consolidated_data.clear();
                resource_ids.clear();
                offsets.clear();
                data_desc = VraDataDesc{};
            }
        };

        // --- Data Process ---

        /// @brief Collects buffer data description and raw data pointer.
        /// @param type_name The type name of the resource, used for ID generation.
        /// @param data_desc Description of the buffer data (usage, memory pattern, etc.).
        /// @param data Raw data pointer and size. The pointer pData_ must remain valid until Batch is called.
        /// @return The generated ResourceId if collected successfully, or std::numeric_limits<ResourceId>::max() if failed.
        /// @note The **order** in which Collect is called for different resources will **determine** their order in the batched buffer!
        ResourceId Collect(const std::string& type_name, VraDataDesc data_desc, VraRawData data);

        /// @brief processes all collected buffer data, grouping them into batches
        void Batch();

        /// @brief clear all collected data and grouped data
        void Clear();

        // --- Batcher Registration and Access ---

        /// @brief register a custom batching strategy
        /// @param batch_id register batch id
        /// @param predicate predicate before grouping
        /// @param batch_method action when grouping
        void RegisterBatcher(
            const BatchId batch_id,
            std::function<bool(const VraDataDesc &)> predicate,
            std::function<void(ResourceId id, VraBatchHandle &batch, const VraDataDesc &data_desc, const VraRawData &data)> batch_method);

        /// @brief get batch data
        /// @param batch_id batch id
        /// @return batch data
        const VraBatchHandle* GetBatch(const BatchId& batch_id) const
        {
            auto it = batch_id_to_index_map_.find(batch_id);
            if (it != batch_id_to_index_map_.end())
            {
                if (it->second < registered_batchers_.size())
                {
                    return &registered_batchers_[it->second].batch_handle;
                }
            }
            return nullptr;
        }

        /// @brief get resource offset
        /// @param batch_id batch id
        /// @param id resource id
        /// @return resource offset
        size_t GetResourceOffset(BatchId batch_id, ResourceId id) const
        {
            const auto* batch = GetBatch(batch_id);
            if (batch)
            {
                 auto it = batch->offsets.find(id);
                 if (it != batch->offsets.end())
                 {
                     return it->second;
                 }
            }
            // Handle error: batch_id or resource_id not found
            std::cerr << "Error: Resource offset not found for Batch ID '" << batch_id << "' and Resource ID '" << id << "'" << std::endl;
            return std::numeric_limits<size_t>::max(); // Indicate invalid offset
        }

        // --- Memory Allocation Flags Suggestion ---

        /// @brief get suggest memory flags for vulkan
        /// @param batch_id batch id
        /// @return suggest memory flags
        VkMemoryPropertyFlags GetSuggestMemoryFlags(BatchId batch_id);

        /// @brief get suggest vma memory flags for vulkan
        /// @param batch_id batch id
        /// @return suggest vma memory flags
        VmaAllocationCreateFlags GetSuggestVmaMemoryFlags(BatchId batch_id);

    private :

        // --- Batcher and Data Handle ---

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

        /// @brief VraDataHandle is a struct that contains a resource id, a type name, a data description, and a raw data.
        /// @brief VraDataHandle is used to store collected data
        struct VraDataHandle
        {
            ResourceId id;
            std::string type_name;
            VraDataDesc data_desc;
            VraRawData raw_data;
        };

        // --- Vulkan Native Objects Cache ---

        VkPhysicalDevice physical_device_handle_;
        VkPhysicalDeviceProperties physical_device_properties_;

        // --- Collected Data Storage ---

        std::vector<VraDataHandle> collected_data_;

        // --- Limits ---

        ResourceIDGenerator id_generator_;
        static constexpr size_t MAX_BUFFER_COUNT = 4096; // Consider if this limit is still necessary/appropriate with the new structure

        // --- Registered Batchers ---

        std::vector<VraBatcher> registered_batchers_;
        std::map<BatchId, size_t> batch_id_to_index_map_; // Use map for sorted iteration if needed, or keep unordered_map for speed


        /// @brief clear grouped buffer data
        void ClearBatch();

        /// @brief register default batchers
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