#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h> // Include VMA header AFTER the implementation define
#include <algorithm>          // for std::find_if, std::remove_if if needed later
#include "vra.h"

namespace vra
{
    const BatchId VraBuiltInBatchIds::GPU_Only = "GPU_Only";
    const BatchId VraBuiltInBatchIds::CPU_GPU_Rarely = "CPU_GPU_Rarely";
    const BatchId VraBuiltInBatchIds::CPU_GPU_Frequently = "CPU_GPU_Frequently";
    const BatchId VraBuiltInBatchIds::GPU_CPU_Rarely = "GPU_CPU_Rarely";
    const BatchId VraBuiltInBatchIds::GPU_CPU_Frequently = "GPU_CPU_Frequently";

    // -------------------------------------
    // --- Data Collector Implementation ---
    // -------------------------------------

    VraDataBatcher::~VraDataBatcher()
    {
        Clear();
    }

    void VraDataBatcher::RegisterBatcher(
        const BatchId batch_id,
        std::function<bool(const VraDataDesc &)> predicate,
        std::function<void(ResourceId, VraBatchHandle &, const VraDataDesc &, const VraRawData &)> batch_action)
    {
        if (batch_id_to_index_map_.count(batch_id))
        {
            // Optionally throw an error or log a warning
            // For now, let's overwrite or ignore if names clash, but unique names are better.
            // For simplicity, we'll assume names are unique for now or the first registration wins.
            // A more robust system might throw or return a bool.
            return;
        }
        registered_batchers_.push_back(VraBatcher{std::move(batch_id), std::move(predicate), std::move(batch_action)});
        batch_id_to_index_map_[registered_batchers_.back().batch_id] = registered_batchers_.size() - 1;
    }

    void VraDataBatcher::RegisterDefaultBatcher()
    {
        RegisterBatcher(
            vra::VraBuiltInBatchIds::GPU_Only,
            [](const VraDataDesc &desc)
            {
                return desc.GetMemoryPattern() == VraDataMemoryPattern::GPU_Only && desc.GetUpdateRate() == VraDataUpdateRate::RarelyOrNever;
            },
            [](ResourceId id, VraBatchHandle &batch, const VraDataDesc &desc, const VraRawData &raw_data)
            {
                const auto &current_item_buffer_create_info = desc.GetBufferCreateInfo();

                // --------------------------
                // ---- initialize batch ----
                // --------------------------

                if (!batch.initialized)
                {
                    batch.data_desc = desc; // Initialize the batch's entire VraDataDesc
                    batch.initialized = true;
                }
                else
                {
                    auto &batch_ci_ref = batch.data_desc.GetBufferCreateInfo();
                    if (current_item_buffer_create_info.usage == 0 ||
                        current_item_buffer_create_info.sharingMode != batch_ci_ref.sharingMode)
                    {
                        return; // Incompatible
                    }

                    batch_ci_ref.usage |= current_item_buffer_create_info.usage;
                    batch_ci_ref.flags |= current_item_buffer_create_info.flags;
                    if (current_item_buffer_create_info.queueFamilyIndexCount > batch_ci_ref.queueFamilyIndexCount &&
                        current_item_buffer_create_info.sharingMode == VK_SHARING_MODE_CONCURRENT)
                    {
                        batch_ci_ref.queueFamilyIndexCount = current_item_buffer_create_info.queueFamilyIndexCount;
                        batch_ci_ref.pQueueFamilyIndices = current_item_buffer_create_info.pQueueFamilyIndices;
                    }
                }

                // -------------------------
                // ---- add resource id ----
                // -------------------------

                batch.resource_ids.push_back(id);

                // ---------------------------
                // ---- add data to batch ----
                // ---------------------------

                if (raw_data.pData_ != nullptr && raw_data.size_ > 0)
                {
                    batch.offsets[id] = batch.consolidated_data.size();
                    const uint8_t *data_ptr = static_cast<const uint8_t *>(raw_data.pData_);
                    batch.consolidated_data.insert(batch.consolidated_data.end(), data_ptr, data_ptr + raw_data.size_);
                }
                else
                {
                    batch.offsets[id] = batch.consolidated_data.size();
                }

                // ----------------------------
                // ---- update buffer size ----
                // ----------------------------

                batch.data_desc.GetBufferCreateInfo().size = batch.consolidated_data.size();
            });

        // Dynamic Sequential Batch
        RegisterBatcher(
            vra::VraBuiltInBatchIds::CPU_GPU_Rarely,
            [](const VraDataDesc &desc)
            {
                return desc.GetMemoryPattern() == VraDataMemoryPattern::CPU_GPU && desc.GetUpdateRate() == VraDataUpdateRate::RarelyOrNever;
            },
            [&physical_device_properties = this->physical_device_properties_](ResourceId id, VraBatchHandle &batch, const VraDataDesc &desc, const VraRawData &raw_data)
            {
                const auto &current_item_buffer_create_info = desc.GetBufferCreateInfo();
                // Get a modifiable reference to the batch's VkBufferCreateInfo
                auto &batch_ci_ref = batch.data_desc.GetBufferCreateInfo();

                // -----------------------------
                // ---- combine buffer info ----
                // -----------------------------

                if (!batch.initialized)
                {
                    batch.data_desc = desc; // Initialize the batch's entire VraDataDesc
                    batch.initialized = true;
                    // batch_ci_ref now refers to the CI within the newly set batch.data_desc                    
                }
                else if (current_item_buffer_create_info.usage == 0 || 
                         current_item_buffer_create_info.sharingMode != batch_ci_ref.sharingMode)
                {
                    return; // Incompatible
                }
                else
                {
                    // Merge into batch_ci_ref
                    batch_ci_ref.usage |= current_item_buffer_create_info.usage;
                    batch_ci_ref.flags |= current_item_buffer_create_info.flags;
                    if (current_item_buffer_create_info.queueFamilyIndexCount > batch_ci_ref.queueFamilyIndexCount &&
                        current_item_buffer_create_info.sharingMode == VK_SHARING_MODE_CONCURRENT)
                    {
                        batch_ci_ref.queueFamilyIndexCount = current_item_buffer_create_info.queueFamilyIndexCount;
                        batch_ci_ref.pQueueFamilyIndices = current_item_buffer_create_info.pQueueFamilyIndices;
                    }
                }

                // -------------------------
                // ---- add resource id ----
                // -------------------------

                batch.resource_ids.push_back(id);

                // ---------------------------------
                // -- combine data with alignment --
                // ---------------------------------

                size_t base_offset_for_item = batch.consolidated_data.size();
                size_t aligned_offset_for_item = base_offset_for_item;

                // Apply alignment if the batch's buffer usage suggests it (e.g., UBO, SSBO)
                if ((batch_ci_ref.usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) != 0)
                {
                    size_t alignment_requirement = physical_device_properties.limits.minUniformBufferOffsetAlignment; 
                    // TODO: Consider other alignment requirements if necessary, e.g., minStorageBufferOffsetAlignment
                    if (alignment_requirement > 0)
                    {
                        aligned_offset_for_item = (base_offset_for_item + alignment_requirement - 1) & ~(alignment_requirement - 1);
                    }
                }

                size_t padding_needed = aligned_offset_for_item - base_offset_for_item;
                if (padding_needed > 0)
                {
                    batch.consolidated_data.insert(batch.consolidated_data.end(), padding_needed, (uint8_t)0); // Pad with zeros
                }

                batch.offsets[id] = aligned_offset_for_item; // Store the correctly aligned offset

                // -------------------------
                // --- add data to batch ---
                // -------------------------

                if (raw_data.pData_ != nullptr && raw_data.size_ > 0)
                {
                    const uint8_t *data_ptr = static_cast<const uint8_t *>(raw_data.pData_);
                    batch.consolidated_data.insert(batch.consolidated_data.end(), data_ptr, data_ptr + raw_data.size_);
                }

                // ----------------------------
                // ---- update buffer size ----
                // ----------------------------

                batch_ci_ref.size = batch.consolidated_data.size();
            });

        // Dynamic Sequential Batch (Frequently)
        RegisterBatcher(
            vra::VraBuiltInBatchIds::CPU_GPU_Frequently,
            [](const VraDataDesc &desc)
            {
                return desc.GetMemoryPattern() == VraDataMemoryPattern::CPU_GPU && desc.GetUpdateRate() == VraDataUpdateRate::Frequent;
            },
            [&physical_device_properties = this->physical_device_properties_](ResourceId id, VraBatchHandle &batch, const VraDataDesc &desc, const VraRawData &raw_data)
            {
                const auto &current_item_buffer_create_info = desc.GetBufferCreateInfo();
                // Get a modifiable reference to the batch's VkBufferCreateInfo
                auto &batch_ci_ref = batch.data_desc.GetBufferCreateInfo();

                // -----------------------------
                // ---- combine buffer info ----
                // -----------------------------

                if (!batch.initialized)
                {
                    batch.data_desc = desc; // Initialize the batch's entire VraDataDesc
                    batch.initialized = true;
                    // batch_ci_ref now refers to the CI within the newly set batch.data_desc
                }
                else if (current_item_buffer_create_info.usage == 0 ||
                         current_item_buffer_create_info.sharingMode != batch_ci_ref.sharingMode)
                {
                    return; // Incompatible
                }
                else
                {
                    // Merge into batch_ci_ref
                    batch_ci_ref.usage |= current_item_buffer_create_info.usage;
                    batch_ci_ref.flags |= current_item_buffer_create_info.flags;
                    if (current_item_buffer_create_info.queueFamilyIndexCount > batch_ci_ref.queueFamilyIndexCount &&
                        current_item_buffer_create_info.sharingMode == VK_SHARING_MODE_CONCURRENT)
                    {
                        batch_ci_ref.queueFamilyIndexCount = current_item_buffer_create_info.queueFamilyIndexCount;
                        batch_ci_ref.pQueueFamilyIndices = current_item_buffer_create_info.pQueueFamilyIndices;
                    }
                }

                // -------------------------
                // ---- add resource id ----
                // -------------------------

                batch.resource_ids.push_back(id);

                // ---------------------------------
                // -- combine data with alignment --
                // ---------------------------------

                size_t base_offset_for_item = batch.consolidated_data.size();
                size_t aligned_offset_for_item = base_offset_for_item;

                // Apply alignment if the batch's buffer usage suggests it (e.g., UBO, SSBO)
                if ((batch_ci_ref.usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) != 0)
                {
                    size_t alignment_requirement = physical_device_properties.limits.minUniformBufferOffsetAlignment;
                    // TODO: Consider other alignment requirements if necessary, e.g., minStorageBufferOffsetAlignment
                    if (alignment_requirement > 0)
                    {
                        aligned_offset_for_item = (base_offset_for_item + alignment_requirement - 1) & ~(alignment_requirement - 1);
                    }
                }

                size_t padding_needed = aligned_offset_for_item - base_offset_for_item;
                if (padding_needed > 0)
                {
                    batch.consolidated_data.insert(batch.consolidated_data.end(), padding_needed, (uint8_t)0); // Pad with zeros
                }

                batch.offsets[id] = aligned_offset_for_item; // Store the correctly aligned offset

                if (raw_data.pData_ != nullptr && raw_data.size_ > 0)
                {
                    const uint8_t *data_ptr = static_cast<const uint8_t *>(raw_data.pData_);
                    batch.consolidated_data.insert(batch.consolidated_data.end(), data_ptr, data_ptr + raw_data.size_);
                }

                batch_ci_ref.size = batch.consolidated_data.size();
            });
    }

    ResourceId VraDataBatcher::Collect(const std::string &type_name, VraDataDesc data_desc, VraRawData data)
    {
        // generate resource id
        ResourceId next_id = id_generator_.GenerateID(type_name);
        if (next_id == std::numeric_limits<ResourceId>::max())
        {
            std::cout << "Registering type: " << type_name << std::endl;
            id_generator_.RegisterType(type_name);
            next_id = id_generator_.GenerateID(type_name);
        }

        // check buffer conditions
        if (data_desc.GetBufferCreateInfo().usage == 0 ||
            collected_data_.size() >= MAX_BUFFER_COUNT ||
            data.pData_ == nullptr ||
            data.size_ == 0)
        {
            std::cerr << "Error: Failed to collect buffer data for type '" << type_name << "'" << std::endl;
            return std::numeric_limits<ResourceId>::max();
        }

        // store buffer data
        collected_data_.emplace_back(next_id, type_name, data_desc, data);

        return next_id;
    }

    VkMemoryPropertyFlags VraDataBatcher::GetSuggestMemoryFlags(BatchId batch_id)
    {
        if (batch_id_to_index_map_.find(batch_id) == batch_id_to_index_map_.cend())
        {
            std::cerr << "Batch id " << batch_id << " not found" << std::endl;
            return VkMemoryPropertyFlagBits();
        }
        const VraDataDesc &data_desc = registered_batchers_[batch_id_to_index_map_[batch_id]].batch_handle.data_desc;
        const VraDataMemoryPattern &data_pattern = data_desc.GetMemoryPattern();
        const VraDataUpdateRate &data_update_rate = data_desc.GetUpdateRate();
        switch (data_pattern)
        {
        case VraDataMemoryPattern::GPU_Only:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        case VraDataMemoryPattern::CPU_GPU:
            if (data_update_rate == VraDataUpdateRate::Frequent)
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        case VraDataMemoryPattern::GPU_CPU:
            if (data_update_rate == VraDataUpdateRate::Frequent)
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        case VraDataMemoryPattern::SOC:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        case VraDataMemoryPattern::Stream_Ring:
            if (data_update_rate == VraDataUpdateRate::Frequent)
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        default:
            return VkMemoryPropertyFlagBits();
        }
    }

    VmaAllocationCreateFlags VraDataBatcher::GetSuggestVmaMemoryFlags(BatchId batch_id)
    {
        if (batch_id_to_index_map_.find(batch_id) == batch_id_to_index_map_.cend())
        {
            std::cerr << "Batch id " << batch_id << " not found" << std::endl;
            return VmaAllocationCreateFlags();
        }
        const VraDataDesc &data_desc = registered_batchers_[batch_id_to_index_map_[batch_id]].batch_handle.data_desc;
        const VraDataMemoryPattern &data_pattern = data_desc.GetMemoryPattern();
        const VraDataUpdateRate &data_update_rate = data_desc.GetUpdateRate();
        switch (data_pattern)
        {
        case VraDataMemoryPattern::GPU_Only:
            return VmaAllocationCreateFlags();

        case VraDataMemoryPattern::CPU_GPU:
            if (data_update_rate == VraDataUpdateRate::Frequent)
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        case VraDataMemoryPattern::GPU_CPU:
            if (data_update_rate == VraDataUpdateRate::Frequent)
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

        case VraDataMemoryPattern::SOC:
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        case VraDataMemoryPattern::Stream_Ring:
            if (data_update_rate == VraDataUpdateRate::Frequent)
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            else if (data_update_rate == VraDataUpdateRate::RarelyOrNever)
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        default:
            return VmaAllocationCreateFlags();
        }
    }

    void VraDataBatcher::ClearBatch()
    {
        for (auto &strategy : registered_batchers_)
        {
            strategy.batch_handle.Clear();
        }
    }

    void VraDataBatcher::Clear()
    {
        collected_data_.clear();
        ClearBatch();
    }

    void VraDataBatcher::Batch()
    {
        ClearBatch();

        // Optional: Estimate sizes and reserve capacity
        std::vector<size_t> estimated_batch_sizes(registered_batchers_.size(), 0);
        for (const auto &data_handle : collected_data_)
        {
            const VraRawData &raw_data = data_handle.raw_data;
            const VraDataDesc &desc = data_handle.data_desc;

            for (size_t i = 0; i < registered_batchers_.size(); ++i)
            {
                if (registered_batchers_[i].predicate(desc))
                {
                    // Basic estimation, doesn't account for padding in dynamic batches yet.
                    // For a more accurate estimation, the predicate or a dedicated estimation function
                    // in the strategy would be needed if padding is significant.
                    estimated_batch_sizes[i] += raw_data.size_;
                    break;
                }
            }
        }
        for (size_t i = 0; i < registered_batchers_.size(); ++i)
        {
            if (estimated_batch_sizes[i] > 0)
            { // Only reserve if there's an estimate
                registered_batchers_[i].batch_handle.consolidated_data.reserve(estimated_batch_sizes[i]);
            }
        }

        // Batch data
        for (const auto &data_handle : collected_data_)
        {
            ResourceId id = data_handle.id;
            const VraRawData &raw_data = data_handle.raw_data;
            const VraDataDesc &desc = data_handle.data_desc;

            for (auto &strategy : registered_batchers_)
            {
                if (strategy.predicate(desc))
                {
                    strategy.batch_method(id, strategy.batch_handle, desc, raw_data);
                    break; // Assume buffer belongs to only one batch based on predicate
                }
            }
        }

        // Shrink to fit for non-dynamic batches (as per original logic)
        for (auto &strategy : registered_batchers_)
        {
            // The original dynamic_sequential_batch did not shrink_to_fit.
            // You might want a flag in VraBatcher to control this behavior.
            if (strategy.batch_id != "CPU_GPU")
            {
                strategy.batch_handle.consolidated_data.shrink_to_fit();
            }
        }
    }

    // -------------------------------------
    // --- VRA Implementation ---
    // -------------------------------------

    VRA::VRA()
    {
    }

    VRA::~VRA()
    {
    }
}
