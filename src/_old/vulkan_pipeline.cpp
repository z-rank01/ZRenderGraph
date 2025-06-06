#include "vulkan_pipeline.h"

VulkanPipelineHelper::~VulkanPipelineHelper()
{
    if (pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
}

bool VulkanPipelineHelper::CreatePipeline(VkDevice device)
{
    device_ = device;
    
    // input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &config_.vertex_input_binding_description;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(config_.vertex_input_attribute_descriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = config_.vertex_input_attribute_descriptions.data();

    // viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)config_.swap_chain_extent.width;
    viewport.height = (float)config_.swap_chain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = config_.swap_chain_extent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // Disable if you want to draw
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // Fill the polygon
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    // 如果模型依然颠倒，可以尝试更改这个值
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // 或者尝试 VK_FRONT_FACE_CLOCKWISE
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    // multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    // depth and stencil testing
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;           // 启用深度测试
    depthStencil.depthWriteEnable = VK_TRUE;          // 启用深度写入
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS; // 使用小于比较操作（标准深度测试）
    depthStencil.depthBoundsTestEnable = VK_FALSE;    // 禁用深度边界测试
    depthStencil.minDepthBounds = 0.0f;               // 可选
    depthStencil.maxDepthBounds = 1.0f;               // 可选
    depthStencil.stencilTestEnable = VK_FALSE;        // 禁用模板测试
    depthStencil.front = {};                          // 可选，前面的模板操作
    depthStencil.back = {};                           // 可选，背面的模板操作

    // color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    // pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(config_.descriptor_set_layouts.size());
    pipelineLayoutInfo.pSetLayouts = config_.descriptor_set_layouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (!Logger::LogWithVkResult(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipeline_layout_), 
        "Failed to create pipeline layout", 
        "Created pipeline layout successfully"))
    {
        return false;
    }

    VkPipelineShaderStageCreateInfo shader_stages[2];
    // 获取顶点着色器模块
    auto it_vert = config_.shader_module_map.find(EShaderType::kVertexShader);
    if (it_vert == config_.shader_module_map.end()) {
        Logger::LogError("Vertex shader module not found in pipeline config map.");
        return false;
    }
    shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = it_vert->second;
    shader_stages[0].pName = "main";
    shader_stages[0].pSpecializationInfo = nullptr;
    shader_stages[0].pNext = nullptr;
    shader_stages[0].flags = 0;

    // 获取片段着色器模块
    auto it_frag = config_.shader_module_map.find(EShaderType::kFragmentShader);
    if (it_frag == config_.shader_module_map.end()) {
        Logger::LogError("Fragment shader module not found in pipeline config map.");
        return false;
    }
    shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = it_frag->second;
    shader_stages[1].pName = "main";
    shader_stages[1].pSpecializationInfo = nullptr;
    shader_stages[1].pNext = nullptr;
    shader_stages[1].flags = 0;

    std::vector<VkDynamicState> dynamicStates =
    {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // pipeline create info
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(std::size(shader_stages));
    pipelineInfo.pStages = shader_stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil; // 将深度测试状态添加到管线中
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipeline_layout_;
    pipelineInfo.renderPass = config_.renderpass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional
    pipelineInfo.pNext = nullptr; // Optional
    pipelineInfo.flags = 0; // Optional


    return Logger::LogWithVkResult(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_), 
        "Failed to create graphics pipeline", 
        "Succeeded in creating graphics pipeline");
}