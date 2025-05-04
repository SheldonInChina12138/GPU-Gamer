#include "Tools.h" // memcpy

int main() {
    //==============Init Vulkan===================
#pragma region Init Vulkan
    GLFWwindow* window = InitWindow(WIDTH, HEIGHT, "Gamer");
    vk::UniqueInstance instance = InitInstance();
    vk::SurfaceKHR surface = InitSurface(instance, window);
    vk::PhysicalDevice physicalDevice = instance->enumeratePhysicalDevices()[0];
    std::optional<uint32_t> graphicsFamily = InitGraphicFamily(physicalDevice, surface);

    // 检查计算队列支持
    std::optional<uint32_t> computeFamily;
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
            computeFamily = i;
            break;
        }
    }
    if (!computeFamily) computeFamily = graphicsFamily; // 回退到图形队列

    // 创建逻辑设备时包含计算队列
    vk::UniqueDevice device = InitDevice(graphicsFamily, physicalDevice, true, computeFamily);

    vk::Queue graphicsQueue = device->getQueue(graphicsFamily.value(), 0);
    vk::Queue computeQueue = device->getQueue(computeFamily.value(), 0); // 计算队列

    // 获取 Surface 支持信息
    vk::SurfaceCapabilitiesKHR capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    vk::SurfaceFormatKHR surfaceFormat = physicalDevice.getSurfaceFormatsKHR(surface)[0];
    vk::Extent2D extent = capabilities.currentExtent;

    // 创建 Swapchain
    vk::UniqueSwapchainKHR swapchain = InitSwapChain(physicalDevice, device, surface, capabilities, surfaceFormat, extent);

    // 获取 Swapchain 图片 & ImageViews
    std::vector<vk::UniqueImageView> imageViews = InitImageViews(device, swapchain, surfaceFormat);

    // 创建 RenderPass
    vk::UniqueRenderPass renderPass = InitRenderPass(surfaceFormat, device);

    // 创建 Framebuffers
    std::vector<vk::UniqueFramebuffer> framebuffers = InitFrameBuffer(imageViews, renderPass, extent, device);
#pragma endregion
    
#pragma region vertex & index
    // Uniform缓冲
    auto [uniformBuffer, uniformBufferMemory] = createBuffer(device, physicalDevice, sizeof(ComputeUBO), vk::BufferUsageFlagBits::eUniformBuffer);//改为ComputeUBO
    // StorageBuffer
    auto [snakeBuffer, snakeBufferMemory] = createBuffer(
        device,
        physicalDevice,
        sizeof(glm::ivec2) * (Grid_Dimen * Grid_Dimen + 3) + sizeof(int) + sizeof(float), // 总大小
        vk::BufferUsageFlagBits::eStorageBuffer      // 改为Storage Buffer
    );

    // 顶点缓冲
    auto [vertexBuffer, vertexBufferMemory] = createBuffer(device, physicalDevice,
        sizeof(Vertex) * VERTEX_COUNT,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);//注意：改为Storage Buffer，用来接收ComputeShader计算出的顶点

    // 索引缓冲
    std::vector<uint16_t> indices(INDEX_COUNT);
    // 为每个正方形填充6个索引
    for (int i = 0; i < Grid_Dimen * Grid_Dimen; i++) {
        const int base = i * 4;
        const int offset = i * 6;

        // 第一个三角形
        indices[offset] = base;
        indices[offset + 1] = base + 1;
        indices[offset + 2] = base + 2;

        // 第二个三角形
        indices[offset + 3] = base;
        indices[offset + 4] = base + 2;
        indices[offset + 5] = base + 3;
    }
    
    auto [indexBuffer, indexBufferMemory] = createBuffer(device, physicalDevice,
        sizeof(uint16_t) * INDEX_COUNT,
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    
    void* indexData = device->mapMemory(*indexBufferMemory, 0, sizeof(uint16_t) * INDEX_COUNT);
    memcpy(indexData, indices.data(), sizeof(uint16_t) * INDEX_COUNT);
    device->unmapMemory(*indexBufferMemory);

    // 简化顶点绑定描述
    vk::VertexInputBindingDescription bindingDesc(0, sizeof(Vertex), vk::VertexInputRate::eVertex);

    // 简化顶点属性描述
    std::array<vk::VertexInputAttributeDescription, 2> attrDesc = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
    };

    vk::PipelineVertexInputStateCreateInfo vertexInput({}, bindingDesc, attrDesc);
#pragma endregion
    
    // 创建2张纹理 ==============
    auto [texWorld0, texWorld0Memory, texWorld0View] =
        createTextureImage(device, physicalDevice, Grid_Dimen, "Tex_world");
    auto [texGameOver, texGameOverMemory, texGameOverView] =
        createTextureImage(device, physicalDevice, Grid_Dimen, "Tex_GameOver");

    // 1. 创建临时命令池用于布局转换
    vk::CommandPoolCreateInfo cmdPoolInfo(
        vk::CommandPoolCreateFlagBits::eTransient,
        computeFamily.value());
    auto tempCmdPool = device->createCommandPoolUnique(cmdPoolInfo);

    // 2. 分配命令缓冲区
    auto cmdBuffers = device->allocateCommandBuffersUnique(
        { *tempCmdPool, vk::CommandBufferLevel::ePrimary, 2 });

    // 3. 为texWorld0设置布局转换
    cmdBuffers[0]->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    vk::ImageMemoryBarrier worldBarrier(
        {}, {},
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *texWorld0,
        { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmdBuffers[0]->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, 0, nullptr, 0, nullptr, 1, &worldBarrier);
    cmdBuffers[0]->end();

    // 4. 为texGameOver设置布局转换
    cmdBuffers[1]->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    vk::ImageMemoryBarrier gameOverBarrier(
        {}, {},
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *texGameOver,
        { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    cmdBuffers[1]->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, 0, nullptr, 0, nullptr, 1, &gameOverBarrier);
    cmdBuffers[1]->end();

    // 5. 提交并等待完成
    vk::SubmitInfo submitInfos[2] = {
        { {}, {}, *cmdBuffers[0], {} },
        { {}, {}, *cmdBuffers[1], {} }
    };
    computeQueue.submit(2, submitInfos, nullptr);
    computeQueue.waitIdle();

    // 创建计算描述符集布局
    vk::DescriptorSetLayoutBinding computeBindings[5] = {
    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // 顶点缓冲
    {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // 控制输入
    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, //蛇身输入
    {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},  // Tex_world
    {4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}   // Tex_GameOver
    };

    vk::DescriptorSetLayoutCreateInfo computeLayoutInfo({}, 5, computeBindings);
    auto computeDescriptorSetLayout = device->createDescriptorSetLayoutUnique(computeLayoutInfo);

    // 创建计算管线布局
    vk::PipelineLayoutCreateInfo computePipelineLayoutInfo({}, 1, &*computeDescriptorSetLayout);
    auto computePipelineLayout = device->createPipelineLayoutUnique(computePipelineLayoutInfo);

    // 创建计算描述符池和集合
    vk::DescriptorPoolSize poolSizes[3] = {
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eUniformBuffer, 1},
        {vk::DescriptorType::eStorageImage, 2} //纹理
    };
    vk::DescriptorPoolCreateInfo descriptorPoolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 3, poolSizes);
    
    auto computeDescriptorPool = device->createDescriptorPoolUnique(descriptorPoolInfo);

    auto computeDescriptorSets = device->allocateDescriptorSetsUnique(
        { *computeDescriptorPool, 1, &*computeDescriptorSetLayout });

    // 更新计算描述符集
    vk::DescriptorBufferInfo vertexBufferInfo(*vertexBuffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo uniformBufferInfo(*uniformBuffer, 0, sizeof(ComputeUBO));
    vk::DescriptorBufferInfo snakeBufferInfo(*snakeBuffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorImageInfo worldImageInfo(
        nullptr, *texWorld0View, vk::ImageLayout::eGeneral);  // 注意布局改为eGeneral
    vk::DescriptorImageInfo gameOverImageInfo(
        nullptr, *texGameOverView, vk::ImageLayout::eGeneral);
    
    vk::WriteDescriptorSet descriptorWrites[5] = {
        {*computeDescriptorSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &vertexBufferInfo},
        {*computeDescriptorSets[0], 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferInfo},
        {*computeDescriptorSets[0], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &snakeBufferInfo},
        {*computeDescriptorSets[0], 3, 0, 1,
         vk::DescriptorType::eStorageImage, &worldImageInfo},
        {*computeDescriptorSets[0], 4, 0, 1,
        vk::DescriptorType::eStorageImage, &gameOverImageInfo}
    };
    device->updateDescriptorSets(5, descriptorWrites, 0, nullptr);

    // 创建计算着色器模块
    auto computeShaderCode = readFile("Shaders/test.comp.spv");
    vk::ShaderModuleCreateInfo computeShaderInfo({}, computeShaderCode.size(),
        reinterpret_cast<const uint32_t*>(computeShaderCode.data()));
    auto computeShaderModule = device->createShaderModuleUnique(computeShaderInfo);

    // 创建计算管线
    vk::PipelineShaderStageCreateInfo computeStageInfo({},
        vk::ShaderStageFlagBits::eCompute, *computeShaderModule, "main");

    vk::ComputePipelineCreateInfo computePipelineInfo({}, computeStageInfo, *computePipelineLayout);
    auto computePipeline = device->createComputePipelineUnique(nullptr, computePipelineInfo).value;

    // 创建计算命令池和缓冲区
    auto computeCommandPool = device->createCommandPoolUnique(
        { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, computeFamily.value() });
    auto computeCommandBuffers = device->allocateCommandBuffersUnique(
        { *computeCommandPool, vk::CommandBufferLevel::ePrimary, 1 });
    auto& computeCommandBuffer = computeCommandBuffers[0];

    // 图形管线部分保持不变
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueDescriptorSet> descriptorSets = InitDescriptorSets(device, descriptorSetLayout, descriptorPool);

    vk::UniqueShaderModule vertShader;
    vk::UniqueShaderModule fragShader;
    std::array<vk::PipelineShaderStageCreateInfo, 2> stages = InitShader("Shaders/test.vert.spv", "Shaders/test.frag.spv", vertShader, fragShader, device);

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline = InitPipeline(extent, device, renderPass, vertexInput, descriptorSetLayout, stages, pipelineLayout);

    // 命令缓冲区录制
    vk::CommandPoolCreateInfo poolInfo({}, graphicsFamily.value());
    auto commandPool = device->createCommandPoolUnique(poolInfo);
    std::vector<vk::UniqueCommandBuffer> commandBuffers =
        device->allocateCommandBuffersUnique({ *commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)framebuffers.size() });

    // 计算同步原语
    auto computeFinishedSemaphore = device->createSemaphoreUnique({});

    for (size_t i = 0; i < commandBuffers.size(); ++i) {
        auto& cmd = commandBuffers[i];
        cmd->begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        // 等待计算完成
        vk::MemoryBarrier memoryBarrier(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eVertexAttributeRead);
        cmd->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexInput,
            {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

        vk::ClearValue clearColor(std::array<float, 4>{0.1f, 0.1f, 0.1f, 1.0f});
        vk::RenderPassBeginInfo rpBegin(*renderPass, *framebuffers[i], { {0,0}, extent }, clearColor);
        cmd->beginRenderPass(rpBegin, vk::SubpassContents::eInline);
        cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        cmd->bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *pipelineLayout,
            0, 1, &*descriptorSets[0],
            0, nullptr
        );

        // 绘制调用 - 现在直接绘制顶点
        cmd->bindVertexBuffers(0, *vertexBuffer, { 0 });
        cmd->bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);
        cmd->drawIndexed(INDEX_COUNT, 1, 0, 0, 0); // 使用索引绘制

        cmd->endRenderPass();
        cmd->end();
    }

    // 录制计算命令缓冲区
    computeCommandBuffer->begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });// 适用于计算命令缓冲区
    computeCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    computeCommandBuffer->bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        *computePipelineLayout,
        0, 1, &*computeDescriptorSets[0],
        0, nullptr);
    computeCommandBuffer->dispatch(3, 3, 1);//【改】这里决定了gl_GlobalInvocationID
    computeCommandBuffer->end();

    // 帧循环
    ComputeUBO ubo{};
    auto imageAvailable = device->createSemaphoreUnique({});
    auto renderFinished = device->createSemaphoreUnique({});
    auto startTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 更新Uniform数据
        auto currentTime = std::chrono::high_resolution_clock::now();
        static auto lastTime = currentTime; // 静态变量保持状态
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        // 更新到 UBO
        ubo.time += deltaTime; // 累计时间
        ubo.control = getControl(window);
        void* data = device->mapMemory(*uniformBufferMemory, 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        device->unmapMemory(*uniformBufferMemory);

        // 提交计算工作
        vk::SubmitInfo computeSubmitInfo({}, {}, *computeCommandBuffer, *computeFinishedSemaphore);
        computeQueue.submit(computeSubmitInfo, nullptr);
        //--------------打印shader参数----------------
        /*void* mappedData;
        vkMapMemory(*device, *snakeBufferMemory, 0, VK_WHOLE_SIZE, 0, &mappedData);
        SnakeSegment* fuckdata = reinterpret_cast<SnakeSegment*>(mappedData);

        printf("时间：%d，长度：%d，方向：(%d, %d)，食物：(%d, %d)，测试：(%d, %d)，蛇头位置：(%d, %d)\n",
            fuckdata->recordTime, fuckdata->snakeLen,
            fuckdata->snakeDir.x, fuckdata->snakeDir.y,
            fuckdata->foodPos.x, fuckdata->foodPos.y,
            fuckdata->tempTest.x, fuckdata->tempTest.y,
            fuckdata->pos.x, fuckdata->pos.y);
        
        vkUnmapMemory(*device, *vertexBufferMemory);*/
        // 图形部分
        uint32_t imageIndex = device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAvailable).value;

        std::array<vk::Semaphore, 2> waitSemaphores = { *computeFinishedSemaphore,
            * imageAvailable
        }; // 等待计算完成
        std::array<vk::PipelineStageFlags, 2> waitStages = {
            vk::PipelineStageFlagBits::eVertexInput,       // 计算完成后顶点数据就绪
            vk::PipelineStageFlagBits::eColorAttachmentOutput // 图像可用后渲染
        };

        vk::SubmitInfo submitInfo = {};
        submitInfo.setWaitSemaphores(waitSemaphores)
            .setWaitDstStageMask(waitStages)
            .setCommandBuffers(*commandBuffers[imageIndex])//提交渲染cmd
            .setSignalSemaphores(*renderFinished);
        graphicsQueue.submit(submitInfo, nullptr);

        vk::PresentInfoKHR presentInfo(1, &*renderFinished, 1, &*swapchain, &imageIndex);
        graphicsQueue.presentKHR(presentInfo);

        device->waitIdle();
    }

    device->waitIdle();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}