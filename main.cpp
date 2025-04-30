#include "Tools.h" // memcpy

const int WIDTH = 800;
const int HEIGHT = 800;

const int Grid_Dimen = 4;//方阵维度
const int VERTEX_COUNT = Grid_Dimen * Grid_Dimen * 4;
const int INDEX_COUNT = Grid_Dimen * Grid_Dimen * 6;

int main() {
    //==============Init Vulkan===================
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

    // Uniform缓冲改为ComputeUBO
    auto [uniformBuffer, uniformBufferMemory] = createBuffer(device, physicalDevice,
        sizeof(ComputeUBO), vk::BufferUsageFlagBits::eUniformBuffer);

    // 顶点缓冲
    auto [vertexBuffer, vertexBufferMemory] = createBuffer(device, physicalDevice,
        sizeof(Vertex) * VERTEX_COUNT,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);//注意：改为Storage Buffer，用来接收ComputeShader计算出的顶点
    std::cout << "fuck";
    // 【改】索引缓冲
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
    std::cout << "fuck1";
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

    // 创建计算描述符集布局
    vk::DescriptorSetLayoutBinding computeBindings[2] = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };
    vk::DescriptorSetLayoutCreateInfo computeLayoutInfo({}, 2, computeBindings);
    auto computeDescriptorSetLayout = device->createDescriptorSetLayoutUnique(computeLayoutInfo);

    // 创建计算管线布局
    vk::PipelineLayoutCreateInfo computePipelineLayoutInfo({}, 1, &*computeDescriptorSetLayout);
    auto computePipelineLayout = device->createPipelineLayoutUnique(computePipelineLayoutInfo);

    // 创建计算描述符池和集合
    vk::DescriptorPoolSize poolSizes[2] = {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eUniformBuffer, 1}
    };
    vk::DescriptorPoolCreateInfo descriptorPoolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 2, poolSizes);
    auto computeDescriptorPool = device->createDescriptorPoolUnique(descriptorPoolInfo);

    auto computeDescriptorSets = device->allocateDescriptorSetsUnique(
        { *computeDescriptorPool, 1, &*computeDescriptorSetLayout });

    // 更新计算描述符集
    vk::DescriptorBufferInfo vertexBufferInfo(*vertexBuffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo uniformBufferInfo(*uniformBuffer, 0, sizeof(ComputeUBO));

    vk::WriteDescriptorSet descriptorWrites[2] = {
        {*computeDescriptorSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &vertexBufferInfo},
        {*computeDescriptorSets[0], 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferInfo}
    };
    device->updateDescriptorSets(2, descriptorWrites, 0, nullptr);

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
    computeCommandBuffer->dispatch(2, 2, 1);//【改】这里决定了gl_GlobalInvocationID
    computeCommandBuffer->end();

    // 帧循环
    auto imageAvailable = device->createSemaphoreUnique({});
    auto renderFinished = device->createSemaphoreUnique({});
    auto startTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 更新Uniform数据
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(currentTime - startTime).count();

        ComputeUBO ubo{ time };
        void* data = device->mapMemory(*uniformBufferMemory, 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        device->unmapMemory(*uniformBufferMemory);

        // 提交计算工作
        vk::SubmitInfo computeSubmitInfo({}, {}, *computeCommandBuffer, *computeFinishedSemaphore);
        computeQueue.submit(computeSubmitInfo, nullptr);
        //--------------打印shader参数----------------
        
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