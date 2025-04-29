#define GLFW_INCLUDE_VULKAN // 自动包含 vulkan.h
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <optional>
#include <stdexcept>
#include <array>
#include <chrono>
#include <cstring>

//vulkan调试工具
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
    "VK_LAYER_LUNARG_api_dump"
};

//读取文件
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file!");
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// 通用内存类型查找函数
uint32_t findMemoryType(
    vk::PhysicalDevice physicalDevice,
    uint32_t typeFilter,
    vk::MemoryPropertyFlags properties)
{
    auto memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

// 通用 Buffer 创建函数
std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>
createBuffer(
    const vk::UniqueDevice& device,  // 改用原始句柄（避免双重 Unique 所有权）
    const vk::PhysicalDevice& physicalDevice,
    size_t size,
    vk::BufferUsageFlags usage,
    const void* data = nullptr
) {
    // 1. 创建 Buffer
    vk::BufferCreateInfo bufferInfo({}, size, usage);
    auto buffer = device->createBufferUnique(bufferInfo);

    // 2. 分配内存
    auto memReq = device->getBufferMemoryRequirements(*buffer);
    vk::MemoryAllocateInfo allocInfo(
        memReq.size,
        findMemoryType(
            physicalDevice,
            memReq.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent
        )
    );
    auto memory = device->allocateMemoryUnique(allocInfo);
    device->bindBufferMemory(*buffer, *memory, 0);

    // 3. 填充数据（可选）
    if (data) {
        void* mapped = device->mapMemory(*memory, 0, size);
        memcpy(mapped, data, size);
        device->unmapMemory(*memory);
    }

    return { std::move(buffer), std::move(memory) };  // 显式转移所有权
}

//-----------------About Vulkan----------------------
//创建窗口(初始化 GLFW，告诉它“我不打算用 OpenGL”)
GLFWwindow* InitWindow(int w, int h, std::string winName) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    return glfwCreateWindow(w, h, winName.c_str(), nullptr, nullptr);
}
//创建 Vulkan 实例（等于：跟系统打招呼“我要用 Vulkan”）
vk::UniqueInstance InitInstance() {
    vk::ApplicationInfo appInfo("Lesson 3", 1, "NoEngine", 1, VK_API_VERSION_1_0);
    vk::InstanceCreateInfo createInfo({}, &appInfo);

    //vk::InstanceCreateInfo createInfo(
    //    vk::InstanceCreateFlags(),
    //    &appInfo,
    //    validationLayers.size(), validationLayers.data(),  // 启用验证层
    //    0, nullptr  // 扩展
    //);

    auto extensions = glfwGetRequiredInstanceExtensions(&createInfo.enabledExtensionCount);//remember to add extentions
    createInfo.ppEnabledExtensionNames = extensions;
    return vk::createInstanceUnique(createInfo);
}

//创建 Surface（vulkan与操作系统窗口(glfw)之间的跑腿小哥，对接作用）
vk::SurfaceKHR InitSurface(
    const vk::UniqueInstance& instance,
    GLFWwindow* window)
{
    VkSurfaceKHR c_surface;
    if (glfwCreateWindowSurface(static_cast<VkInstance>(*instance), window, nullptr, &c_surface) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface!");
    return c_surface;
}

//创建硬件队列家族（画师家族）
std::optional<uint32_t> InitGraphicFamily(
    const vk::PhysicalDevice& physicalDevice,
    const vk::SurfaceKHR& surface)
{
    auto families = physicalDevice.getQueueFamilyProperties();
    std::optional<uint32_t> graphicsFamily;
    for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics && physicalDevice.getSurfaceSupportKHR(i, surface)) {
            graphicsFamily = i;
            break;
        }
    }
    if (!graphicsFamily.has_value()) throw std::runtime_error("no suitable queue family found");
    return graphicsFamily;
}

//逻辑设备：软件层来控制物理设备的中间层，就像个仪表盘一样——仪表盘接受人类的命令，然后控制硬件工作
vk::UniqueDevice InitDevice(
    const std::optional<uint32_t>& graphicsFamily,
    const vk::PhysicalDevice& physicalDevice,
    bool enableComputeQueue = false,  // 【新增】可选参数，默认false
    const std::optional<uint32_t>& computeFamily = std::nullopt) // 【新增】计算队列家族
{
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    float priority = 1.0f;

    // 始终添加图形队列
    queueInfos.emplace_back(vk::DeviceQueueCreateFlags{},
        graphicsFamily.value(),
        1, &priority);

    // 【新增】如果需要计算队列且与图形队列不同
    if (enableComputeQueue && computeFamily &&
        computeFamily.value() != graphicsFamily.value()) {
        queueInfos.emplace_back(vk::DeviceQueueCreateFlags{},
            computeFamily.value(),
            1, &priority);
    }

    // 重要！注意设备也需要扩展
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // 【新增】如果需要计算着色器，启用相关特性
    vk::PhysicalDeviceFeatures deviceFeatures;

    vk::DeviceCreateInfo deviceCreateInfo(
        {},
        queueInfos,
        {},
        deviceExtensions,
        &deviceFeatures);

    return physicalDevice.createDeviceUnique(deviceCreateInfo);
}

//传送带：运送framebuffer的工具，用来做多帧缓冲
vk::UniqueSwapchainKHR InitSwapChain(
    const vk::PhysicalDevice& physicalDevice,
    const vk::UniqueDevice& device,
    const vk::SurfaceKHR& surface,
    const vk::SurfaceCapabilitiesKHR& capabilities,
    const vk::SurfaceFormatKHR& surfaceFormat,
    const vk::Extent2D& extent)
{
    auto presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
    vk::SwapchainCreateInfoKHR swapchainInfo({}, surface,
        capabilities.minImageCount, surfaceFormat.format, surfaceFormat.colorSpace,
        extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive, {}, vk::SurfaceTransformFlagBitsKHR::eIdentity,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, presentModes[0], VK_TRUE);

    return device->createSwapchainKHRUnique(swapchainInfo);
}

//ImageView：传送带上各个画板（framebuffer）的画纸说明书，因为它决定了每张图片的格式、底色、用途（颜色、深度）等基本信息
std::vector<vk::UniqueImageView> InitImageViews(
    const vk::UniqueDevice& device,
    const vk::UniqueSwapchainKHR& swapchain,
    const vk::SurfaceFormatKHR& surfaceFormat)
{
    auto images = device->getSwapchainImagesKHR(*swapchain);
    std::vector<vk::UniqueImageView> imageViews;
    for (auto image : images) {
        vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, surfaceFormat.format,
            {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        imageViews.push_back(device->createImageViewUnique(viewInfo));
    }
    return imageViews;
}

//RenderPass：提前规划好的绘画步骤说明书，subpass则是每一步骤。默认带一个subpass.
vk::UniqueRenderPass InitRenderPass(
    const vk::SurfaceFormatKHR& surfaceFormat,
    const vk::UniqueDevice& device)
{
    vk::AttachmentDescription colorAttachment({}, surfaceFormat.format, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

    vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics);
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    vk::RenderPassCreateInfo renderPassInfo({}, 1, &colorAttachment, 1, &subpass);
    return device->createRenderPassUnique(renderPassInfo);
}

//Framebuffer
std::vector<vk::UniqueFramebuffer> InitFrameBuffer(
    const std::vector<vk::UniqueImageView>& imageViews,
    const vk::UniqueRenderPass& renderPass,
    const vk::Extent2D& extent,
    const vk::UniqueDevice& device)
{
    std::vector<vk::UniqueFramebuffer> framebuffers;
    for (auto& view : imageViews) {
        vk::FramebufferCreateInfo fbInfo({}, *renderPass, 1, &*view, extent.width, extent.height, 1);
        framebuffers.push_back(device->createFramebufferUnique(fbInfo));
    }
    return framebuffers;
}

//Pipeline
vk::UniquePipeline InitPipeline(
    const vk::Extent2D& extent,
    const vk::UniqueDevice& device,
    const vk::UniqueRenderPass& renderPass,
    const vk::PipelineVertexInputStateCreateInfo& vertexInput,
    const vk::UniqueDescriptorSetLayout& descriptorSetLayout,
    std::array<vk::PipelineShaderStageCreateInfo, 2> stages,
    vk::UniquePipelineLayout& pipelineLayout)
{
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList);
    vk::Viewport viewport(0.0f, (float)extent.height, (float)extent.width, -(float)extent.height, 0.0f, 1.0f);
    vk::Rect2D scissor({ 0, 0 }, {extent.width, extent.height});
    vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1, &scissor);
    vk::PipelineRasterizationStateCreateInfo rasterizer({}, false, false, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);
    rasterizer.lineWidth = 1.0;//修改vulkan验证层错误
    vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1);
    vk::PipelineColorBlendAttachmentState blendState(false);
    blendState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendStateCreateInfo colorBlending({}, false, vk::LogicOp::eCopy, 1, &blendState);

    // 【修改】Pipeline Layout 现在包含 Descriptor Set
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &*descriptorSetLayout);
    pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutInfo);

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = *pipelineLayout;
    pipelineInfo.renderPass = *renderPass;
    pipelineInfo.subpass = 0;

    auto pipelineResult = device->createGraphicsPipelineUnique({}, pipelineInfo);
    return std::move(pipelineResult.value);
}

//DescriptorSet
std::vector<vk::UniqueDescriptorSet, std::allocator<vk::UniqueDescriptorSet>> InitDescriptorSets(
    const vk::UniqueDevice& device,
    vk::UniqueDescriptorSetLayout& descriptorSetLayout,
    vk::UniqueDescriptorPool& descriptorPool
)
{
    // 2. 创建 Descriptor Set Layout
    vk::DescriptorSetLayoutBinding binding(
        0, // binding
        vk::DescriptorType::eUniformBuffer,
        1,
        vk::ShaderStageFlagBits::eFragment
    );
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, 1, &binding);
    descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutInfo);

    // 3. 创建 Descriptor Pool & Set
    vk::DescriptorPoolSize DpoolSize(vk::DescriptorType::eUniformBuffer, 1);
    vk::DescriptorPoolCreateInfo DpoolInfo({}, 1, 1, &DpoolSize);
    descriptorPool = device->createDescriptorPoolUnique(DpoolInfo);

    vk::DescriptorSetAllocateInfo allocSetInfo(*descriptorPool, 1, &*descriptorSetLayout);
    return device->allocateDescriptorSetsUnique(allocSetInfo);
}

//加载shader
std::array<vk::PipelineShaderStageCreateInfo, 2> InitShader(
    const std::string& vertPath,
    const std::string& fragPath,
    vk::UniqueShaderModule& vertShader,
    vk::UniqueShaderModule& fragShader,
    const vk::UniqueDevice& device)
{
    auto vertCode = readFile(vertPath);
    auto fragCode = readFile(fragPath);
    vk::ShaderModuleCreateInfo vertInfo({}, vertCode.size(), reinterpret_cast<const uint32_t*>(vertCode.data()));
    vk::ShaderModuleCreateInfo fragInfo({}, fragCode.size(), reinterpret_cast<const uint32_t*>(fragCode.data()));
    vertShader = device->createShaderModuleUnique(vertInfo);
    fragShader = device->createShaderModuleUnique(fragInfo);

    vk::PipelineShaderStageCreateInfo vertStage({}, vk::ShaderStageFlagBits::eVertex, *vertShader, "main");
    vk::PipelineShaderStageCreateInfo fragStage({}, vk::ShaderStageFlagBits::eFragment, *fragShader, "main");
    return { vertStage, fragStage };//使用array而非传统数组，不然无法作形参传递
}