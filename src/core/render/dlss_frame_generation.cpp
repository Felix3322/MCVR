#include "core/render/dlss_frame_generation.hpp"

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/vulkan/buffer.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/descriptor.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/pipeline.hpp"
#include "core/vulkan/swapchain.hpp"
#include "core/vulkan/sync.hpp"

#include <glm/ext.hpp>

#include <nvsdk_ngx_defs_dlssg.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_params.h>

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

glm::vec3 normalizeOrFallback(const glm::vec3 &value, const glm::vec3 &fallback) {
    float len = glm::length(value);
    if (len < 1e-6f) { return fallback; }
    return value / len;
}

VkImageLayout targetPresentLayout() {
#ifdef USE_AMD
    return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}

VkFilter chooseCopyFilter(uint32_t srcWidth, uint32_t srcHeight, uint32_t dstWidth, uint32_t dstHeight) {
    return (srcWidth == dstWidth && srcHeight == dstHeight) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
}

} // namespace

DlssFrameGenerationController::DlssFrameGenerationController() = default;

DlssFrameGenerationController::~DlssFrameGenerationController() {
    destroy();
}

void DlssFrameGenerationController::init(std::shared_ptr<Framework> framework, std::shared_ptr<NgxContext> ngxContext) {
    framework_ = framework;
    ngxContext_ = ngxContext;

    available_ = false;
    if (!framework || !ngxContext || !ngxContext->isInitialized()) { return; }

    if (NVSDK_NGX_FAILED(ngxContext->queryDlssFGAvailable())) {
        std::cerr << "[DLSSFG] Frame Generation is unavailable on this system." << std::endl;
        return;
    }

    available_ = true;
}

void DlssFrameGenerationController::destroy() {
    destroyFeature();
    destroyResources();
    depthConversionPipeline_ = nullptr;
    ngxContext_ = nullptr;
    framework_.reset();
    available_ = false;
    firstFrame_ = true;
    loggedFeatureCreation_ = false;
    loggedInterpolationActive_ = false;
    loggedInterpolationDisabled_ = false;
    loggedBackpressureFallback_ = false;
    hasDisableInterpolationSignal_ = false;
    lastDisableInterpolationRequested_ = false;
    backbufferFrameId_ = 0;
}

bool DlssFrameGenerationController::isAvailable() const {
    return available_;
}

void DlssFrameGenerationController::advanceBackbufferFrameId() {
    backbufferFrameId_++;
}

void DlssFrameGenerationController::invalidateFeature() {
    destroyFeature();
    destroyResources();
}

bool DlssFrameGenerationController::supportsStorageFormat(VkFormat format) const {
    auto framework = framework_.lock();
    if (!framework) { return false; }

    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(framework->physicalDevice()->vkPhysicalDevice(), format, &props);
    return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

bool DlssFrameGenerationController::ensureDepthConversionPipeline() {
    if (depthConversionPipeline_ != nullptr) { return true; }

    auto framework = framework_.lock();
    if (!framework) { return false; }

    auto shader = vk::Shader::create(
        framework->device(), (Renderer::folderPath / "shaders/world/upscaler/linear_to_device_depth_comp.spv").string());
    depthConversionPipeline_ = vk::ComputePipelineBuilder{}
                                   .defineShader(shader)
                                   .definePipelineLayout(depthDescriptorTables_[0])
                                   .build(framework->device());
    return depthConversionPipeline_ != nullptr;
}

void DlssFrameGenerationController::destroyResources() {
    deviceDepthImages_.clear();
    preparedMotionVectorImages_.clear();
    interpolatedOutputImages_.clear();
    disableInterpolationBuffers_.clear();
    fgCompletionFences_.clear();
    fgCompletionPending_.clear();
    depthDescriptorTables_.clear();
    depthConversionPipeline_ = nullptr;
    renderWidth_ = 0;
    renderHeight_ = 0;
    displayWidth_ = 0;
    displayHeight_ = 0;
    backbufferFormat_ = VK_FORMAT_UNDEFINED;
    hasDisableInterpolationSignal_ = false;
    lastDisableInterpolationRequested_ = false;
    loggedBackpressureFallback_ = false;
}

bool DlssFrameGenerationController::ensureResources(uint32_t renderWidth, uint32_t renderHeight) {
    auto framework = framework_.lock();
    if (!framework || !available_) { return false; }

    VkExtent2D extent = framework->swapchain()->vkExtent();
    VkFormat swapchainFormat = framework->swapchain()->vkSurfaceFormat().format;

    if (renderWidth == 0 || renderHeight == 0 || extent.width == 0 || extent.height == 0) { return false; }
    if (!supportsStorageFormat(swapchainFormat)) {
        std::cerr << "[DLSSFG] Swapchain format does not support storage images: " << swapchainFormat << std::endl;
        available_ = false;
        return false;
    }

    bool needsRebuild = renderWidth_ != renderWidth || renderHeight_ != renderHeight || displayWidth_ != extent.width ||
                        displayHeight_ != extent.height || backbufferFormat_ != swapchainFormat ||
                        interpolatedOutputImages_.size() != framework->swapchain()->imageCount();
    if (!needsRebuild) { return true; }

    destroyFeature();
    destroyResources();

    renderWidth_ = renderWidth;
    renderHeight_ = renderHeight;
    displayWidth_ = extent.width;
    displayHeight_ = extent.height;
    backbufferFormat_ = swapchainFormat;

    uint32_t size = framework->swapchain()->imageCount();
    depthDescriptorTables_.resize(size);
    deviceDepthImages_.resize(size);
    preparedMotionVectorImages_.resize(size);
    interpolatedOutputImages_.resize(size);
    disableInterpolationBuffers_.resize(size);
    fgCompletionFences_.resize(size);
    fgCompletionPending_.assign(size, false);

    for (uint32_t i = 0; i < size; i++) {
        depthDescriptorTables_[i] = vk::DescriptorTableBuilder{}
                                        .beginDescriptorLayoutSet()
                                        .beginDescriptorLayoutSetBinding()
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 0,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 2,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 3,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .endDescriptorLayoutSetBinding()
                                        .endDescriptorLayoutSet()
                                        .definePushConstant({
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                            .offset = 0,
                                            .size = sizeof(float) * 4 + sizeof(uint32_t) * 2,
                                        })
                                        .build(framework->device());

        deviceDepthImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        preparedMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        interpolatedOutputImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, displayWidth_, displayHeight_, 1, backbufferFormat_,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        disableInterpolationBuffers_[i] =
            vk::HostVisibleBuffer::create(framework->vma(), framework->device(), 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        fgCompletionFences_[i] = vk::Fence::create(framework->device(), true);
    }

    return ensureDepthConversionPipeline();
}

void DlssFrameGenerationController::destroyFeature() {
    if (dlssgHandle_ != nullptr) {
        NVSDK_NGX_VULKAN_ReleaseFeature(dlssgHandle_);
        dlssgHandle_ = nullptr;
    }
}

bool DlssFrameGenerationController::pollFrameGenerationCompletion(uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (!framework || frameIndex >= fgCompletionFences_.size() || !fgCompletionPending_[frameIndex] ||
        fgCompletionFences_[frameIndex] == nullptr) {
        return false;
    }

    VkResult fenceStatus = vkGetFenceStatus(framework->device()->vkDevice(), fgCompletionFences_[frameIndex]->vkFence());
    if (fenceStatus == VK_NOT_READY) { return false; }
    if (fenceStatus != VK_SUCCESS) {
        std::cerr << "[DLSSFG] Failed to query FG fence state: " << fenceStatus << std::endl;
        fgCompletionPending_[frameIndex] = false;
        return false;
    }

    disableInterpolationBuffers_[frameIndex]->downloadFromBuffer();
    bool disableInterpolation =
        (*reinterpret_cast<uint8_t *>(disableInterpolationBuffers_[frameIndex]->mappedPtr())) != 0;
    fgCompletionPending_[frameIndex] = false;
    hasDisableInterpolationSignal_ = true;
    lastDisableInterpolationRequested_ = disableInterpolation;

    if (disableInterpolation) {
        if (!loggedInterpolationDisabled_) {
            std::cout << "[DLSSFG] NGX requested real-frame-only present for the current sequence; "
                         "generated frames are temporarily disabled." << std::endl;
            loggedInterpolationDisabled_ = true;
        }
    } else {
        loggedInterpolationDisabled_ = false;
    }

    return true;
}

void DlssFrameGenerationController::refreshCompletedFrameGenerationWork() {
    for (uint32_t i = 0; i < fgCompletionPending_.size(); i++) { pollFrameGenerationCompletion(i); }
}

bool DlssFrameGenerationController::prepareFrameGenerationSlot(uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (!framework || frameIndex >= fgCompletionFences_.size() || fgCompletionFences_[frameIndex] == nullptr) {
        return false;
    }

    refreshCompletedFrameGenerationWork();
    if (fgCompletionPending_[frameIndex] && !pollFrameGenerationCompletion(frameIndex)) {
        if (!loggedBackpressureFallback_) {
            std::cout << "[DLSSFG] Skipping generated-frame submission this frame to protect base FPS; "
                         "the previous FG workload has not retired yet." << std::endl;
            loggedBackpressureFallback_ = true;
        }
        return false;
    }

    loggedBackpressureFallback_ = false;
    VkResult resetResult = vkResetFences(framework->device()->vkDevice(), 1, &fgCompletionFences_[frameIndex]->vkFence());
    if (resetResult != VK_SUCCESS) {
        std::cerr << "[DLSSFG] Failed to reset FG fence: " << resetResult << std::endl;
        return false;
    }

    return true;
}

bool DlssFrameGenerationController::ensureFeature(uint32_t renderWidth, uint32_t renderHeight) {
    if (dlssgHandle_ != nullptr) { return true; }

    auto framework = framework_.lock();
    if (!framework || !ngxContext_ || !ensureResources(renderWidth, renderHeight)) { return false; }

    NVSDK_NGX_Parameter *params = ngxContext_->params();
    if (!params) { return false; }

    NVSDK_NGX_DLSSG_Create_Params createParams{};
    createParams.Width = displayWidth_;
    createParams.Height = displayHeight_;
    createParams.NativeBackbufferFormat = static_cast<unsigned int>(backbufferFormat_);
    createParams.RenderWidth = renderWidth_;
    createParams.RenderHeight = renderHeight_;
    createParams.DynamicResolutionScaling = (renderWidth_ != displayWidth_) || (renderHeight_ != displayHeight_);

    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_DLSSG_Parameter_AsyncCreateEnabled, 1);

    std::shared_ptr<vk::CommandBuffer> cmdBuffer = vk::CommandBuffer::create(framework->device(), framework->mainCommandPool());
    cmdBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    NVSDK_NGX_Result result = NGX_VK_CREATE_DLSSG(cmdBuffer->vkCommandBuffer(), 0x1, 0x1, &dlssgHandle_, params,
                                                  &createParams);
    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[DLSSFG] Failed to create Frame Generation feature: " << getNGXResultString(result) << std::endl;
        dlssgHandle_ = nullptr;
        return false;
    }

    std::shared_ptr<vk::Fence> createFence = vk::Fence::create(framework->device());
    cmdBuffer->end()->submitMainQueueIndividual(framework->device(), createFence);
    VkResult fenceResult =
        vkWaitForFences(framework->device()->vkDevice(), 1, &createFence->vkFence(), VK_TRUE, UINT64_MAX);
    if (fenceResult != VK_SUCCESS) {
        std::cerr << "[DLSSFG] Failed waiting for feature creation fence: " << fenceResult << std::endl;
        destroyFeature();
        return false;
    }

    if (!loggedFeatureCreation_) {
        std::cout << "[DLSSFG] Feature created successfully. Render "
                  << renderWidth_ << "x" << renderHeight_
                  << " -> Display " << displayWidth_ << "x" << displayHeight_ << std::endl;
        loggedFeatureCreation_ = true;
    }

    return true;
}

bool DlssFrameGenerationController::present(std::shared_ptr<FrameworkContext> context,
                                            std::shared_ptr<PipelineContext> pipelineContext) {
    if (!available_ || !Renderer::options.dlssFrameGeneration || !context || !pipelineContext ||
        pipelineContext->worldPipelineContext == nullptr) {
        return false;
    }

    auto framework = framework_.lock();
    if (!framework || !framework->isRunning() || !ngxContext_ || !ngxContext_->isInitialized()) { return false; }

    auto worldPipelineContext = pipelineContext->worldPipelineContext;
    auto hudlessImage = worldPipelineContext->outputImage;
    auto linearDepthImage = worldPipelineContext->linearDepthImage;
    auto motionVectorImage = worldPipelineContext->motionVectorImage;
    auto backbufferImage = context->swapchainImage;

    if (!hudlessImage || !linearDepthImage || !motionVectorImage || !backbufferImage) { return false; }

    uint32_t frameIndex = context->frameIndex;
    uint32_t renderWidth = motionVectorImage->width();
    uint32_t renderHeight = motionVectorImage->height();
    if (!ensureFeature(renderWidth, renderHeight)) { return false; }
    if (!prepareFrameGenerationSlot(frameIndex)) {
        firstFrame_ = true;
        return false;
    }

    auto deviceDepthImage = deviceDepthImages_[frameIndex];
    auto preparedMotionVectorImage = preparedMotionVectorImages_[frameIndex];
    auto interpolatedOutputImage = interpolatedOutputImages_[frameIndex];
    auto disableInterpolationBuffer = disableInterpolationBuffers_[frameIndex];

    std::memset(disableInterpolationBuffer->mappedPtr(), 0, disableInterpolationBuffer->size());
    disableInterpolationBuffer->flush();

    auto buffers = Renderer::instance().buffers();
    auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
    auto lastWorldUBO = static_cast<vk::Data::WorldUBO *>(buffers->lastWorldUniformBuffer()->mappedPtr());
    if (!worldUBO || !lastWorldUBO) { return false; }

    VkImageLayout initialBackbufferLayout = backbufferImage->imageLayout();
    VkImageLayout initialHudlessLayout = hudlessImage->imageLayout();
    VkImageLayout initialLinearDepthLayout = linearDepthImage->imageLayout();
    VkImageLayout initialMotionVectorLayout = motionVectorImage->imageLayout();
    VkImageLayout initialDeviceDepthLayout = deviceDepthImage->imageLayout();
    VkImageLayout initialPreparedMotionVectorLayout = preparedMotionVectorImage->imageLayout();
    VkImageLayout initialInterpolatedOutputLayout = interpolatedOutputImage->imageLayout();

    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    std::shared_ptr<vk::CommandBuffer> fgCommandBuffer =
        vk::CommandBuffer::create(framework->device(), framework->mainCommandPool());
    fgCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    fgCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = initialLinearDepthLayout,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = linearDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = initialMotionVectorLayout,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = motionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = deviceDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = deviceDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = preparedMotionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = preparedMotionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = initialBackbufferLayout,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = backbufferImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = initialHudlessLayout,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = hudlessImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = interpolatedOutputImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = interpolatedOutputImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    linearDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    motionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    deviceDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    preparedMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    backbufferImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    hudlessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    interpolatedOutputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    depthDescriptorTables_[frameIndex]->bindImage(linearDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
    depthDescriptorTables_[frameIndex]->bindImage(deviceDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    depthDescriptorTables_[frameIndex]->bindImage(motionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    depthDescriptorTables_[frameIndex]->bindImage(preparedMotionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 3);

    struct PushConstants {
        float cameraNear;
        float cameraFar;
        uint32_t width;
        uint32_t height;
        float jitterX;
        float jitterY;
    } pushConstants{0.1f, 10000.0f, renderWidth_, renderHeight_, worldUBO->cameraJitter.x, worldUBO->cameraJitter.y};

    fgCommandBuffer->bindDescriptorTable(depthDescriptorTables_[frameIndex], VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(depthConversionPipeline_);
    vkCmdPushConstants(fgCommandBuffer->vkCommandBuffer(), depthDescriptorTables_[frameIndex]->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(fgCommandBuffer->vkCommandBuffer(), (renderWidth_ + 15) / 16, (renderHeight_ + 15) / 16, 1);

    fgCommandBuffer->barriersMemory({{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT}});

    NVSDK_NGX_Resource_VK backbufferResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        backbufferImage->vkImageView(), backbufferImage->vkImage(), vk::wholeColorSubresourceRange,
        backbufferImage->vkFormat(), displayWidth_, displayHeight_, false);
    NVSDK_NGX_Resource_VK depthResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        deviceDepthImage->vkImageView(), deviceDepthImage->vkImage(), vk::wholeColorSubresourceRange,
        deviceDepthImage->vkFormat(), renderWidth_, renderHeight_, false);
    NVSDK_NGX_Resource_VK motionVectorResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        preparedMotionVectorImage->vkImageView(), preparedMotionVectorImage->vkImage(), vk::wholeColorSubresourceRange,
        preparedMotionVectorImage->vkFormat(), renderWidth_, renderHeight_, false);
    NVSDK_NGX_Resource_VK hudlessResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        hudlessImage->vkImageView(), hudlessImage->vkImage(), vk::wholeColorSubresourceRange, hudlessImage->vkFormat(),
        displayWidth_, displayHeight_, false);
    NVSDK_NGX_Resource_VK interpolatedOutputResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        interpolatedOutputImage->vkImageView(), interpolatedOutputImage->vkImage(), vk::wholeColorSubresourceRange,
        interpolatedOutputImage->vkFormat(), displayWidth_, displayHeight_, true);
    NVSDK_NGX_Resource_VK disableInterpolationResource = NVSDK_NGX_Create_Buffer_Resource_VK(
        disableInterpolationBuffer->vkBuffer(), static_cast<unsigned int>(disableInterpolationBuffer->size()), true);

    NVSDK_NGX_VK_DLSSG_Eval_Params evalParams{};
    evalParams.pBackbuffer = &backbufferResource;
    evalParams.pDepth = &depthResource;
    evalParams.pMVecs = &motionVectorResource;
    evalParams.pHudless = &hudlessResource;
    evalParams.pUI = nullptr;
    evalParams.pNoPostProcessingColor = nullptr;
    evalParams.pBidirectionalDistortionField = nullptr;
    evalParams.pOutputInterpFrame = &interpolatedOutputResource;
    evalParams.pOutputRealFrame = nullptr;
    evalParams.pOutputDisableInterpolation = &disableInterpolationResource;

    glm::mat4 clipToPrevClip = lastWorldUBO->cameraProjMat * lastWorldUBO->cameraEffectedViewMat *
                               worldUBO->cameraEffectedViewMatInv * worldUBO->cameraProjMatInv;
    glm::mat4 prevClipToClip = worldUBO->cameraProjMat * worldUBO->cameraEffectedViewMat *
                               lastWorldUBO->cameraEffectedViewMatInv * lastWorldUBO->cameraProjMatInv;
    glm::mat4 clipToLensClip(1.0f);

    glm::vec3 cameraRight =
        normalizeOrFallback(glm::vec3(worldUBO->cameraEffectedViewMatInv[0][0], worldUBO->cameraEffectedViewMatInv[1][0],
                                      worldUBO->cameraEffectedViewMatInv[2][0]),
                                                glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 cameraUp =
        normalizeOrFallback(glm::vec3(worldUBO->cameraEffectedViewMatInv[0][1], worldUBO->cameraEffectedViewMatInv[1][1],
                                      worldUBO->cameraEffectedViewMatInv[2][1]),
                                             glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 cameraForward = normalizeOrFallback(
        -glm::vec3(worldUBO->cameraEffectedViewMatInv[0][2], worldUBO->cameraEffectedViewMatInv[1][2],
                   worldUBO->cameraEffectedViewMatInv[2][2]),
        glm::vec3(0.0f, 0.0f, -1.0f));
    glm::vec3 lastCameraForward = normalizeOrFallback(
        -glm::vec3(lastWorldUBO->cameraEffectedViewMatInv[0][2], lastWorldUBO->cameraEffectedViewMatInv[1][2],
                   lastWorldUBO->cameraEffectedViewMatInv[2][2]),
        glm::vec3(0.0f, 0.0f, -1.0f));
    float cameraTravel = glm::length(glm::vec3(worldUBO->cameraPos) - glm::vec3(lastWorldUBO->cameraPos));
    float forwardDot = glm::dot(cameraForward, lastCameraForward);
    bool shouldReset = firstFrame_ || cameraTravel > 6.0f || forwardDot < 0.35f;

    NVSDK_NGX_DLSSG_Opt_Eval_Params optEval{};
    std::memcpy(optEval.cameraViewToClip, glm::value_ptr(worldUBO->cameraProjMat), sizeof(glm::mat4));
    std::memcpy(optEval.clipToCameraView, glm::value_ptr(worldUBO->cameraProjMatInv), sizeof(glm::mat4));
    std::memcpy(optEval.clipToLensClip, glm::value_ptr(clipToLensClip), sizeof(glm::mat4));
    std::memcpy(optEval.clipToPrevClip, glm::value_ptr(clipToPrevClip), sizeof(glm::mat4));
    std::memcpy(optEval.prevClipToClip, glm::value_ptr(prevClipToClip), sizeof(glm::mat4));

    optEval.multiFrameCount = 1;
    optEval.multiFrameIndex = 1;
    optEval.jitterOffset[0] =
        renderWidth_ > 0 ? (2.0f * worldUBO->cameraJitter.x / static_cast<float>(renderWidth_)) : 0.0f;
    optEval.jitterOffset[1] =
        renderHeight_ > 0 ? (2.0f * worldUBO->cameraJitter.y / static_cast<float>(renderHeight_)) : 0.0f;
    optEval.mvecScale[0] = renderWidth_ > 0 ? (1.0f / static_cast<float>(renderWidth_)) : 0.0f;
    optEval.mvecScale[1] = renderHeight_ > 0 ? (1.0f / static_cast<float>(renderHeight_)) : 0.0f;
    optEval.cameraPinholeOffset[0] = 0.0f;
    optEval.cameraPinholeOffset[1] = 0.0f;
    optEval.cameraPos[0] = static_cast<float>(worldUBO->cameraPos.x);
    optEval.cameraPos[1] = static_cast<float>(worldUBO->cameraPos.y);
    optEval.cameraPos[2] = static_cast<float>(worldUBO->cameraPos.z);
    optEval.cameraUp[0] = cameraUp.x;
    optEval.cameraUp[1] = cameraUp.y;
    optEval.cameraUp[2] = cameraUp.z;
    optEval.cameraRight[0] = cameraRight.x;
    optEval.cameraRight[1] = cameraRight.y;
    optEval.cameraRight[2] = cameraRight.z;
    optEval.cameraFwd[0] = cameraForward.x;
    optEval.cameraFwd[1] = cameraForward.y;
    optEval.cameraFwd[2] = cameraForward.z;
    optEval.cameraNear = 0.1f;
    optEval.cameraFar = 10000.0f;
    optEval.cameraFOV = 2.0f * std::atan(1.0f / std::abs(worldUBO->cameraProjMat[1][1]));
    optEval.cameraAspectRatio =
        displayHeight_ > 0 ? (static_cast<float>(displayWidth_) / static_cast<float>(displayHeight_)) : 1.0f;
    optEval.colorBuffersHDR = framework->swapchain()->isHdrOutputActive();
    optEval.depthInverted = false;
    optEval.cameraMotionIncluded = true;
    optEval.reset = shouldReset;
    optEval.automodeOverrideReset = false;
    optEval.notRenderingGameFrames = false;
    optEval.orthoProjection = false;
    optEval.motionVectorsInvalidValue = 0.0f;
    optEval.motionVectorsDilated = false;
    optEval.menuDetectionEnabled = false;
    optEval.mvecsSubrectBase = {0, 0};
    optEval.mvecsSubrectSize = {renderWidth_, renderHeight_};
    optEval.depthSubrectBase = {0, 0};
    optEval.depthSubrectSize = {renderWidth_, renderHeight_};
    optEval.hudLessSubrectBase = {0, 0};
    optEval.hudLessSubrectSize = {displayWidth_, displayHeight_};
    optEval.uiSubrectBase = {0, 0};
    optEval.uiSubrectSize = {0, 0};
    optEval.backbufferSubrectBase = {0, 0};
    optEval.backbufferSubrectSize = {displayWidth_, displayHeight_};

    NVSDK_NGX_Parameter *params = ngxContext_->params();
    NVSDK_NGX_Parameter_SetULL(params, NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, backbufferFrameId_++);
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_DLSSG_Parameter_EvalFlags,
                              NVSDK_NGX_DLSSG_EvalFlags_UpdateOnlyInsideExtents);

    NVSDK_NGX_Result evalResult =
        NGX_VK_EVALUATE_DLSSG(fgCommandBuffer->vkCommandBuffer(), dlssgHandle_, params, &evalParams, &optEval);
    if (NVSDK_NGX_FAILED(evalResult)) {
        std::cerr << "[DLSSFG] Evaluate failed: " << getNGXResultString(evalResult) << std::endl;

        fgCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .oldLayout = backbufferImage->imageLayout(),
                  .newLayout = initialBackbufferLayout,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = backbufferImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .oldLayout = hudlessImage->imageLayout(),
                  .newLayout = initialHudlessLayout,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = hudlessImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .oldLayout = linearDepthImage->imageLayout(),
                  .newLayout = initialLinearDepthLayout,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = linearDepthImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .oldLayout = motionVectorImage->imageLayout(),
                  .newLayout = initialMotionVectorLayout,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = motionVectorImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});
        backbufferImage->imageLayout() = initialBackbufferLayout;
        hudlessImage->imageLayout() = initialHudlessLayout;
        linearDepthImage->imageLayout() = initialLinearDepthLayout;
        motionVectorImage->imageLayout() = initialMotionVectorLayout;
        deviceDepthImage->imageLayout() = initialDeviceDepthLayout;
        preparedMotionVectorImage->imageLayout() = initialPreparedMotionVectorLayout;
        interpolatedOutputImage->imageLayout() = initialInterpolatedOutputLayout;
        fgCommandBuffer->end();
        firstFrame_ = true;
        return false;
    }

    fgCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = backbufferImage->imageLayout(),
              .newLayout = targetPresentLayout(),
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = backbufferImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = hudlessImage->imageLayout(),
              .newLayout = initialHudlessLayout,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = hudlessImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = linearDepthImage->imageLayout(),
              .newLayout = initialLinearDepthLayout,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = linearDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = motionVectorImage->imageLayout(),
              .newLayout = initialMotionVectorLayout,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = motionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});
    backbufferImage->imageLayout() = targetPresentLayout();
    hudlessImage->imageLayout() = initialHudlessLayout;
    linearDepthImage->imageLayout() = initialLinearDepthLayout;
    motionVectorImage->imageLayout() = initialMotionVectorLayout;

    std::shared_ptr<vk::Semaphore> fgReadySemaphore = vk::Semaphore::create(framework->device());
    std::shared_ptr<vk::Fence> fgReadyFence = fgCompletionFences_[frameIndex];
    fgCommandBuffer->end()->submitMainQueue(
        framework->device(),
        {.waitSemaphoresAndStageMasks =
             {{context->commandProcessedSemaphore->vkSemaphore(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}},
         .signalSemaphores = {fgReadySemaphore->vkSemaphore()},
         .signalFence = fgReadyFence->vkFence()});
    fgCompletionPending_[frameIndex] = true;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &fgReadySemaphore->vkSemaphore();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &framework->swapchain()->vkSwapchain();
    presentInfo.pImageIndices = &context->frameIndex;

    VkResult presentResult = vkQueuePresentKHR(framework->device()->mainVkQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || vk::Window::framebufferResized ||
        Renderer::options.needRecreate || framework->pipeline()->needRecreate) {
        framework->recreate();
        firstFrame_ = true;
        return true;
    }
    if (presentResult != VK_SUCCESS) {
        std::cerr << "[DLSSFG] failed to present real frame: " << presentResult << std::endl;
        return false;
    }

    pollFrameGenerationCompletion(frameIndex);
    bool disableInterpolation = hasDisableInterpolationSignal_ && lastDisableInterpolationRequested_;
    firstFrame_ = false;
    if (disableInterpolation) {
        return true;
    }
    loggedInterpolationDisabled_ = false;

    std::shared_ptr<vk::Semaphore> acquireSemaphore = vk::Semaphore::create(framework->device());
    uint32_t interpolationImageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(framework->device()->vkDevice(), framework->swapchain()->vkSwapchain(),
                                                   0, acquireSemaphore->vkSemaphore(), VK_NULL_HANDLE,
                                                   &interpolationImageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        framework->recreate();
        return true;
    }
    if (acquireResult == VK_NOT_READY || acquireResult == VK_TIMEOUT) {
        if (!loggedBackpressureFallback_) {
            std::cout << "[DLSSFG] Skipping interpolated present this frame to protect base FPS; "
                         "no swapchain image was immediately available." << std::endl;
            loggedBackpressureFallback_ = true;
        }
        return true;
    }
    if (acquireResult != VK_SUCCESS) {
        std::cerr << "[DLSSFG] failed to acquire swapchain image for interpolated frame: " << acquireResult
                  << std::endl;
        return true;
    }
    loggedBackpressureFallback_ = false;

    auto interpolationSwapchainImage = framework->swapchain()->swapchainImages()[interpolationImageIndex];
    VkImageLayout initialInterpolationLayout = interpolationSwapchainImage->imageLayout();

    std::shared_ptr<vk::CommandBuffer> interpolationCommandBuffer =
        vk::CommandBuffer::create(framework->device(), framework->mainCommandPool());
    interpolationCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    interpolationCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
              .oldLayout = interpolatedOutputImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = interpolatedOutputImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .oldLayout = initialInterpolationLayout,
              .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = interpolationSwapchainImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});
    interpolatedOutputImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    interpolationSwapchainImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkImageBlit interpolationBlit{};
    interpolationBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    interpolationBlit.srcOffsets[1] = {static_cast<int32_t>(displayWidth_), static_cast<int32_t>(displayHeight_), 1};
    interpolationBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    interpolationBlit.dstOffsets[1] = {static_cast<int32_t>(displayWidth_), static_cast<int32_t>(displayHeight_), 1};
    vkCmdBlitImage(interpolationCommandBuffer->vkCommandBuffer(), interpolatedOutputImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, interpolationSwapchainImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &interpolationBlit,
                   chooseCopyFilter(displayWidth_, displayHeight_, displayWidth_, displayHeight_));

    interpolationCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = interpolationSwapchainImage->imageLayout(),
              .newLayout = targetPresentLayout(),
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = interpolationSwapchainImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});
    interpolationSwapchainImage->imageLayout() = targetPresentLayout();

    std::shared_ptr<vk::Semaphore> interpolationReadySemaphore = vk::Semaphore::create(framework->device());
    interpolationCommandBuffer->end()->submitMainQueue(
        framework->device(),
        {.waitSemaphoresAndStageMasks = {{acquireSemaphore->vkSemaphore(), VK_PIPELINE_STAGE_TRANSFER_BIT}},
         .signalSemaphores = {interpolationReadySemaphore->vkSemaphore()},
         .signalFence = VK_NULL_HANDLE});

    VkPresentInfoKHR interpolationPresentInfo{};
    interpolationPresentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    interpolationPresentInfo.waitSemaphoreCount = 1;
    interpolationPresentInfo.pWaitSemaphores = &interpolationReadySemaphore->vkSemaphore();
    interpolationPresentInfo.swapchainCount = 1;
    interpolationPresentInfo.pSwapchains = &framework->swapchain()->vkSwapchain();
    interpolationPresentInfo.pImageIndices = &interpolationImageIndex;

    VkResult interpolationPresentResult =
        vkQueuePresentKHR(framework->device()->mainVkQueue(), &interpolationPresentInfo);
    if (interpolationPresentResult == VK_ERROR_OUT_OF_DATE_KHR || interpolationPresentResult == VK_SUBOPTIMAL_KHR ||
        vk::Window::framebufferResized || Renderer::options.needRecreate || framework->pipeline()->needRecreate) {
        framework->recreate();
        return true;
    }
    if (interpolationPresentResult != VK_SUCCESS) {
        std::cerr << "[DLSSFG] failed to present interpolated frame: " << interpolationPresentResult << std::endl;
    }

    if (!loggedInterpolationActive_) {
        std::cout << "[DLSSFG] Interpolated frames are being presented. In-game FPS counters may still reflect "
                     "rendered frames instead of displayed frames." << std::endl;
        loggedInterpolationActive_ = true;
    }

    return true;
}
