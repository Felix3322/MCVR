#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

class Framework;
struct FrameworkContext;
struct PipelineContext;

class DlssFrameGenerationController : public SharedObject<DlssFrameGenerationController> {
  public:
    DlssFrameGenerationController();
    ~DlssFrameGenerationController();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<NgxContext> ngxContext);
    void destroy();

    bool isAvailable() const;
    void advanceBackbufferFrameId();
    void invalidateFeature();

    bool present(std::shared_ptr<FrameworkContext> context, std::shared_ptr<PipelineContext> pipelineContext);

  private:
    bool ensureDepthConversionPipeline();
    bool ensureResources(uint32_t renderWidth, uint32_t renderHeight);
    bool ensureFeature(uint32_t renderWidth, uint32_t renderHeight);
    void destroyFeature();
    void destroyResources();
    bool supportsStorageFormat(VkFormat format) const;
    bool pollFrameGenerationCompletion(uint32_t frameIndex);
    void refreshCompletedFrameGenerationWork();
    bool prepareFrameGenerationSlot(uint32_t frameIndex);

  private:
    std::weak_ptr<Framework> framework_;
    std::shared_ptr<NgxContext> ngxContext_;

    bool available_ = false;
    bool firstFrame_ = true;
    bool loggedFeatureCreation_ = false;
    bool loggedInterpolationActive_ = false;
    bool loggedInterpolationDisabled_ = false;
    bool loggedBackpressureFallback_ = false;
    bool hasDisableInterpolationSignal_ = false;
    bool lastDisableInterpolationRequested_ = false;
    uint64_t backbufferFrameId_ = 0;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t displayWidth_ = 0;
    uint32_t displayHeight_ = 0;
    VkFormat backbufferFormat_ = VK_FORMAT_UNDEFINED;

    NVSDK_NGX_Handle *dlssgHandle_ = nullptr;

    std::vector<std::shared_ptr<vk::DescriptorTable>> depthDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> depthConversionPipeline_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> deviceDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> preparedMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> interpolatedOutputImages_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> disableInterpolationBuffers_;
    std::vector<std::shared_ptr<vk::Fence>> fgCompletionFences_;
    std::vector<bool> fgCompletionPending_;
};
