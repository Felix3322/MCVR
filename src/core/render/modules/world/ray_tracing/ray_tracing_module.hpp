#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"

#include <filesystem>
#include <optional>
#include <unordered_map>

class Framework;
struct FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct RayTracingModuleContext;

class Atmosphere;
class AtmosphereContext;
class WorldPrepare;
class WorldPrepareContext;

struct RayTracingPushConstant {
    uint32_t numRayBounces;
    float directLightStrength;
    float indirectLightStrength;
    float basicRadiance;
    uint32_t pbrSamplingMode;
    uint32_t transparentSplitMode;
    float farFieldStartDistanceChunks;
    uint32_t farFieldMaterialMode;
    uint32_t flags;
};

enum RayTracingTransparentSplitMode {
    RAY_TRACING_TRANSPARENT_SPLIT_MODE_DETERMINISTIC = 0,
    RAY_TRACING_TRANSPARENT_SPLIT_MODE_STOCHASTIC = 1,
};

enum RayTracingFarFieldMaterialMode {
    RAY_TRACING_FAR_FIELD_MATERIAL_MODE_FULL_PBR = 0,
    RAY_TRACING_FAR_FIELD_MATERIAL_MODE_FLAT_SURFACE = 1,
};

class RayTracingModule : public WorldModule, public SharedObject<RayTracingModule> {
    friend RayTracingModuleContext;
    friend Atmosphere;
    friend AtmosphereContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.ray_tracing.name";
    constexpr static uint32_t inputImageNum = 0;
    constexpr static uint32_t outputImageNum = 15;

    RayTracingModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;

    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

    uint32_t hitGroupIndexForName(const std::string &groupName) const;
    uint32_t shadowHitGroupIndex() const;
    uint32_t fallbackHitGroupIndex() const;

  private:
    constexpr static uint32_t sharcCapacity = 1u << 22;
    constexpr static uint32_t sharcResolveWorkgroupSize = 64;

    struct SharcConfigData {
        uint32_t hashEntriesAddress[2];
        uint32_t lockAddress[2];
        uint32_t accumulationAddress[2];
        uint32_t resolvedAddress[2];
        glm::vec4 cameraPosition;
        glm::vec4 cameraPositionPrev;
        float sceneScale;
        float radianceScale;
        uint32_t accumulationFrameNum;
        uint32_t staleFrameNumMax;
        uint32_t capacity;
        uint32_t frameIndex;
        uint32_t enableAntiFireflyFilter;
        uint32_t useLockBuffer;
        uint32_t debugMode;
        uint32_t updateDownsampleFactor;
        glm::vec4 dirtyRegionCenterRadius;
        glm::uvec4 sceneState;
    };

  private:
    struct HitShaderPaths {
        std::optional<std::filesystem::path> anyHit;
        std::optional<std::filesystem::path> closestHit;
        std::optional<std::filesystem::path> intersection;
    };

    struct ParsedHitGroupConfig {
        std::string name;
        std::optional<std::filesystem::path> anyHit;
        std::optional<std::filesystem::path> closestHit;
        std::optional<std::filesystem::path> intersection;
        VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    };

    struct HitShaderGroupDefinition {
        std::string name;
        std::shared_ptr<vk::Shader> closestHitShader;
        std::shared_ptr<vk::Shader> anyHitShader;
        std::shared_ptr<vk::Shader> intersectionShader;
        VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    };

    struct MissShaderDefinition {
        std::string name;
        uint32_t index = 0;
        std::shared_ptr<vk::Shader> shader;
    };

  private:
    void initDescriptorTables();
    void initImages();
    void initPipeline();
    void initSBT();
    void initSharc();
    void updateSharcConfig(uint32_t frameIndex);

  private:
    // input
    // none

    // ray tracing
    std::shared_ptr<vk::Shader> worldRayGenUpdateShader_;
    std::shared_ptr<vk::Shader> worldRayGenQueryShader_;

    std::vector<MissShaderDefinition> missShaders_;
    std::vector<HitShaderGroupDefinition> hitShaderGroups_;
    std::unordered_map<std::string, uint32_t> hitGroupNameToIndex_;
    uint32_t shadowHitGroupIndex_ = 0;
    uint32_t fallbackHitGroupIndex_ = 0;
    uint32_t missGroupCount_ = 0;
    uint32_t hitGroupCount_ = 0;
    bool sharcCompatible_ = false;
    bool useSharcRuntime_ = false;

    std::shared_ptr<vk::Shader> worldPostColorToDepthVertShader_;
    std::shared_ptr<vk::Shader> worldPostColorToDepthFragShader_;
    std::shared_ptr<vk::Shader> worldPostVertShader_;
    std::shared_ptr<vk::Shader> worldPostFragShader_;
    std::shared_ptr<vk::Shader> worldToneMappingVertShader_;
    std::shared_ptr<vk::Shader> worldToneMappingFragShader_;
    std::shared_ptr<vk::Shader> radianceHistCompShader_;
    std::shared_ptr<vk::Shader> worldLightMapVertShader_;
    std::shared_ptr<vk::Shader> worldLightMapFragShader_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> rayTracingDescriptorTables_;
    std::shared_ptr<vk::RayTracingPipeline> rayTracingUpdatePipeline_;
    std::shared_ptr<vk::RayTracingPipeline> rayTracingQueryPipeline_;
    std::vector<std::shared_ptr<vk::SBT>> sharcUpdateSbts_;
    std::vector<std::shared_ptr<vk::SBT>> sharcQuerySbts_;

    std::shared_ptr<vk::Shader> sharcResolveCompShader_;
    std::shared_ptr<vk::ComputePipeline> sharcResolvePipeline_;

    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> sharcConfigBuffers_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcHashEntriesBuffer_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcLockBuffer_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcAccumulationBuffer_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcResolvedBuffer_;

    glm::dvec3 sharcPrevCameraPos_ = glm::dvec3(0.0);
    uint32_t sharcFrameIndex_ = 0;
    bool sharcFirstFrame_ = true;
    uint32_t sharcPrevLightRevision_ = 0;
    uint32_t sharcStableUpdateStride_ = 1;
    uint32_t sharcEffectiveUpdateDownsampleFactor_ = 1;

    uint32_t numRayBounces_ = 4;
    bool useJitter_ = true;
    float directLightStrength_ = 1.0f;
    float indirectLightStrength_ = 16.0f;
    float basicRadiance_ = 0.001f;
    uint32_t pbrSamplingMode_ = 1;
    uint32_t transparentSplitMode_ = RAY_TRACING_TRANSPARENT_SPLIT_MODE_DETERMINISTIC;
    float farFieldStartDistanceChunks_ = 32.0f;
    uint32_t farFieldMaterialMode_ = RAY_TRACING_FAR_FIELD_MATERIAL_MODE_FLAT_SURFACE;
    bool useSharc_ = true;
    uint32_t sharcDebugMode_ = 0;
    std::string shaderPackPath_;
    float sharcSceneScale_ = 64.0f;
    uint32_t sharcAccumulationFrameNum_ = 64;
    uint32_t sharcStaleFrameNumMax_ = 256;
    uint32_t sharcUpdateDownsampleFactor_ = 5;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrNoisyOutputImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDiffuseDirectLightImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDiffuseIndirectLightImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitSpecularImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitClearImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitBaseEmissionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> fogImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitRefractionImages_;

    // submodules
    std::shared_ptr<Atmosphere> atmosphere_;
    std::shared_ptr<WorldPrepare> worldPrepare_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

struct RayTracingModuleContext : public WorldModuleContext, SharedObject<RayTracingModuleContext> {
    std::weak_ptr<RayTracingModule> rayTracingModule;

    // input
    // none

    // ray tracing
    std::shared_ptr<vk::DescriptorTable> rayTracingDescriptorTable;
    std::shared_ptr<vk::SBT> sharcUpdateSbt;
    std::shared_ptr<vk::SBT> sharcQuerySbt;

    // output
    std::shared_ptr<vk::DeviceLocalImage> hdrNoisyOutputImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> specularHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDiffuseDirectLightImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDiffuseIndirectLightImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitSpecularImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitClearImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitBaseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> fogImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitRefractionImage;

    // submodule
    std::shared_ptr<AtmosphereContext> atmosphereContext;
    std::shared_ptr<WorldPrepareContext> worldPrepareContext;

    RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                            std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                            std::shared_ptr<RayTracingModule> rayTracingModule);

    void render() override;
};
