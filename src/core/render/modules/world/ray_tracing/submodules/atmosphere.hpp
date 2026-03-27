#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <array>

class Framework;
struct FrameworkContext;
class RayTracingModule;
struct RayTracingModuleContext;

struct AtmosphereContext;

class Atmosphere : public SharedObject<Atmosphere> {
    friend RayTracingModule;
    friend RayTracingModuleContext;
    friend AtmosphereContext;

  public:
    Atmosphere();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule);

    void build();

    void setPlanetRadius(float value);
    void setAtmosphereTopRadius(float value);
    void setRayleighScaleHeight(float value);
    void setMieScaleHeight(float value);
    void setRayleighScatteringCoefficient(const glm::vec3 &value);
    void setMieAnisotropy(float value);
    void setMieScatteringCoefficient(const glm::vec3 &value);
    void setMinimumViewCosine(float value);
    void setSunRadiance(const glm::vec3 &value);
    void setMoonRadiance(const glm::vec3 &value);
    void applyToSkyUbo(vk::Data::SkyUBO &ubo) const;

  private:
    void initDescriptorTables();
    void initImages();
    void initAtmLUTRenderPass();
    void initAtmCubeMapRenderPass();
    void initFrameBuffers();
    void initAtmLUTPipeline();
    void initAtmCubeMapPipeline();

  private:
    bool lutRendered_ = false;

    std::weak_ptr<Framework> framework_;
    std::weak_ptr<RayTracingModule> rayTracingModule_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> atmDescriptorTables_;

    std::shared_ptr<vk::Shader> atmLUTVertShader_;
    std::shared_ptr<vk::Shader> atmLUTFragShader_;
    std::shared_ptr<vk::DeviceLocalImage> atmLUTImage_;
    std::shared_ptr<vk::Sampler> atmLUTImageSampler_;
    std::shared_ptr<vk::RenderPass> atmLUTRenderPass_;
    std::shared_ptr<vk::Framebuffer> atmLUTFramebuffer_;
    std::shared_ptr<vk::GraphicsPipeline> atmLUTPipeline_;

    std::shared_ptr<vk::Shader> atmCubeMapVertShader_;
    std::shared_ptr<vk::Shader> atmCubeMapFragShader_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> atmCubeMapImages_;
    std::vector<std::shared_ptr<vk::Sampler>> atmCubeMapImageSamplers_;
    std::shared_ptr<vk::RenderPass> atmCubeMapRenderPass_;
    std::vector<std::array<std::shared_ptr<vk::Framebuffer>, 6>> atmCubeMapFramebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> atmCubeMapPipeline_;

    float planetRadius_ = 6360000.0f;
    float atmosphereTopRadius_ = 6460000.0f;
    float rayleighScaleHeight_ = 8000.0f;
    float mieScaleHeight_ = 1200.0f;
    glm::vec3 rayleighScatteringCoefficient_ = glm::vec3(5.802e-6f, 13.558e-6f, 33.100e-6f);
    float mieAnisotropy_ = 0.80f;
    glm::vec3 mieScatteringCoefficient_ = glm::vec3(21.000e-6f, 21.000e-6f, 21.000e-6f);
    float minimumViewCosine_ = 0.02f;
    glm::vec3 sunRadiance_ = glm::vec3(16.0f);
    glm::vec3 moonRadiance_ = glm::vec3(0.08f, 0.1f, 0.2f);

    std::vector<std::shared_ptr<AtmosphereContext>> contexts_;
};

struct AtmosphereContext : public SharedObject<AtmosphereContext> {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<RayTracingModuleContext> rayTracingModuleContext;
    std::weak_ptr<Atmosphere> atmosphere;

    std::shared_ptr<vk::DescriptorTable> atmDescriptorTable;

    std::shared_ptr<vk::DeviceLocalImage> atmLUTImage;
    std::shared_ptr<vk::Framebuffer> atmLUTFramebuffer;

    std::shared_ptr<vk::DeviceLocalImage> atmCubeMapImage;
    std::array<std::shared_ptr<vk::Framebuffer>, 6> atmCubeMapFramebuffer;

    AtmosphereContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<Atmosphere> atmosphere);

    void render();
};
