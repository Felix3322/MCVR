#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/modules/world/ray_tracing/submodules/atmosphere.hpp"
#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#include "core/render/chunks.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

RayTracingModule::RayTracingModule() {}

void RayTracingModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrNoisyOutputImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    specularHitDepthImages_.resize(size);
    firstHitDepthImages_.resize(size);
    firstHitDiffuseDirectLightImages_.resize(size);
    firstHitDiffuseIndirectLightImages_.resize(size);
    firstHitSpecularImages_.resize(size);
    firstHitClearImages_.resize(size);
    firstHitBaseEmissionImages_.resize(size);
    fogImages_.resize(size);
    firstHitRefractionImages_.resize(size);

    atmosphere_ = Atmosphere::create(framework, shared_from_this());
    worldPrepare_ = WorldPrepare::create(framework, shared_from_this());
}

bool RayTracingModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    return true;
}

bool RayTracingModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    uint32_t width, height;
    bool set = false;
    for (auto &image : images) {
        if (image != nullptr) {
            if (!set) {
                width = image->width();
                height = image->height();
                set = true;
            } else {
                if (image->width() != width || image->height() != height) { return false; }
            }
        }
    }

    if (!set) { return false; }

    auto framework = framework_.lock();
    for (int i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width, height, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    hdrNoisyOutputImages_[frameIndex] = images[0];
    diffuseAlbedoImages_[frameIndex] = images[1];
    specularAlbedoImages_[frameIndex] = images[2];
    normalRoughnessImages_[frameIndex] = images[3];
    motionVectorImages_[frameIndex] = images[4];
    linearDepthImages_[frameIndex] = images[5];
    specularHitDepthImages_[frameIndex] = images[6];
    firstHitDepthImages_[frameIndex] = images[7];
    firstHitDiffuseDirectLightImages_[frameIndex] = images[8];
    firstHitDiffuseIndirectLightImages_[frameIndex] = images[9];
    firstHitSpecularImages_[frameIndex] = images[10];
    firstHitClearImages_[frameIndex] = images[11];
    firstHitBaseEmissionImages_[frameIndex] = images[12];
    fogImages_[frameIndex] = images[13];
    firstHitRefractionImages_[frameIndex] = images[14];

    return true;
}

void RayTracingModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) { return value == "render_pipeline.true"; };
    auto parseDebugMode = [](const std::string &value) -> uint32_t {
        if (value == "off" || value == "0" || value == "render_pipeline.false") return 0;
        if (value == "hash_grid" || value == "1") return 1;
        if (value == "occupancy" || value == "2") return 2;
        if (value == "heatmap" || value == "3") return 3;
        try {
            return static_cast<uint32_t>(std::max(0, std::stoi(value)));
        } catch (...) { return 0; }
    };
    auto parsePbrSamplingMode = [](const std::string &value) -> uint32_t {
        if (value == "render_pipeline.module.ray_tracing.attribute.pbr_sampling.nearest") return 0;
        if (value == "render_pipeline.module.ray_tracing.attribute.pbr_sampling.bilinear") return 1;
        return 0; // nearest by default
    };
    auto parseTransparentSplitMode = [](const std::string &value) -> uint32_t {
        if (value == "render_pipeline.module.ray_tracing.attribute.transparent_split_mode.stochastic" ||
            value == "stochastic") {
            return RAY_TRACING_TRANSPARENT_SPLIT_MODE_STOCHASTIC;
        }
        if (value == "render_pipeline.module.ray_tracing.attribute.transparent_split_mode.deterministic" ||
            value == "deterministic") {
            return RAY_TRACING_TRANSPARENT_SPLIT_MODE_DETERMINISTIC;
        }
        return RAY_TRACING_TRANSPARENT_SPLIT_MODE_DETERMINISTIC;
    };
    auto parseFarFieldMaterialMode = [](const std::string &value) -> uint32_t {
        if (value == "render_pipeline.module.ray_tracing.attribute.far_field_material_mode.flat_surface" ||
            value == "flat_surface") {
            return RAY_TRACING_FAR_FIELD_MATERIAL_MODE_FLAT_SURFACE;
        }
        return RAY_TRACING_FAR_FIELD_MATERIAL_MODE_FULL_PBR;
    };
    auto parseFloat = [](const std::string &value, float fallback) -> float {
        try {
            return std::stof(value);
        } catch (...) { return fallback; }
    };
    auto parseVec3 = [](const std::string &value, const glm::vec3 &fallback) -> glm::vec3 {
        std::string normalized = value;
        for (char &ch : normalized) {
            if (ch == ',' || ch == ';' || ch == '(' || ch == ')' || ch == '[' || ch == ']') { ch = ' '; }
        }

        std::stringstream stream(normalized);
        glm::vec3 parsed(0.0f);
        if (!(stream >> parsed.x >> parsed.y >> parsed.z)) { return fallback; }
        return parsed;
    };
    auto clampNonNegativeVec3 = [](const glm::vec3 &value) {
        return glm::vec3(std::max(0.0f, value.x), std::max(0.0f, value.y), std::max(0.0f, value.z));
    };
    constexpr float basicRadianceScale = 1e-3f;
    constexpr float distanceScale = 1000.0f;
    auto buffers = Renderer::instance().buffers();

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.ray_tracing.attribute.num_ray_bounces") {
            numRayBounces_ = std::stoi(value);
#ifdef DEBUG
            std::cout << "Ray tracing num bounces: " << numRayBounces_ << std::endl;
#endif
        } else if (key == "render_pipeline.module.ray_tracing.attribute.use_jitter") {
            useJitter_ = parseBool(value);
            buffers->setUseJitter(useJitter_);
#ifdef DEBUG
            std::cout << "Ray tracing use jitter: " << (useJitter_ ? "True" : "False") << std::endl;
#endif
        } else if (key == "render_pipeline.module.ray_tracing.attribute.direct_light_strength") {
            directLightStrength_ = std::max(0.0f, parseFloat(value, directLightStrength_));
        } else if (key == "render_pipeline.module.ray_tracing.attribute.indirect_light_strength") {
            indirectLightStrength_ = std::max(0.0f, parseFloat(value, indirectLightStrength_));
        } else if (key == "render_pipeline.module.ray_tracing.attribute.basic_radiance") {
            basicRadiance_ =
                std::max(0.0f, parseFloat(value, basicRadiance_ / basicRadianceScale)) * basicRadianceScale;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.pbr_sampling_mode") {
            pbrSamplingMode_ = parsePbrSamplingMode(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.transparent_split_mode") {
            transparentSplitMode_ = parseTransparentSplitMode(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.far_field_start_distance_chunks") {
            farFieldStartDistanceChunks_ = std::max(0.0f, parseFloat(value, farFieldStartDistanceChunks_));
        } else if (key == "render_pipeline.module.ray_tracing.attribute.far_field_material_mode") {
            farFieldMaterialMode_ = parseFarFieldMaterialMode(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.atmosphere_planet_radius") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setPlanetRadius(std::max(1.0f, parseFloat(value, 6360.0f) * distanceScale));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.atmosphere_top_radius") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setAtmosphereTopRadius(std::max(1.0f, parseFloat(value, 6460.0f) * distanceScale));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.rayleigh_scale_height") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setRayleighScaleHeight(std::max(0.0f, parseFloat(value, 8.0f) * distanceScale));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.mie_scale_height") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setMieScaleHeight(std::max(0.0f, parseFloat(value, 1.2f) * distanceScale));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.rayleigh_scattering_coefficient") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setRayleighScatteringCoefficient(
                    clampNonNegativeVec3(parseVec3(value, glm::vec3(5.802e-6f, 13.558e-6f, 33.100e-6f))));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.mie_anisotropy") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setMieAnisotropy(std::clamp(parseFloat(value, 0.80f), -0.999f, 0.999f));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.mie_scattering_coefficient") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setMieScatteringCoefficient(
                    clampNonNegativeVec3(parseVec3(value, glm::vec3(21.000e-6f, 21.000e-6f, 21.000e-6f))));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.minimum_view_cosine") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setMinimumViewCosine(std::clamp(parseFloat(value, 0.02f), -1.0f, 1.0f));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.sun_radiance") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setSunRadiance(clampNonNegativeVec3(parseVec3(value, glm::vec3(16.0f))));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.moon_radiance") {
            if (atmosphere_ != nullptr) {
                atmosphere_->setMoonRadiance(clampNonNegativeVec3(parseVec3(value, glm::vec3(0.08f, 0.1f, 0.2f))));
            }
        } else if (key == "render_pipeline.module.ray_tracing.attribute.use_sharc") {
            useSharc_ = parseBool(value);
#ifdef DEBUG
            std::cout << "Ray tracing use sharc: " << (useSharc_ ? "True" : "False") << std::endl;
#endif
        } else if (key == "render_pipeline.module.ray_tracing.attribute.sharc_debug_mode") {
            sharcDebugMode_ = parseDebugMode(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.shader_pack_path") {
            shaderPackPath_ = value;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.sharc_scene_scale") {
            sharcSceneScale_ = std::max(0.001f, std::stof(value));
        } else if (key == "render_pipeline.module.ray_tracing.attribute.sharc_accumulation_frame_num") {
            sharcAccumulationFrameNum_ = std::max(1, std::stoi(value));
        } else if (key == "render_pipeline.module.ray_tracing.attribute.sharc_stale_frame_num_max") {
            sharcStaleFrameNumMax_ = std::max(8, std::stoi(value));
        } else if (key == "render_pipeline.module.ray_tracing.attribute.sharc_update_downsample_factor") {
            sharcUpdateDownsampleFactor_ = static_cast<uint32_t>(std::max(1, std::stoi(value)));
        }
    }
}

void RayTracingModule::build() {
    atmosphere_->build();
    worldPrepare_->build();

    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    initDescriptorTables();
    initSharc();
    initImages();
    initPipeline();
    initSBT();

    for (int i = 0; i < size; i++) {
        contexts_[i] =
            RayTracingModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());

        // set rayTracingModuleContext of sub-modules, order is important
        atmosphere_->contexts_[i]->rayTracingModuleContext =
            std::static_pointer_cast<RayTracingModuleContext>(contexts_[i]);
        worldPrepare_->contexts_[i]->rayTracingModuleContext =
            std::static_pointer_cast<RayTracingModuleContext>(contexts_[i]);
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &RayTracingModule::contexts() {
    return contexts_;
}

void RayTracingModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        if (rayTracingDescriptorTables_[i] != nullptr)
            rayTracingDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                             0, 0, index);
    }
}

void RayTracingModule::preClose() {
    contexts_.clear();
    worldPrepare_ = nullptr;
    atmosphere_ = nullptr;
}

uint32_t RayTracingModule::hitGroupIndexForName(const std::string &groupName) const {
    auto iter = hitGroupNameToIndex_.find(groupName);
    if (iter != hitGroupNameToIndex_.end()) { return iter->second; }
    return fallbackHitGroupIndex_;
}

uint32_t RayTracingModule::shadowHitGroupIndex() const {
    return shadowHitGroupIndex_;
}

uint32_t RayTracingModule::fallbackHitGroupIndex() const {
    return fallbackHitGroupIndex_;
}

void RayTracingModule::initDescriptorTables() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    rayTracingDescriptorTables_.resize(size);

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet() // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 4096, // a very big number
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // world atmosphere LUT
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // world atmosphere cube map
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 1
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: TLAS(s)
                    .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: blasOffsets
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: vertex buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: index buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: last vertex buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // binding 5: last index buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // binding 6: position buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 7, // binding 7: material buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 8, // binding 8: last position buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // binding 9: texture mapping
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: last obj to world mat
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 2
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: current world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: last world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: sky ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 3
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: hdrNoisyImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: diffuseAlbedoImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: specularAlbedoImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: normalRoughnessImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: motionVectorImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // binding 5: linearDepthImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // binding 6: specularHitDepth
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 7, // binding 7: firstHitDepthImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 8, // binding 8: firstHitDiffuseDirectLightImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // binding 9: firstHitDiffuseIndirectLightImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: firstHitSpecularImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 11, // binding 11: firstHitClearImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 12, // binding 12: firstHitBaseEmissionImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 13, // binding 13: fogImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 14, // binding 14: firstHitRefractionImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 4
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: SHaRC config
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: SHaRC hash entries
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: SHaRC accumulation
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: SHaRC resolved
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: SHaRC lock
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant({
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                    .offset = 0,
                    .size = sizeof(RayTracingPushConstant),
                })
                .build(framework->device());
    }
}

void RayTracingModule::initSharc() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t size = framework->swapchain()->imageCount();

    sharcConfigBuffers_.resize(size);

    const VkBufferUsageFlags sharcStorageUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    sharcHashEntriesBuffer_ =
        vk::DeviceLocalBuffer::create(vma, device, false, static_cast<size_t>(sharcCapacity) * sizeof(uint64_t),
                                      sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);
    sharcLockBuffer_ =
        vk::DeviceLocalBuffer::create(vma, device, false, static_cast<size_t>(sharcCapacity) * sizeof(uint32_t),
                                      sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);
    sharcAccumulationBuffer_ =
        vk::DeviceLocalBuffer::create(vma, device, false, static_cast<size_t>(sharcCapacity) * sizeof(uint32_t) * 4,
                                      sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);
    sharcResolvedBuffer_ =
        vk::DeviceLocalBuffer::create(vma, device, false, static_cast<size_t>(sharcCapacity) * sizeof(uint32_t) * 4,
                                      sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);

    auto clearCommandPool = vk::CommandPool::create(framework->physicalDevice(), device);
    auto clearCommandBuffer = vk::CommandBuffer::create(device, clearCommandPool);
    clearCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcHashEntriesBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcLockBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcAccumulationBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcResolvedBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    clearCommandBuffer->end();
    clearCommandBuffer->submitMainQueueIndividual(device);
    vkQueueWaitIdle(device->mainVkQueue());

    for (uint32_t i = 0; i < size; i++) {
        sharcConfigBuffers_[i] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(SharcConfigData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
}

void RayTracingModule::updateSharcConfig(uint32_t frameIndex) {
    auto buffers = Renderer::instance().buffers();
    auto worldUbo = reinterpret_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
    if (worldUbo == nullptr) { return; }

    glm::dvec3 currentCameraPos(worldUbo->cameraPos.x, worldUbo->cameraPos.y, worldUbo->cameraPos.z);
    auto lightingDirtyState = Renderer::instance().world()->chunks()->lightingDirtyState();
    if (sharcFirstFrame_) {
        sharcPrevCameraPos_ = currentCameraPos;
        sharcPrevLightRevision_ = lightingDirtyState.sceneLightRevision;
        sharcFirstFrame_ = false;
    }

    const bool dirtyRegionActive =
        lightingDirtyState.active && lightingDirtyState.centerRadius.w > 0.0f && lightingDirtyState.dirtyFramesRemaining > 0;
    const bool lightRevisionChanged = lightingDirtyState.sceneLightRevision != sharcPrevLightRevision_;
    uint32_t effectiveDownsampleFactor = std::max(1u, sharcUpdateDownsampleFactor_);
    uint32_t stableUpdateStride = 1;
    if (dirtyRegionActive || lightRevisionChanged) {
        effectiveDownsampleFactor = std::max(1u, (effectiveDownsampleFactor + 1u) / 2u);
    } else if (lightingDirtyState.framesSinceLastDirty >= 48u) {
        stableUpdateStride = 3u;
        effectiveDownsampleFactor = std::min(12u, effectiveDownsampleFactor * 2u + 1u);
    } else if (lightingDirtyState.framesSinceLastDirty >= 16u) {
        stableUpdateStride = 2u;
        effectiveDownsampleFactor = std::min(10u, effectiveDownsampleFactor * 2u);
    }
    sharcEffectiveUpdateDownsampleFactor_ = effectiveDownsampleFactor;
    sharcStableUpdateStride_ = stableUpdateStride;

    auto splitAddress = [](VkDeviceAddress address) {
        return std::array<uint32_t, 2>{static_cast<uint32_t>(address & 0xFFFFFFFFu),
                                       static_cast<uint32_t>((address >> 32) & 0xFFFFFFFFu)};
    };

    SharcConfigData config{};
    const auto hashEntriesAddress = splitAddress(sharcHashEntriesBuffer_->bufferAddress());
    const auto lockAddress = splitAddress(sharcLockBuffer_->bufferAddress());
    const auto accumulationAddress = splitAddress(sharcAccumulationBuffer_->bufferAddress());
    const auto resolvedAddress = splitAddress(sharcResolvedBuffer_->bufferAddress());
    config.hashEntriesAddress[0] = hashEntriesAddress[0];
    config.hashEntriesAddress[1] = hashEntriesAddress[1];
    config.lockAddress[0] = lockAddress[0];
    config.lockAddress[1] = lockAddress[1];
    config.accumulationAddress[0] = accumulationAddress[0];
    config.accumulationAddress[1] = accumulationAddress[1];
    config.resolvedAddress[0] = resolvedAddress[0];
    config.resolvedAddress[1] = resolvedAddress[1];
    config.cameraPosition = glm::vec4(static_cast<float>(currentCameraPos.x), static_cast<float>(currentCameraPos.y),
                                      static_cast<float>(currentCameraPos.z), 0.0f);
    config.cameraPositionPrev =
        glm::vec4(static_cast<float>(sharcPrevCameraPos_.x), static_cast<float>(sharcPrevCameraPos_.y),
                  static_cast<float>(sharcPrevCameraPos_.z), 0.0f);
    config.sceneScale = sharcSceneScale_;
    config.radianceScale = 1000.0f;
    config.accumulationFrameNum = sharcAccumulationFrameNum_;
    config.staleFrameNumMax = sharcStaleFrameNumMax_;
    config.capacity = sharcCapacity;
    config.frameIndex = sharcFrameIndex_;
    config.enableAntiFireflyFilter = 1;
    config.useLockBuffer = 0;
    config.debugMode = sharcDebugMode_;
    config.updateDownsampleFactor = sharcEffectiveUpdateDownsampleFactor_;
    config.dirtyRegionCenterRadius =
        dirtyRegionActive ? lightingDirtyState.centerRadius : glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
    config.sceneState = glm::uvec4(0u);
    config.sceneState.x = static_cast<glm::uint>(lightingDirtyState.sceneLightRevision);
    config.sceneState.y = static_cast<glm::uint>(sharcPrevLightRevision_);
    config.sceneState.z = static_cast<glm::uint>(sharcStableUpdateStride_);
    config.sceneState.w = static_cast<glm::uint>((dirtyRegionActive ? 0x1u : 0u) |
                                                 (lightRevisionChanged ? 0x2u : 0u));

    sharcConfigBuffers_[frameIndex]->uploadToBuffer(&config);

    sharcPrevCameraPos_ = currentCameraPos;
    sharcPrevLightRevision_ = lightingDirtyState.sceneLightRevision;
    sharcFrameIndex_++;
}

void RayTracingModule::initImages() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmLUTImageSampler_,
                                                                  atmosphere_->atmLUTImage_, 0, 1);
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmCubeMapImageSamplers_[i],
                                                                  atmosphere_->atmCubeMapImages_[i], 0, 2, 7);

        rayTracingDescriptorTables_[i]->bindImage(hdrNoisyOutputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 0);
        rayTracingDescriptorTables_[i]->bindImage(diffuseAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 1);
        rayTracingDescriptorTables_[i]->bindImage(specularAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 2);
        rayTracingDescriptorTables_[i]->bindImage(normalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 3);
        rayTracingDescriptorTables_[i]->bindImage(motionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 4);
        rayTracingDescriptorTables_[i]->bindImage(linearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 5);
        rayTracingDescriptorTables_[i]->bindImage(specularHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 6);
        rayTracingDescriptorTables_[i]->bindImage(firstHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 7);
        rayTracingDescriptorTables_[i]->bindImage(firstHitDiffuseDirectLightImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 8);
        rayTracingDescriptorTables_[i]->bindImage(firstHitDiffuseIndirectLightImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                  9);
        rayTracingDescriptorTables_[i]->bindImage(firstHitSpecularImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 10);
        rayTracingDescriptorTables_[i]->bindImage(firstHitClearImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 11);
        rayTracingDescriptorTables_[i]->bindImage(firstHitBaseEmissionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 12);
        rayTracingDescriptorTables_[i]->bindImage(fogImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 13);
        rayTracingDescriptorTables_[i]->bindImage(firstHitRefractionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 14);

        rayTracingDescriptorTables_[i]->bindBuffer(sharcConfigBuffers_[i], 4, 0);
        rayTracingDescriptorTables_[i]->bindBuffer(sharcHashEntriesBuffer_, 4, 1);
        rayTracingDescriptorTables_[i]->bindBuffer(sharcAccumulationBuffer_, 4, 2);
        rayTracingDescriptorTables_[i]->bindBuffer(sharcResolvedBuffer_, 4, 3);
        rayTracingDescriptorTables_[i]->bindBuffer(sharcLockBuffer_, 4, 4);
    }
}

void RayTracingModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();

    std::filesystem::path builtInShaderPackZipPath = Renderer::folderPath / "shaders/world/ray_tracing/internal.zip";
    auto extractShaderPack = [](const std::filesystem::path &zipPath, const std::filesystem::path &destinationPath) {
        std::error_code ec;
        std::filesystem::remove_all(destinationPath, ec);
        std::filesystem::create_directories(destinationPath, ec);
        if (ec) {
            std::cerr << "[Ray Tracing] Failed to prepare shader temp dir: " << destinationPath << std::endl;
            return false;
        }

        void *reader = mz_zip_reader_create();
        if (reader == nullptr) {
            std::cerr << "[Ray Tracing] Failed to create zip reader for " << zipPath << std::endl;
            return false;
        }

        int32_t openResult = mz_zip_reader_open_file(reader, zipPath.string().c_str());
        if (openResult != MZ_OK) {
            mz_zip_reader_delete(&reader);
            std::cerr << "[Ray Tracing] Failed to open zip file: " << zipPath << ", error: " << openResult << std::endl;
            return false;
        }

        int32_t extractResult = mz_zip_reader_save_all(reader, destinationPath.string().c_str());
        int32_t closeResult = mz_zip_reader_close(reader);
        mz_zip_reader_delete(&reader);

        if (extractResult != MZ_OK) {
            std::cerr << "[Ray Tracing] Failed to extract zip file: " << zipPath << ", error: " << extractResult
                      << std::endl;
            return false;
        }
        if (closeResult != MZ_OK) {
            std::cerr << "[Ray Tracing] Failed to close zip file: " << zipPath << ", error: " << closeResult
                      << std::endl;
            return false;
        }
        return true;
    };

    auto fileExtension = [](const std::filesystem::path &path) {
        std::string extension = path.extension().string();
        if (!extension.empty() && extension.front() == '.') { extension.erase(extension.begin()); }
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return extension;
    };

    std::filesystem::path rayGenShaderPath;
    std::string defaultHitGroupName;
    std::vector<ParsedHitGroupConfig> parsedHitGroups;
    std::vector<std::string> includeDirectories;
    std::unordered_map<std::string, std::shared_ptr<vk::Shader>> shaderCache;
    std::unordered_map<std::string, std::filesystem::path> rootShaderFileByName;
    std::unordered_map<std::string, HitShaderPaths> classifiedHitShaderGroups;
    std::filesystem::path activeShaderPackExtractPath;
    std::string shaderPackLoadError;
    auto loadRuntimeShader = [&](const std::filesystem::path &path, VkShaderStageFlagBits stage,
                                 std::unordered_map<std::string, std::string> definitions =
                                     std::unordered_map<std::string, std::string>{}) {
        std::string cacheKey = path.string() + "#" + std::to_string(static_cast<uint32_t>(stage));
        if (definitions.empty()) {
            auto cacheIter = shaderCache.find(cacheKey);
            if (cacheIter != shaderCache.end()) { return cacheIter->second; }
        }
        auto shader = vk::Shader::create(device, path.string(), stage, definitions, includeDirectories);
        if (definitions.empty()) { shaderCache[cacheKey] = shader; }
        return shader;
    };

    auto loadShaderPack = [&](const std::filesystem::path &shaderPackZipPath) {
        shaderPackLoadError.clear();
        activeShaderPackExtractPath.clear();

        const std::filesystem::path shaderExtractRoot = Renderer::folderPath / "temp/shaders/world/ray_tracing";
        std::filesystem::path shaderExtractPath = shaderExtractRoot / shaderPackZipPath.stem();

        if (!extractShaderPack(shaderPackZipPath, shaderExtractPath)) {
            shaderPackLoadError = "failed to read or extract shader pack";
            return false;
        }

        includeDirectories = {shaderExtractPath.string()};
        activeShaderPackExtractPath = shaderExtractPath;

        shaderCache.clear();
        rootShaderFileByName.clear();
        classifiedHitShaderGroups.clear();
        parsedHitGroups.clear();
        missShaders_.clear();
        rayGenShaderPath.clear();
        defaultHitGroupName.clear();

        try {
            std::vector<std::filesystem::path> rootShaderFiles;
            for (const auto &entry : std::filesystem::directory_iterator(shaderExtractPath)) {
                if (entry.is_regular_file()) { rootShaderFiles.push_back(entry.path()); }
            }
            std::sort(rootShaderFiles.begin(), rootShaderFiles.end(),
                      [](const std::filesystem::path &lhs, const std::filesystem::path &rhs) {
                          return lhs.filename().string() < rhs.filename().string();
                      });

            for (const auto &shaderPath : rootShaderFiles) {
                const std::string fileName = shaderPath.filename().string();
                rootShaderFileByName[fileName] = shaderPath;

                const std::string extension = fileExtension(shaderPath);
                if (extension == "rahit" || extension == "rchit" || extension == "rint") {
                    auto &group = classifiedHitShaderGroups[shaderPath.stem().string()];
                    if (extension == "rahit") {
                        if (!group.anyHit.has_value()) { group.anyHit = shaderPath; }
                    } else if (extension == "rchit") {
                        if (!group.closestHit.has_value()) { group.closestHit = shaderPath; }
                    } else {
                        if (!group.intersection.has_value()) { group.intersection = shaderPath; }
                    }
                }
            }

            auto resolveRootShaderPath = [&](const std::string &reference, const std::string &usageLabel) {
                if (reference.empty()) { throw std::runtime_error(usageLabel + " cannot be empty"); }

                std::filesystem::path relative(reference);
                if (relative.has_parent_path()) {
                    throw std::runtime_error(usageLabel + " must reference a root shader file: " + reference);
                }

                const std::string fileName = relative.filename().string();
                if (auto iter = rootShaderFileByName.find(fileName); iter != rootShaderFileByName.end()) {
                    return iter->second;
                }
                throw std::runtime_error(usageLabel + " references missing file: " + reference);
            };

            auto findClassifiedHitGroup = [&](const std::string &groupName) -> const HitShaderPaths * {
                if (auto iter = classifiedHitShaderGroups.find(groupName); iter != classifiedHitShaderGroups.end()) {
                    return &iter->second;
                }
                return nullptr;
            };

            const std::filesystem::path shaderConfigPath = shaderExtractPath / "configs.json";
            if (!std::filesystem::exists(shaderConfigPath)) {
                throw std::runtime_error("missing required shader config: " + shaderConfigPath.string());
            }

            nlohmann::json shaderConfig;
            std::ifstream shaderConfigStream(shaderConfigPath);
            if (!shaderConfigStream.is_open()) {
                throw std::runtime_error("failed to open shader config: " + shaderConfigPath.string());
            }
            shaderConfigStream >> shaderConfig;
            if (!shaderConfig.is_object()) { throw std::runtime_error("shader config root must be an object"); }

            auto requiredStringField = [&](const nlohmann::json &jsonValue, const std::string &key,
                                           const std::string &context) {
                auto iter = jsonValue.find(key);
                if (iter == jsonValue.end() || !iter->is_string()) {
                    throw std::runtime_error(context + "." + key + " must be a string");
                }
                return iter->get<std::string>();
            };

            auto optionalStringField = [&](const nlohmann::json &jsonValue, const std::string &key,
                                           const std::string &context) {
                auto iter = jsonValue.find(key);
                if (iter == jsonValue.end()) { return std::optional<std::string>{}; }
                if (!iter->is_string()) { throw std::runtime_error(context + "." + key + " must be a string"); }
                return std::optional<std::string>(iter->get<std::string>());
            };

            auto requiredObjectField = [&](const nlohmann::json &jsonValue, const std::string &key,
                                           const std::string &context) -> const nlohmann::json & {
                auto iter = jsonValue.find(key);
                if (iter == jsonValue.end() || !iter->is_object()) {
                    throw std::runtime_error(context + "." + key + " must be an object");
                }
                return *iter;
            };

            auto requiredArrayField = [&](const nlohmann::json &jsonValue, const std::string &key,
                                          const std::string &context) -> const nlohmann::json & {
                auto iter = jsonValue.find(key);
                if (iter == jsonValue.end() || !iter->is_array()) {
                    throw std::runtime_error(context + "." + key + " must be an array");
                }
                return *iter;
            };

            auto sharcIter = shaderConfig.find("sharc_compatible");
            if (sharcIter == shaderConfig.end() || !sharcIter->is_boolean()) {
                throw std::runtime_error("root.sharc_compatible must be a boolean");
            }
            sharcCompatible_ = sharcIter->get<bool>();
            useSharcRuntime_ = useSharc_ && sharcCompatible_;
            if (useSharc_ && !sharcCompatible_) {
#ifdef DEBUG
                std::cout << "SHaRC requested but disabled by shader config " << shaderConfigPath.filename()
                          << std::endl;
#endif
            }

            const std::string rayGenShaderRef = requiredStringField(shaderConfig, "rgen", "root");
            rayGenShaderPath = resolveRootShaderPath(rayGenShaderRef, "root.rgen");

            const nlohmann::json &hitGroupsJson = requiredObjectField(shaderConfig, "hit_groups", "root");
            if (hitGroupsJson.empty()) { throw std::runtime_error("root.hit_groups must not be empty"); }

            defaultHitGroupName = requiredStringField(shaderConfig, "default_hit_group", "root");
            if (!hitGroupsJson.contains(defaultHitGroupName)) {
                throw std::runtime_error("default hit group is missing in root.hit_groups: " + defaultHitGroupName);
            }

            std::vector<std::string> hitGroupNames;
            hitGroupNames.reserve(hitGroupsJson.size());
            hitGroupNames.push_back(defaultHitGroupName);
            for (auto iter = hitGroupsJson.begin(); iter != hitGroupsJson.end(); ++iter) {
                if (iter.key() != defaultHitGroupName) { hitGroupNames.push_back(iter.key()); }
            }
            std::sort(hitGroupNames.begin() + 1, hitGroupNames.end());

            auto parseGroupType = [&](const nlohmann::json &groupJson, const std::string &groupName) {
                const std::string typeValue = requiredStringField(groupJson, "type", "root.hit_groups." + groupName);
                if (typeValue == "triangle") { return VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; }
                if (typeValue == "aabb") { return VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR; }
                throw std::runtime_error("unsupported hit group type for " + groupName + ": " + typeValue);
            };

            auto parseHitGroup = [&](const std::string &groupName,
                                     const std::optional<std::filesystem::path> &fallbackClosestHit) {
                auto groupIter = hitGroupsJson.find(groupName);
                if (groupIter == hitGroupsJson.end() || !groupIter->is_object()) {
                    throw std::runtime_error("root.hit_groups." + groupName + " must be an object");
                }
                const nlohmann::json &groupJson = *groupIter;
                const nlohmann::json &shadersJson =
                    requiredObjectField(groupJson, "shaders", "root.hit_groups." + groupName);
                const HitShaderPaths *classifiedGroup = findClassifiedHitGroup(groupName);

                ParsedHitGroupConfig parsedGroup;
                parsedGroup.name = groupName;
                if (auto closestRef =
                        optionalStringField(shadersJson, "rchit", "root.hit_groups." + groupName + ".shaders");
                    closestRef.has_value()) {
                    parsedGroup.closestHit =
                        resolveRootShaderPath(*closestRef, "root.hit_groups." + groupName + ".shaders.rchit");
                } else if (classifiedGroup != nullptr && classifiedGroup->closestHit.has_value()) {
                    parsedGroup.closestHit = *classifiedGroup->closestHit;
                }

                if (auto anyRef =
                        optionalStringField(shadersJson, "rahit", "root.hit_groups." + groupName + ".shaders");
                    anyRef.has_value()) {
                    parsedGroup.anyHit =
                        resolveRootShaderPath(*anyRef, "root.hit_groups." + groupName + ".shaders.rahit");
                } else if (classifiedGroup != nullptr && classifiedGroup->anyHit.has_value()) {
                    parsedGroup.anyHit = *classifiedGroup->anyHit;
                }

                if (auto intersectionRef =
                        optionalStringField(shadersJson, "rint", "root.hit_groups." + groupName + ".shaders");
                    intersectionRef.has_value()) {
                    parsedGroup.intersection =
                        resolveRootShaderPath(*intersectionRef, "root.hit_groups." + groupName + ".shaders.rint");
                } else if (classifiedGroup != nullptr && classifiedGroup->intersection.has_value()) {
                    parsedGroup.intersection = *classifiedGroup->intersection;
                }

                if (!parsedGroup.closestHit.has_value() && !parsedGroup.intersection.has_value() &&
                    fallbackClosestHit.has_value()) {
                    parsedGroup.closestHit = fallbackClosestHit;
                }

                VkRayTracingShaderGroupTypeKHR requestedType = parseGroupType(groupJson, groupName);
                if (parsedGroup.intersection.has_value()) {
                    parsedGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
                } else if (requestedType == VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR) {
                    throw std::runtime_error("aabb hit group requires rint shader: " + groupName);
                } else {
                    parsedGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                }

                if (!parsedGroup.closestHit.has_value() && !parsedGroup.intersection.has_value()) {
                    throw std::runtime_error("hit group must provide rchit or rint: " + groupName);
                }
                return parsedGroup;
            };

            ParsedHitGroupConfig defaultHitGroup = parseHitGroup(defaultHitGroupName, std::nullopt);
            parsedHitGroups.push_back(defaultHitGroup);
            for (size_t i = 1; i < hitGroupNames.size(); i++) {
                parsedHitGroups.push_back(parseHitGroup(hitGroupNames[i], defaultHitGroup.closestHit));
            }

            const nlohmann::json &missShadersJson = requiredArrayField(shaderConfig, "miss", "root");
            if (missShadersJson.empty()) { throw std::runtime_error("root.miss must not be empty"); }

            missShaders_.resize(missShadersJson.size());
            for (size_t i = 0; i < missShadersJson.size(); i++) {
                const nlohmann::json &missJson = missShadersJson[i];
                if (!missJson.is_object()) {
                    throw std::runtime_error("root.miss[" + std::to_string(i) + "] must be an object");
                }

                const std::string missShaderRef =
                    requiredStringField(missJson, "shader", "root.miss[" + std::to_string(i) + "]");
                missShaders_[i] = {
                    .name = requiredStringField(missJson, "name", "root.miss[" + std::to_string(i) + "]"),
                    .index = static_cast<uint32_t>(i),
                    .shader = loadRuntimeShader(
                        resolveRootShaderPath(missShaderRef, "root.miss[" + std::to_string(i) + "].shader"),
                        VK_SHADER_STAGE_MISS_BIT_KHR),
                };
            }
        } catch (const std::exception &e) {
            shaderPackLoadError = e.what();
            return false;
        }

        return true;
    };

    std::filesystem::path shaderPackZipPath =
        shaderPackPath_.empty() ? builtInShaderPackZipPath : std::filesystem::path(shaderPackPath_);
    bool loadedCustomShaderPack = !shaderPackPath_.empty();

    bool shaderPackLoaded = loadShaderPack(shaderPackZipPath);
    if (!shaderPackLoaded && loadedCustomShaderPack) {
        std::cerr << "[Ray Tracing] Failed to load custom shader pack: " << shaderPackZipPath
                  << ". Falling back to built-in shader pack. Reason: " << shaderPackLoadError << std::endl;
        shaderPackZipPath = builtInShaderPackZipPath;
        shaderPackLoaded = loadShaderPack(shaderPackZipPath);
    }

    if (!shaderPackLoaded) {
        std::cerr << "[Ray Tracing] Failed to initialize ray tracing shaders from shader pack: " << shaderPackZipPath
                  << ". Reason: " << shaderPackLoadError << std::endl;
        exit(EXIT_FAILURE);
    }

    auto addGroupMapping = [&](const std::string &groupName, uint32_t index) {
        hitGroupNameToIndex_[groupName] = index;
    };

    hitShaderGroups_.clear();
    hitGroupNameToIndex_.clear();
    bool fallbackHitGroupFound = false;
    bool shadowHitGroupFound = false;

    for (const auto &parsedGroup : parsedHitGroups) {
        HitShaderGroupDefinition hitGroup;
        hitGroup.name = parsedGroup.name;
        hitGroup.type = parsedGroup.type;
        if (parsedGroup.closestHit.has_value()) {
            hitGroup.closestHitShader = loadRuntimeShader(*parsedGroup.closestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        }
        if (parsedGroup.anyHit.has_value()) {
            hitGroup.anyHitShader = loadRuntimeShader(*parsedGroup.anyHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
        }
        if (parsedGroup.intersection.has_value()) {
            hitGroup.intersectionShader =
                loadRuntimeShader(*parsedGroup.intersection, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
        }

        uint32_t groupIndex = static_cast<uint32_t>(hitShaderGroups_.size());
        hitShaderGroups_.push_back(hitGroup);
        addGroupMapping(parsedGroup.name, groupIndex);

        if (!fallbackHitGroupFound && parsedGroup.name == defaultHitGroupName) {
            fallbackHitGroupIndex_ = groupIndex;
            fallbackHitGroupFound = true;
        }
        if (!shadowHitGroupFound && parsedGroup.name == "shadow") {
            shadowHitGroupIndex_ = groupIndex;
            shadowHitGroupFound = true;
        }
    }

    if (!fallbackHitGroupFound) {
        std::cerr << "[Ray Tracing] Failed to resolve fallback hit group index for " << defaultHitGroupName
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!shadowHitGroupFound) { shadowHitGroupIndex_ = fallbackHitGroupIndex_; }

    for (const auto &[groupName, _] : classifiedHitShaderGroups) {
        if (hitGroupNameToIndex_.find(groupName) == hitGroupNameToIndex_.end()) {
            addGroupMapping(groupName, fallbackHitGroupIndex_);
        }
    }
    hitGroupCount_ = static_cast<uint32_t>(hitShaderGroups_.size());

    missGroupCount_ = static_cast<uint32_t>(missShaders_.size());

    std::unordered_map<std::string, std::string> queryDefinitions;
    std::unordered_map<std::string, std::string> updateDefinitions;
    if (useSharcRuntime_) {
        queryDefinitions = {{"SHARC_QUERY", "1"}, {"USE_SHARC", "1"}};
        updateDefinitions = {{"SHARC_UPDATE", "1"}, {"USE_SHARC", "1"}};
    }

    worldRayGenQueryShader_ = loadRuntimeShader(rayGenShaderPath, VK_SHADER_STAGE_RAYGEN_BIT_KHR, queryDefinitions);
    worldRayGenUpdateShader_ =
        useSharcRuntime_ ? loadRuntimeShader(rayGenShaderPath, VK_SHADER_STAGE_RAYGEN_BIT_KHR, updateDefinitions) :
                           worldRayGenQueryShader_;

    auto buildPipeline = [&](const std::shared_ptr<vk::Shader> &rayGenShader) {
        vk::RayTracingPipelineBuilder builder;
        auto &stageBuilder = builder.beginShaderStage();

        uint32_t stageIndex = 0;
        stageBuilder.defineShaderStage(rayGenShader, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
        stageIndex++;

        std::vector<uint32_t> missGeneralStageIndices;
        missGeneralStageIndices.reserve(missShaders_.size());
        for (const auto &missShader : missShaders_) {
            missGeneralStageIndices.push_back(stageIndex);
            stageBuilder.defineShaderStage(missShader.shader, VK_SHADER_STAGE_MISS_BIT_KHR);
            stageIndex++;
        }

        struct HitStageIndices {
            VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            uint32_t closestHit = VK_SHADER_UNUSED_KHR;
            uint32_t anyHit = VK_SHADER_UNUSED_KHR;
            uint32_t intersection = VK_SHADER_UNUSED_KHR;
        };
        std::vector<HitStageIndices> hitStageIndices;
        hitStageIndices.reserve(hitShaderGroups_.size());

        for (const auto &hitGroup : hitShaderGroups_) {
            HitStageIndices stageIndices;
            stageIndices.type = hitGroup.type;
            if (hitGroup.closestHitShader) {
                stageIndices.closestHit = stageIndex;
                stageBuilder.defineShaderStage(hitGroup.closestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
                stageIndex++;
            }
            if (hitGroup.anyHitShader) {
                stageIndices.anyHit = stageIndex;
                stageBuilder.defineShaderStage(hitGroup.anyHitShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
                stageIndex++;
            }
            if (hitGroup.intersectionShader) {
                stageIndices.intersection = stageIndex;
                stageBuilder.defineShaderStage(hitGroup.intersectionShader, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
                stageIndex++;
            }
            hitStageIndices.push_back(stageIndices);
        }
        stageBuilder.endShaderStage();

        auto &groupBuilder = builder.beginShaderGroup();
        groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR,
                                       VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
        for (uint32_t missGeneralStageIndex : missGeneralStageIndices) {
            groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, missGeneralStageIndex,
                                           VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
        }
        for (const auto &hitStage : hitStageIndices) {
            groupBuilder.defineShaderGroup(hitStage.type, VK_SHADER_UNUSED_KHR, hitStage.closestHit, hitStage.anyHit,
                                           hitStage.intersection);
        }
        groupBuilder.endShaderGroup();

        return builder.definePipelineLayout(rayTracingDescriptorTables_[0]).build(device);
    };

    rayTracingUpdatePipeline_ = buildPipeline(worldRayGenUpdateShader_);
    rayTracingQueryPipeline_ = buildPipeline(worldRayGenQueryShader_);

    if (useSharcRuntime_) {
        sharcResolveCompShader_ = loadRuntimeShader(activeShaderPackExtractPath / "sharc_resolve.comp",
                                                    VK_SHADER_STAGE_COMPUTE_BIT, {{"USE_SHARC", "1"}});
        sharcResolvePipeline_ = vk::ComputePipelineBuilder{}
                                    .defineShader(sharcResolveCompShader_)
                                    .definePipelineLayout(rayTracingDescriptorTables_[0])
                                    .build(device);
    } else {
        sharcResolveCompShader_ = nullptr;
        sharcResolvePipeline_ = nullptr;
    }
}

void RayTracingModule::initSBT() {
    auto framework = framework_.lock();

    sharcUpdateSbts_.resize(framework->swapchain()->imageCount());
    sharcQuerySbts_.resize(framework->swapchain()->imageCount());
    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        sharcUpdateSbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                              rayTracingUpdatePipeline_, missGroupCount_, hitGroupCount_);
        sharcQuerySbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                             rayTracingQueryPipeline_, missGroupCount_, hitGroupCount_);
    }
}

RayTracingModuleContext::RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<RayTracingModule> rayTracingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      rayTracingModule(rayTracingModule),
      rayTracingDescriptorTable(rayTracingModule->rayTracingDescriptorTables_[frameworkContext->frameIndex]),
      sharcUpdateSbt(rayTracingModule->sharcUpdateSbts_[frameworkContext->frameIndex]),
      sharcQuerySbt(rayTracingModule->sharcQuerySbts_[frameworkContext->frameIndex]),
      hdrNoisyOutputImage(rayTracingModule->hdrNoisyOutputImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(rayTracingModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(rayTracingModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(rayTracingModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(rayTracingModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(rayTracingModule->linearDepthImages_[frameworkContext->frameIndex]),
      specularHitDepthImage(rayTracingModule->specularHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(rayTracingModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDiffuseDirectLightImage(
          rayTracingModule->firstHitDiffuseDirectLightImages_[frameworkContext->frameIndex]),
      firstHitDiffuseIndirectLightImage(
          rayTracingModule->firstHitDiffuseIndirectLightImages_[frameworkContext->frameIndex]),
      firstHitSpecularImage(rayTracingModule->firstHitSpecularImages_[frameworkContext->frameIndex]),
      firstHitClearImage(rayTracingModule->firstHitClearImages_[frameworkContext->frameIndex]),
      firstHitBaseEmissionImage(rayTracingModule->firstHitBaseEmissionImages_[frameworkContext->frameIndex]),
      fogImage(rayTracingModule->fogImages_[frameworkContext->frameIndex]),
      firstHitRefractionImage(rayTracingModule->firstHitRefractionImages_[frameworkContext->frameIndex]),
      atmosphereContext(rayTracingModule->atmosphere_->contexts_[frameworkContext->frameIndex]),
      worldPrepareContext(rayTracingModule->worldPrepare_->contexts_[frameworkContext->frameIndex]) {}

void RayTracingModuleContext::render() {
    atmosphereContext->render();
    worldPrepareContext->render();

    if (worldPrepareContext->tlas == nullptr) {
#ifdef DEUBG
        std::cout << "tlas is nullptr" << std::endl;
#endif
        return;
    }

    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = rayTracingModule.lock();

    rayTracingDescriptorTable->bindAS(worldPrepareContext->tlas, 1, 0);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->blasOffsetsBuffer, 1, 1);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->vertexBufferAddr, 1, 2);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->indexBufferAddr, 1, 3);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastVertexBufferAddr, 1, 4);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastIndexBufferAddr, 1, 5);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->positionBufferAddr, 1, 6);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->materialBufferAddr, 1, 7);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastPositionBufferAddr, 1, 8);

    auto buffers = Renderer::instance().buffers();
    auto worldBuffer = buffers->worldUniformBuffer();

    rayTracingDescriptorTable->bindBuffer(buffers->textureMappingBuffer(), 1, 9);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastObjToWorldMat, 1, 10);
    rayTracingDescriptorTable->bindBuffer(worldBuffer, 2, 0);
    rayTracingDescriptorTable->bindBuffer(buffers->lastWorldUniformBuffer(), 2, 1);
    rayTracingDescriptorTable->bindBuffer(buffers->skyUniformBuffer(), 2, 2);
    if (module->useSharcRuntime_) {
        module->updateSharcConfig(context->frameIndex);
        rayTracingDescriptorTable->bindBuffer(module->sharcConfigBuffers_[context->frameIndex], 4, 0);
    }

    RayTracingPushConstant pc{
        .numRayBounces = module->numRayBounces_,
        .directLightStrength = module->directLightStrength_,
        .indirectLightStrength = module->indirectLightStrength_,
        .basicRadiance = module->basicRadiance_,
        .pbrSamplingMode = module->pbrSamplingMode_,
        .transparentSplitMode = module->transparentSplitMode_,
        .farFieldStartDistanceChunks = module->farFieldStartDistanceChunks_,
        .farFieldMaterialMode = module->farFieldMaterialMode_,
    };
    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), rayTracingDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(RayTracingPushConstant), &pc);

    auto chooseSrc = [](VkImageLayout oldLayout, VkPipelineStageFlags2 fallbackStage, VkAccessFlags2 fallbackAccess,
                        VkPipelineStageFlags2 &outStage, VkAccessFlags2 &outAccess) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            outStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            outAccess = 0;
        } else {
            outStage = fallbackStage;
            outAccess = fallbackAccess;
        }
    };

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
    auto addBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &img, VkImageLayout newLayout) {
        if (!img) return;
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(img->imageLayout(), VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        barriers.push_back({
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = img->imageLayout(),
            .newLayout = newLayout,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = img,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        img->imageLayout() = newLayout;
    };

    addBarrier(hdrNoisyOutputImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(diffuseAlbedoImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularAlbedoImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(normalRoughnessImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(motionVectorImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(linearDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularHitDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDiffuseDirectLightImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDiffuseIndirectLightImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitSpecularImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitClearImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitBaseEmissionImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(fogImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitRefractionImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(atmosphereContext->atmCubeMapImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (!barriers.empty()) { worldCommandBuffer->barriersBufferImage({}, barriers); }

    if (module->useSharcRuntime_) {
        const uint32_t updateDownsampleFactor = std::max(1u, module->sharcEffectiveUpdateDownsampleFactor_);
        const uint32_t updateWidth =
            (hdrNoisyOutputImage->width() + updateDownsampleFactor - 1) / updateDownsampleFactor;
        const uint32_t updateHeight =
            (hdrNoisyOutputImage->height() + updateDownsampleFactor - 1) / updateDownsampleFactor;

        worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
            ->bindRTPipeline(module->rayTracingUpdatePipeline_)
            ->raytracing(sharcUpdateSbt, updateWidth, updateHeight, 1);

        worldCommandBuffer->barriersMemory({{
            .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        }});

        worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->sharcResolvePipeline_);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(),
                      (RayTracingModule::sharcCapacity + RayTracingModule::sharcResolveWorkgroupSize - 1) /
                          RayTracingModule::sharcResolveWorkgroupSize,
                      1, 1);

        worldCommandBuffer->barriersMemory({{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        }});
    }

    worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
        ->bindRTPipeline(module->rayTracingQueryPipeline_)
        ->raytracing(sharcQuerySbt, hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height(), 1);
}
