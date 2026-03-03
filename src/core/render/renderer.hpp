#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <filesystem>

class Textures;
class Framework;
class Buffers;
class World;

struct Options {
    uint32_t maxFps = 1e6;
    uint32_t inactivityFpsLimit = 1e6;
    bool vsync = true;
    uint32_t dlssMode = 1;
    uint32_t upscalerType = 1;
    uint32_t upscalerQuality = 0;
    uint32_t denoiserMode = 1;
    bool hdrOutput = false;
    bool dlssFrameGeneration = false;
    bool dlssFrameGenerationActive = false;
    uint32_t rayBounces = 4;
    bool ommEnabled = false; // Opacity Micro Maps (disabled by default until Phase 1 validated)
    uint32_t ommBakerLevel = 4; // OMM baker max subdivision level (1-8)
    bool simplifiedIndirect = false; // Skip detail textures on indirect bounces + simplify shadow AHS
    bool outputScale2x = false; // Render world at 2x display resolution, Lanczos downscale
    bool reflexEnabled = false;
    bool reflexBoost = false;
    bool vrrMode = false;
    uint32_t debugMode = 0;
    bool needRecreate = false;

    uint32_t chunkBuildingBatchSize = 2;
    uint32_t chunkBuildingTotalBatches = 4;
};

class Renderer : public Singleton<Renderer> {
    friend class Singleton<Renderer>;

  public:
    static std::filesystem::path folderPath;
    static Options options;

    ~Renderer();

    std::shared_ptr<Framework> framework();
    std::shared_ptr<Textures> textures();
    std::shared_ptr<Buffers> buffers();
    std::shared_ptr<World> world();

    void close();

  private:
    Renderer(GLFWwindow *window);

    std::shared_ptr<Framework> framework_;
    std::shared_ptr<Textures> textures_;
    std::shared_ptr<Buffers> buffers_;
    std::shared_ptr<World> world_;
};
