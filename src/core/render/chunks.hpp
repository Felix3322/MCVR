#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/world.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class Framework;
class Chunks;

struct ChunkBuildTask {
    int x, y, z;
    int64_t id;
    int geometryCount;
    int *geometryTypes;
    const char **geometryGroupNames;
    int *geometryTextures;
    int *vertexFormats;
    int *vertexCounts;
    vk::VertexFormat::PBRVertex **vertices;
    uint64_t lightStateHash;
    bool isImportant;
};

struct ChunkBuildData : public SharedObject<ChunkBuildData> {
    int64_t id;
    int x, y, z;
    int64_t version;
    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    uint64_t lightStateHash;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::string> geometryGroupNames;
    std::vector<std::vector<vk::VertexFormat::PBRVertex>> vertices;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> vertexBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indexBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> positionBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> materialBuffers;
    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<vk::BLASBuilder> blasBuilder;

    ChunkBuildData(int64_t id,
                   int x,
                   int y,
                   int z,
                   int64_t version,
                   uint32_t allVertexCount,
                   uint32_t allIndexCount,
                   uint32_t geometryCount,
                   uint64_t lightStateHash,
                   std::vector<World::GeometryTypes> &&geometryTypes,
                   std::vector<std::string> &&geometryGroupNames,
                   std::vector<std::vector<vk::VertexFormat::PBRVertex>> &&vertices,
                   std::vector<std::vector<uint32_t>> &&indices);

    void build();
};

struct Chunk1;

struct ChunkBuildDataBatch : public SharedObject<ChunkBuildDataBatch> {
    std::vector<std::shared_ptr<ChunkBuildData>> batchData;

    ChunkBuildDataBatch(uint32_t maxBatchSize,
                        std::set<int64_t> &queuedIndex,
                        std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        glm::vec3 cameraPos);
};

class ChunkBuildScheduler : public SharedObject<ChunkBuildScheduler> {
  public:
    ChunkBuildScheduler(std::set<int64_t> &queuedIndex,
                        std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        std::recursive_mutex &mutex,
                        std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData,
                        Chunks *owner,
                        uint32_t chunkBuildingBatchSize,
                        uint32_t chunkBuildingTotalBatches);

    void tryCheckBatchesFinish();
    void waitAllBatchesFinish();
    void tryScheduleBatches(uint32_t maxBatchSize);

    uint32_t chunkBuildingBatchSize();
    uint32_t chunkBuildingTotalBatches();

  private:
    std::set<int64_t> &queuedIndex_;
    std::vector<std::shared_ptr<Chunk1>> &chunks_;
    std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas_;
    std::recursive_mutex &mutex_;
    std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData_;
    Chunks *owner_;

    std::queue<std::shared_ptr<vk::Fence>> freeFences_;
    std::list<std::shared_ptr<vk::Fence>> buildingFences_;
    std::list<std::shared_ptr<ChunkBuildDataBatch>> buildingBatches_;

    uint32_t chunkBuildingBatchSize_;
    uint32_t chunkBuildingTotalBatches_;
};

struct ChunkRenderData : public SharedObject<ChunkRenderData> {
    int x, y, z;
    std::shared_ptr<vk::BLAS> blas;
    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    std::shared_ptr<std::vector<World::GeometryTypes>> geometryTypes;
    std::shared_ptr<std::vector<std::string>> geometryGroupNames;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> vertexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> indexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> positionBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> materialBuffers;
    std::shared_ptr<std::vector<std::vector<vk::VertexFormat::PBRVertex>>> vertices;
    std::shared_ptr<std::vector<std::vector<uint32_t>>> indices;
};

struct Chunk1 : public SharedObject<Chunk1> {
    constexpr static float T_HALF = 200; // ms
    constexpr static float T_WEIGHT = 1.0;

    constexpr static float D_HALF = 96; // blocks
    constexpr static float D_SENSITIVITY = 1.5;
    constexpr static float D_WEIGHT = 1.2;

    int x, y, z;
    int64_t latestVersion = 0;
    std::chrono::steady_clock::time_point lastUpdate;

    std::shared_ptr<vk::BLAS> blas;
    int64_t blasVersion = -1;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> vertexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> indexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> positionBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> materialBuffers;

    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    uint64_t lightStateHash = 0;
    bool hasLightStateHash = false;
    std::shared_ptr<std::vector<World::GeometryTypes>> geometryTypes;
    std::shared_ptr<std::vector<std::string>> geometryGroupNames;
    std::shared_ptr<std::vector<std::vector<vk::VertexFormat::PBRVertex>>> vertices;
    std::shared_ptr<std::vector<std::vector<uint32_t>>> indices;

    float buildFactor(std::chrono::steady_clock::time_point currentTime, glm::vec3 cameraPos);

    bool enqueue(std::shared_ptr<ChunkBuildData> chunkBuildData);
    void invalidate();
    std::shared_ptr<ChunkRenderData> tryGetValid();
};

struct ChunkPackedData {
    uint32_t geometryCount;
};

struct LightingDirtyState {
    bool active = false;
    glm::vec4 centerRadius = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
    uint32_t sceneLightRevision = 0;
    uint32_t framesSinceLastDirty = 0;
    uint32_t dirtyFramesRemaining = 0;
};

class Chunks : public SharedObject<Chunks> {
    friend World;

  public:
    Chunks(std::shared_ptr<Framework> framework);

    void reset(uint32_t numChunks);
    void resetScheduler();
    void resetFrame();
    void invalidateChunk(int id);
    void markLightSectionDirty(int sectionX, int sectionY, int sectionZ, int lightType);
    void queueChunkBuild(ChunkBuildTask task);

    bool isChunkReady(int64_t id);
    LightingDirtyState lightingDirtyState();

    void close();

    std::recursive_mutex &mutex();
    std::vector<std::shared_ptr<Chunk1>> &chunks();
    std::shared_ptr<ChunkBuildScheduler> chunkBuildScheduler();
    std::vector<std::shared_ptr<vk::BLASBuilder>> &importantBLASBuilders();
    std::shared_ptr<vk::HostVisibleBuffer> chunkPackedData();

  private:
    void noteLightingDirtySections(const glm::ivec3 &minSection, const glm::ivec3 &maxSection);

    std::recursive_mutex mutex_;
    std::vector<std::shared_ptr<Chunk1>> chunks_;
    std::shared_ptr<vk::HostVisibleBuffer> chunkPackedData_ = nullptr;
    std::vector<std::shared_ptr<ChunkBuildData>> chunkBuildDatas_;
    std::set<int64_t> queuedIndex_;
    std::shared_ptr<ChunkBuildScheduler> chunkBuildScheduler_;

    std::shared_ptr<std::vector<std::shared_ptr<vk::BLASBuilder>>> importantBLASBuilders_;
    bool hasLightingDirtySections_ = false;
    bool lightingDirtyQueuedThisFrame_ = false;
    glm::ivec3 lightingDirtyMinSection_ = glm::ivec3(0);
    glm::ivec3 lightingDirtyMaxSection_ = glm::ivec3(0);
    uint32_t lightingDirtyFramesRemaining_ = 0;
    uint32_t sceneLightRevision_ = 0;
    uint32_t framesSinceLastLightingDirty_ = 1024;
};
