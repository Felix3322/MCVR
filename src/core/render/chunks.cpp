#include "core/render/chunks.hpp"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/vulkan/vertex.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

ChunkBuildData::ChunkBuildData(int64_t id,
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
                               std::vector<std::vector<uint32_t>> &&indices)
    : id(id),
      x(x),
      y(y),
      z(z),
      version(version),
      allVertexCount(allVertexCount),
      allIndexCount(allIndexCount),
      geometryCount(geometryCount),
      lightStateHash(lightStateHash),
      geometryTypes(std::move(geometryTypes)),
      geometryGroupNames(std::move(geometryGroupNames)),
      vertices(std::move(vertices)),
      indices(std::move(indices)),
      blas(nullptr),
      blasBuilder(nullptr) {}

void ChunkBuildData::build() {
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    for (int i = 0; i < geometryCount; i++) {
        auto vertexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, vertices[i].size() * sizeof(vk::VertexFormat::PBRVertex),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vertexBuffer->uploadToStagingBuffer(vertices[i].data());
        vertexBuffers.push_back(vertexBuffer);

        auto indexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, indices[i].size() * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        indexBuffer->uploadToStagingBuffer(indices[i].data());
        indexBuffers.push_back(indexBuffer);

        auto positionVertices = vk::Vertex::buildPositionVertices(vertices[i]);
        auto positionBuffer = vk::DeviceLocalBuffer::create(
            vma, device, positionVertices.size() * sizeof(vk::VertexFormat::PositionVertex),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        positionBuffer->uploadToStagingBuffer(positionVertices.data());
        positionBuffers.push_back(positionBuffer);

        auto materialVertices = vk::Vertex::buildMaterialVertices(vertices[i]);
        auto materialBuffer = vk::DeviceLocalBuffer::create(
            vma, device, materialVertices.size() * sizeof(vk::VertexFormat::MaterialVertex),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        materialBuffer->uploadToStagingBuffer(materialVertices.data());
        materialBuffers.push_back(materialBuffer);
    }

    blasBuilder = vk::BLASBuilder::create();
    auto blasGeometryBuilder = blasBuilder->beginGeometries();
    for (int i = 0; i < geometryCount; i++) {
        blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRVertex>(
            vertexBuffers[i], vertices[i].size(), indexBuffers[i], indices[i].size(),
            geometryTypes[i] == World::WORLD_SOLID);
    }
    blasGeometryBuilder->endGeometries();
    blas = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
               ->querySizeInfo(device)
               ->allocateBuffers(physicalDevice, device, vma)
               ->build(device);
}

ChunkBuildDataBatch::ChunkBuildDataBatch(uint32_t maxBatchSize,
                                         std::set<int64_t> &queuedIndexSet,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         glm::vec3 cameraPos) {
    std::vector<int64_t> queuedIndices;
    std::copy(queuedIndexSet.begin(), queuedIndexSet.end(), std::back_inserter(queuedIndices));
    auto currentTime = std::chrono::steady_clock::now();
    std::sort(queuedIndices.begin(), queuedIndices.end(), [&](int64_t a, int64_t b) -> bool {
        return chunks[a]->buildFactor(currentTime, cameraPos) > chunks[b]->buildFactor(currentTime, cameraPos);
    });

    for (int i = 0; i < std::min((size_t)maxBatchSize, queuedIndices.size()); i++) {
        auto iter = queuedIndexSet.find(queuedIndices[i]);
        if (iter != queuedIndexSet.end()) { queuedIndexSet.erase(iter); }

        auto data = chunkBuildDatas[queuedIndices[i]];
        data->build();
        batchData.push_back(data);
    }
}

ChunkBuildScheduler::ChunkBuildScheduler(std::set<int64_t> &queuedIndex,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         std::recursive_mutex &mutex,
                                         std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData,
                                         Chunks *owner,
                                         uint32_t chunkBuildingBatchSize,
                                         uint32_t chunkBuildingTotalBatches)
    : queuedIndex_(queuedIndex),
      chunks_(chunks),
      chunkBuildDatas_(chunkBuildDatas),
      mutex_(mutex),
      chunkPackedData_(chunkPackedData),
      owner_(owner),
      chunkBuildingBatchSize_(chunkBuildingBatchSize),
      chunkBuildingTotalBatches_(chunkBuildingTotalBatches) {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    uint32_t numFences = chunkBuildingTotalBatches_;
    for (int i = 0; i < numFences; i++) { freeFences_.push(vk::Fence::create(device)); }
}

void ChunkBuildScheduler::tryCheckBatchesFinish() {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterBatch != buildingBatches_.end();) {
        if (vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, 0) == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);

            for (auto chunkBuildData : (*iterBatch)->batchData) {
                bool lightStateChanged = chunks_[chunkBuildData->id]->enqueue(chunkBuildData);
                if (lightStateChanged) {
                    int sectionX = chunkBuildData->x >> 4;
                    int sectionY = chunkBuildData->y >> 4;
                    int sectionZ = chunkBuildData->z >> 4;
                    if (owner_ != nullptr) { owner_->markLightSectionDirty(sectionX, sectionY, sectionZ, 0); }
                }

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterBatch = buildingBatches_.erase(iterBatch);
        }
    }
}

void ChunkBuildScheduler::waitAllBatchesFinish() {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterBatch != buildingBatches_.end();) {
        if (vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, UINT64_MAX) == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);

            for (auto chunkBuildData : (*iterBatch)->batchData) {
                bool lightStateChanged = chunks_[chunkBuildData->id]->enqueue(chunkBuildData);
                if (lightStateChanged) {
                    int sectionX = chunkBuildData->x >> 4;
                    int sectionY = chunkBuildData->y >> 4;
                    int sectionZ = chunkBuildData->z >> 4;
                    if (owner_ != nullptr) { owner_->markLightSectionDirty(sectionX, sectionY, sectionZ, 0); }
                }

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterBatch = buildingBatches_.erase(iterBatch);
        }
    }
}

void ChunkBuildScheduler::tryScheduleBatches(uint32_t maxBatchSize) {
    if (!Renderer::instance().framework()->isRunning()) return;
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (!freeFences_.empty() && !queuedIndex_.empty()) {
        auto fence = freeFences_.front();

        glm::vec3 cameraPos = Renderer::instance().world()->getCameraPos();
        auto chunkBuildDataBatch =
            ChunkBuildDataBatch::create(maxBatchSize, queuedIndex_, chunks_, chunkBuildDatas_, cameraPos);

        auto framework = Renderer::instance().framework();
        auto vma = framework->vma();
        auto device = framework->device();
        auto physicalDevice = Renderer::instance().framework()->physicalDevice();
        auto secondaryQueueIndex = physicalDevice->secondaryQueueIndex();

        auto worldAsyncBuffer = framework->worldAsyncCommandBuffer();

        if (chunkBuildDataBatch->batchData.size() > 0) {
            worldAsyncBuffer->begin();

            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                for (int i = 0; i < chunkBuildData->geometryCount; i++) {
                    chunkBuildData->vertexBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                    chunkBuildData->indexBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                    chunkBuildData->positionBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                    chunkBuildData->materialBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                }
            }

            std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                for (int i = 0; i < chunkBuildData->geometryCount; i++) {
                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->vertexBuffers[i],
                    });

                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->indexBuffers[i],
                    });

                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->positionBuffers[i],
                    });

                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->materialBuffers[i],
                    });
                }
                worldAsyncBuffer->barriersBufferImage(bufferBarriers, {});
            }

            std::vector<std::shared_ptr<vk::BLASBuilder>> builders;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                builders.push_back(chunkBuildData->blasBuilder);
            }
            vk::BLASBuilder::batchSubmit(builders, worldAsyncBuffer);

            worldAsyncBuffer->end();

            VkSubmitInfo vkSubmitInfo = {};
            vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            vkSubmitInfo.waitSemaphoreCount = 0;
            vkSubmitInfo.pWaitSemaphores = nullptr;
            vkSubmitInfo.pWaitDstStageMask = nullptr;
            vkSubmitInfo.commandBufferCount = 1;
            vkSubmitInfo.pCommandBuffers = &worldAsyncBuffer->vkCommandBuffer();
            vkSubmitInfo.signalSemaphoreCount = 0;
            vkSubmitInfo.pSignalSemaphores = nullptr;

            vkQueueSubmit(device->secondaryQueue(), 1, &vkSubmitInfo, fence->vkFence());

            freeFences_.pop();
            buildingFences_.push_back(fence);
            buildingBatches_.push_back(chunkBuildDataBatch);
        }
    }
}

uint32_t ChunkBuildScheduler::chunkBuildingBatchSize() {
    return chunkBuildingBatchSize_;
}

uint32_t ChunkBuildScheduler::chunkBuildingTotalBatches() {
    return chunkBuildingTotalBatches_;
}

float Chunk1::buildFactor(std::chrono::steady_clock::time_point currentTime, glm::vec3 cameraPos) {
    double tDiff = std::chrono::duration<double, std::milli>(currentTime - lastUpdate).count();
    double dDiff = glm::distance(cameraPos, glm::vec3{x, y, z});

    double tScore = 1 - exp(-tDiff / T_HALF);
    double dScore = 1 / (1 + pow(dDiff / D_HALF, D_SENSITIVITY));
    double score = pow(tScore, T_WEIGHT) * pow(dScore, D_WEIGHT);

    return score;
}

bool Chunk1::enqueue(std::shared_ptr<ChunkBuildData> chunkBuildData) {
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();
    bool lightStateChanged = !hasLightStateHash || lightStateHash != chunkBuildData->lightStateHash;

    lastUpdate = std::chrono::steady_clock::now();
    x = chunkBuildData->x;
    y = chunkBuildData->y;
    z = chunkBuildData->z;
    lightStateHash = chunkBuildData->lightStateHash;
    hasLightStateHash = true;

    if (chunkBuildData->version > blasVersion) {
        blasVersion = chunkBuildData->version;

        gc.collect(blas);
        blas = chunkBuildData->blas;

        gc.collect(vertexBuffers);
        vertexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers));

        gc.collect(indexBuffers);
        indexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers));

        gc.collect(positionBuffers);
        positionBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->positionBuffers));

        gc.collect(materialBuffers);
        materialBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->materialBuffers));
    } else {
        gc.collect(chunkBuildData->blas);

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers)));

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers)));

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->positionBuffers)));

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->materialBuffers)));
    }

    allVertexCount = chunkBuildData->allVertexCount;
    allIndexCount = chunkBuildData->allIndexCount;
    geometryCount = chunkBuildData->geometryCount;
    geometryTypes = std::make_shared<std::vector<World::GeometryTypes>>(std::move(chunkBuildData->geometryTypes));
    geometryGroupNames = std::make_shared<std::vector<std::string>>(std::move(chunkBuildData->geometryGroupNames));
    vertices =
        std::make_shared<std::vector<std::vector<vk::VertexFormat::PBRVertex>>>(std::move(chunkBuildData->vertices));
    indices = std::make_shared<std::vector<std::vector<uint32_t>>>(std::move(chunkBuildData->indices));
    return lightStateChanged;
}

void Chunk1::invalidate() {
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();

    lastUpdate = std::chrono::steady_clock::now();

    blasVersion = latestVersion++;

    gc.collect(blas);
    blas = nullptr;

    gc.collect(vertexBuffers);
    vertexBuffers = nullptr;

    gc.collect(indexBuffers);
    indexBuffers = nullptr;

    gc.collect(positionBuffers);
    positionBuffers = nullptr;

    gc.collect(materialBuffers);
    materialBuffers = nullptr;
    hasLightStateHash = false;
    lightStateHash = 0;
}

std::shared_ptr<ChunkRenderData> Chunk1::tryGetValid() {
    auto ret = ChunkRenderData::create();
    ret->x = x;
    ret->y = y;
    ret->z = z;
    ret->blas = blas;
    ret->vertexBuffers = vertexBuffers;
    ret->indexBuffers = indexBuffers;
    ret->positionBuffers = positionBuffers;
    ret->materialBuffers = materialBuffers;
    ret->allVertexCount = allVertexCount;
    ret->allIndexCount = allIndexCount;
    ret->geometryCount = geometryCount;
    ret->geometryTypes = geometryTypes;
    ret->geometryGroupNames = geometryGroupNames;
    ret->vertices = vertices;
    ret->indices = indices;

    return ret;
}

Chunks::Chunks(std::shared_ptr<Framework> framework) {
    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
}

void Chunks::reset(uint32_t numChunks) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    auto vma = framework->vma();
    vkQueueWaitIdle(device->mainVkQueue());
    vkQueueWaitIdle(device->secondaryQueue());

    int size = Renderer::instance().framework()->swapchain()->imageCount();

    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();

    chunks_.clear();
    chunks_.resize(numChunks);
    chunkPackedData_ =
        vk::HostVisibleBuffer::create(vma, device, numChunks * sizeof(ChunkPackedData),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    chunkBuildDatas_.clear();
    chunkBuildDatas_.resize(numChunks);
    queuedIndex_.clear();
    hasLightingDirtySections_ = false;
    lightingDirtyQueuedThisFrame_ = false;
    lightingDirtyFramesRemaining_ = 0;
    sceneLightRevision_ = 0;
    framesSinceLastLightingDirty_ = 1024;

    for (int i = 0; i < numChunks; i++) {
        chunks_[i] = Chunk1::create();
        chunkBuildDatas_[i] = nullptr;
    }

    uint32_t chunkBuildingBatchSize = Renderer::instance().options.chunkBuildingBatchSize;
    uint32_t chunkBuildingTotalBatches = Renderer::instance().options.chunkBuildingTotalBatches;
    chunkBuildScheduler_ =
        ChunkBuildScheduler::create(queuedIndex_, chunks_, chunkBuildDatas_, mutex_, chunkPackedData_, this,
                                    chunkBuildingBatchSize, chunkBuildingTotalBatches);
}

void Chunks::resetScheduler() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (chunkBuildScheduler_ == nullptr) return;

    chunkBuildScheduler_->waitAllBatchesFinish();

    uint32_t chunkBuildingBatchSize = Renderer::instance().options.chunkBuildingBatchSize;
    uint32_t chunkBuildingTotalBatches = Renderer::instance().options.chunkBuildingTotalBatches;
    chunkBuildScheduler_ =
        ChunkBuildScheduler::create(queuedIndex_, chunks_, chunkBuildDatas_, mutex_, chunkPackedData_, this,
                                    chunkBuildingBatchSize, chunkBuildingTotalBatches);
}

void Chunks::resetFrame() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();

    gc.collect(importantBLASBuilders_);
    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();

    if (!lightingDirtyQueuedThisFrame_ && framesSinceLastLightingDirty_ < UINT32_MAX) {
        framesSinceLastLightingDirty_++;
    }
    lightingDirtyQueuedThisFrame_ = false;
    if (lightingDirtyFramesRemaining_ > 0) {
        lightingDirtyFramesRemaining_--;
        if (lightingDirtyFramesRemaining_ == 0) { hasLightingDirtySections_ = false; }
    }
}

void Chunks::invalidateChunk(int id) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id >= 0 && id < static_cast<int>(chunks_.size())) {
        auto &chunk = chunks_[id];
        if (chunk != nullptr && chunk->hasLightStateHash) {
            int sectionX = chunk->x >> 4;
            int sectionY = chunk->y >> 4;
            int sectionZ = chunk->z >> 4;
            noteLightingDirtySections(glm::ivec3(sectionX - 1, sectionY - 1, sectionZ - 1),
                                      glm::ivec3(sectionX + 1, sectionY + 1, sectionZ + 1));
        }
    }
    chunks_[id]->invalidate();

    ChunkPackedData data = {
        .geometryCount = 0,
    };

    chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData), id * sizeof(ChunkPackedData));
}

// maybe called async
void Chunks::queueChunkBuild(ChunkBuildTask task) {
    uint32_t allVertexCount = 0, allIndexCount = 0;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::string> geometryGroupNames;
    std::vector<std::vector<vk::VertexFormat::PBRVertex>> vertices;
    std::vector<std::vector<uint32_t>> indices;

    for (int i = 0; i < task.geometryCount; i++) {
        World::GeometryTypes geometryType = static_cast<World::GeometryTypes>(task.geometryTypes[i]);
        int geometryTexture = task.geometryTextures[i];
        geometryTypes.push_back(geometryType);
        if (task.geometryGroupNames != nullptr && task.geometryGroupNames[i] != nullptr) {
            geometryGroupNames.emplace_back(task.geometryGroupNames[i]);
        } else {
            geometryGroupNames.emplace_back("default");
        }

        auto &geometryVertices = vertices.emplace_back();
        auto &geometryIndices = indices.emplace_back();

        geometryVertices.resize(task.vertexCounts[i]);
        std::memcpy(geometryVertices.data(), task.vertices[i],
                    task.vertexCounts[i] * sizeof(vk::VertexFormat::PBRVertex));

        for (int j = 0; j < task.vertexCounts[i]; j += 4) {
            geometryIndices.push_back(j + 0);
            geometryIndices.push_back(j + 1);
            geometryIndices.push_back(j + 2);
            geometryIndices.push_back(j + 2);
            geometryIndices.push_back(j + 3);
            geometryIndices.push_back(j + 0);
        }

        allVertexCount += geometryVertices.size();
        allIndexCount += geometryIndices.size();
    }

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    std::unique_lock<std::recursive_mutex> lock(mutex_);

    std::shared_ptr<ChunkBuildData> chunkBuildData = ChunkBuildData::create(
        task.id, task.x, task.y, task.z, chunks_[task.id]->latestVersion++, allVertexCount, allIndexCount,
        task.geometryCount, task.lightStateHash, std::move(geometryTypes), std::move(geometryGroupNames),
        std::move(vertices), std::move(indices));

    if (task.isImportant) {
        chunkBuildData->build();
        for (int i = 0; i < chunkBuildData->geometryCount; i++) {
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->vertexBuffers[i],
                                                                      chunkBuildData->indexBuffers[i]);
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->positionBuffers[i]);
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->materialBuffers[i]);
        }
        importantBLASBuilders_->push_back(chunkBuildData->blasBuilder);

        bool lightStateChanged = chunks_[task.id]->enqueue(chunkBuildData);
        if (lightStateChanged) {
            int sectionX = task.x >> 4;
            int sectionY = task.y >> 4;
            int sectionZ = task.z >> 4;
            noteLightingDirtySections(glm::ivec3(sectionX - 1, sectionY - 1, sectionZ - 1),
                                      glm::ivec3(sectionX + 1, sectionY + 1, sectionZ + 1));
        }

        ChunkPackedData data = {
            .geometryCount = chunkBuildData->geometryCount,
        };

        chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData), chunkBuildData->id * sizeof(ChunkPackedData));
    } else {
        queuedIndex_.insert(task.id);
        chunkBuildDatas_[task.id] = chunkBuildData;
    }
}

bool Chunks::isChunkReady(int64_t id) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto chunkRenderData = chunks_[id]->tryGetValid();
    return chunkRenderData->blas != nullptr;
}

void Chunks::markLightSectionDirty(int sectionX, int sectionY, int sectionZ, int lightType) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    (void)lightType;
    noteLightingDirtySections(glm::ivec3(sectionX - 1, sectionY - 1, sectionZ - 1),
                              glm::ivec3(sectionX + 1, sectionY + 1, sectionZ + 1));
}

LightingDirtyState Chunks::lightingDirtyState() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    LightingDirtyState state{};
    state.sceneLightRevision = sceneLightRevision_;
    state.framesSinceLastDirty = framesSinceLastLightingDirty_;
    state.dirtyFramesRemaining = lightingDirtyFramesRemaining_;
    state.active = hasLightingDirtySections_ && lightingDirtyFramesRemaining_ > 0;
    if (!state.active) { return state; }

    glm::vec3 minWorld = glm::vec3(lightingDirtyMinSection_) * 16.0f;
    glm::vec3 maxWorld = glm::vec3(lightingDirtyMaxSection_ + glm::ivec3(1)) * 16.0f;
    glm::vec3 center = (minWorld + maxWorld) * 0.5f;
    glm::vec3 halfExtent = (maxWorld - minWorld) * 0.5f;
    state.centerRadius = glm::vec4(center, glm::length(halfExtent) + 16.0f);
    return state;
}

void Chunks::close() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (chunkBuildScheduler_ != nullptr) {
        chunkBuildScheduler_->waitAllBatchesFinish();
        chunkBuildScheduler_ = nullptr;
    }

    queuedIndex_.clear();
    chunkBuildDatas_.clear();
    importantBLASBuilders_ = nullptr;
    chunkPackedData_ = nullptr;
    chunks_.clear();
}

std::recursive_mutex &Chunks::mutex() {
    return mutex_;
}

std::vector<std::shared_ptr<Chunk1>> &Chunks::chunks() {
    return chunks_;
}

std::shared_ptr<ChunkBuildScheduler> Chunks::chunkBuildScheduler() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    return chunkBuildScheduler_;
}

std::vector<std::shared_ptr<vk::BLASBuilder>> &Chunks::importantBLASBuilders() {
    return *importantBLASBuilders_;
}

std::shared_ptr<vk::HostVisibleBuffer> Chunks::chunkPackedData() {
    return chunkPackedData_;
}

void Chunks::noteLightingDirtySections(const glm::ivec3 &minSection, const glm::ivec3 &maxSection) {
    if (!hasLightingDirtySections_) {
        lightingDirtyMinSection_ = minSection;
        lightingDirtyMaxSection_ = maxSection;
        hasLightingDirtySections_ = true;
    } else {
        lightingDirtyMinSection_ = glm::min(lightingDirtyMinSection_, minSection);
        lightingDirtyMaxSection_ = glm::max(lightingDirtyMaxSection_, maxSection);
    }

    if (!lightingDirtyQueuedThisFrame_) {
        sceneLightRevision_++;
        lightingDirtyQueuedThisFrame_ = true;
    }

    lightingDirtyFramesRemaining_ = std::max(lightingDirtyFramesRemaining_, 24u);
    framesSinceLastLightingDirty_ = 0;
}
