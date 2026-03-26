#include "com_radiance_client_proxy_world_ChunkProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"

#include <iostream>

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_initNative(JNIEnv *, jclass, jint chunkNum) {
    Renderer::instance().world()->chunks()->reset(chunkNum);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_rebuildSingle(JNIEnv *,
                                                                                     jclass,
                                                                                     jint originX,
                                                                                     jint originY,
                                                                                     jint originZ,
                                                                                     jlong index,
                                                                                     jint geometryCount,
                                                                                     jlong geometryTypes,
                                                                                     jlong geometryGroupNames,
                                                                                     jlong geometryTextures,
                                                                                     jlong vertexFormats,
                                                                                     jlong vertexCounts,
                                                                                     jlong vertexAddrs,
                                                                                     jlong lightStateHash,
                                                                                     jboolean important) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->queueChunkBuild(ChunkBuildTask{
        .x = originX,
        .y = originY,
        .z = originZ,
        .id = index,
        .geometryCount = geometryCount,
        .geometryTypes = reinterpret_cast<int *>(geometryTypes),
        .geometryGroupNames = reinterpret_cast<const char **>(geometryGroupNames),
        .geometryTextures = reinterpret_cast<int *>(geometryTextures),
        .vertexFormats = reinterpret_cast<int *>(vertexFormats),
        .vertexCounts = reinterpret_cast<int *>(vertexCounts),
        .vertices = reinterpret_cast<vk::VertexFormat::PBRVertex **>(vertexAddrs),
        .lightStateHash = static_cast<uint64_t>(lightStateHash),
        .isImportant = static_cast<bool>(important),
    });
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_isChunkReady(JNIEnv *, jclass, jlong id) {
    auto world = Renderer::instance().world();
    if (world == nullptr)
        return false;
    else
        return world->chunks()->isChunkReady(id);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_invalidateSingle(JNIEnv *, jclass, jlong index) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->invalidateChunk(index);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_markLightDirtySection(JNIEnv *,
                                                                                             jclass,
                                                                                             jint sectionX,
                                                                                             jint sectionY,
                                                                                             jint sectionZ,
                                                                                             jint lightType) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->markLightSectionDirty(sectionX, sectionY, sectionZ, lightType);
}
