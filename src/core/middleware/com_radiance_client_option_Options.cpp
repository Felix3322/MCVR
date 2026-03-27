#include "com_radiance_client_option_Options.h"

#include "core/all_extern.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/streamline_context.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

namespace {

int getDisplayRefreshRate() {
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) return 0;

    auto framework = renderer->framework();
    if (framework == nullptr) return 0;

    auto window = framework->window();
    if (window == nullptr) return 0;

    GLFWmonitor *monitor = GLFW_GetWindowMonitor(window->window());
    if (monitor == nullptr) monitor = GLFW_GetPrimaryMonitor();
    if (monitor == nullptr) return 0;

    const GLFWvidmode *mode = GLFW_GetVideoMode(monitor);
    return mode == nullptr ? 0 : mode->refreshRate;
}

void applyReflexSettings() {
    if (!StreamlineContext::isReflexAvailable()) return;

    sl::ReflexMode mode = sl::ReflexMode::eOff;
    if (Renderer::options.reflexEnabled) {
        mode = Renderer::options.reflexBoost ? sl::ReflexMode::eLowLatencyWithBoost : sl::ReflexMode::eLowLatency;
    }

    uint32_t frameLimitUs = 0;
    if (Renderer::options.vrrMode && Renderer::options.reflexEnabled) {
        int hz = getDisplayRefreshRate();
        if (hz > 0) {
            uint32_t targetFps = (3600u * static_cast<uint32_t>(hz)) / (static_cast<uint32_t>(hz) + 3600u);
            if (targetFps > 0) frameLimitUs = 1000000u / targetFps;
        }
    } else {
        uint32_t maxFps = Renderer::options.maxFps;
        if (maxFps > 0 && maxFps < 1000000u) {
            frameLimitUs = 1000000u / maxFps;
        }
    }

    StreamlineContext::setReflexOptions(mode, frameLimitUs);
}

} // namespace

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMaxFps(JNIEnv *,
                                                                               jclass,
                                                                               jint maxFps,
                                                                               jboolean write) {
    Renderer::options.maxFps = maxFps;
    applyReflexSettings();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetInactivityFpsLimit(JNIEnv *,
                                                                                           jclass,
                                                                                           jint inactivityFpsLimit,
                                                                                           jboolean write) {
    Renderer::options.inactivityFpsLimit = inactivityFpsLimit;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetVsync(JNIEnv *,
                                                                              jclass,
                                                                              jboolean vsync,
                                                                              jboolean write) {
    Renderer::options.vsync = vsync;
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrOutput(JNIEnv *,
                                                                                  jclass,
                                                                                  jboolean hdrOutput,
                                                                                  jboolean write) {
    Renderer::options.hdrOutput = hdrOutput;
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssFrameGeneration(JNIEnv *,
                                                                                             jclass,
                                                                                             jboolean enabled,
                                                                                             jboolean write) {
    Renderer::options.dlssFrameGeneration = enabled;
    auto world = Renderer::is_initialized() ? Renderer::instance().world() : nullptr;
    bool worldRendering = world != nullptr && world->shouldRender();

    if (!enabled) {
        if (Renderer::options.dlssFrameGenerationActive) {
            Renderer::options.needRecreate = true;
        }
        Renderer::options.dlssFrameGenerationActive = false;
        return;
    }

    if (write && worldRendering) {
        Renderer::options.needRecreate = true;
    }
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_option_Options_nativeHasDlssFrameGenerationAvailable(JNIEnv *,
                                                                                                          jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return JNI_FALSE;
    return framework->hasDlssFrameGenerationAvailable() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRayBounces(JNIEnv *,
                                                                                   jclass,
                                                                                   jint rayBounces,
                                                                                   jboolean write) {
    Renderer::options.rayBounces = rayBounces;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingBatchSize(
    JNIEnv *, jclass, jint chunkBuildingBatchSize, jboolean write) {
    Renderer::options.chunkBuildingBatchSize = chunkBuildingBatchSize;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingTotalBatches(
    JNIEnv *, jclass, jint chunkBuildingTotalBatches, jboolean write) {
    Renderer::options.chunkBuildingTotalBatches = chunkBuildingTotalBatches;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetReflexEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.reflexEnabled = enabled;
    applyReflexSettings();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetReflexBoost(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.reflexBoost = enabled;
    applyReflexSettings();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOutputScale2x(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.outputScale2x = enabled;
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSimplifiedIndirect(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.simplifiedIndirect = enabled;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_option_Options_nativeIsReflexSupported(
    JNIEnv *, jclass) {
    return StreamlineContext::isReflexAvailable() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetVrrMode(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.vrrMode = enabled;
    applyReflexSettings();
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetDisplayRefreshRate(
    JNIEnv *, jclass) {
    return static_cast<jint>(getDisplayRefreshRate());
}
