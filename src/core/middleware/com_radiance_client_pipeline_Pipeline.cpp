#include "com_radiance_client_pipeline_Pipeline.h"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <exception>
#include <iostream>

namespace {

void throwJavaRuntimeException(JNIEnv *env, const char *message) {
    jclass runtimeExceptionClass = env->FindClass("java/lang/RuntimeException");
    if (runtimeExceptionClass != nullptr) {
        env->ThrowNew(runtimeExceptionClass, message != nullptr ? message : "Native pipeline build failed");
    }
}

} // namespace

JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_buildNative(JNIEnv *env, jclass, jlong paramsLongPtr) {
    try {
        WorldPipelineBuildParams *params = reinterpret_cast<WorldPipelineBuildParams *>(paramsLongPtr);
        auto pipeline = Renderer::instance().framework()->pipeline();
        if (pipeline != nullptr) Renderer::instance().framework()->pipeline()->buildWorldPipelineBlueprint(params);
    } catch (const std::exception &e) {
        throwJavaRuntimeException(env, e.what());
    } catch (...) {
        throwJavaRuntimeException(env, "Unknown native exception while building pipeline");
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_collectNativeModules(JNIEnv *, jclass) {
    Pipeline::collectWorldModules();
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_pipeline_Pipeline_isNativeModuleAvailable(JNIEnv *env,
                                                                                              jclass,
                                                                                              jstring name) {
    if (name == nullptr) return JNI_FALSE;
    const char *nativeString = env->GetStringUTFChars(name, nullptr);
    if (nativeString == nullptr) return JNI_FALSE;
    bool available = Pipeline::worldModuleConstructors.find(nativeString) != Pipeline::worldModuleConstructors.end();
    env->ReleaseStringUTFChars(name, nativeString);
    return available ? JNI_TRUE : JNI_FALSE;
}
