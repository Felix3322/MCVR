#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/disney.glsl"
#include "util/environment_fx.glsl"
#include "util/random.glsl"
#include "util/ray.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform samplerCube skyFull;

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 4) readonly buffer LastVertexBufferAddr {
    uint64_t addrs[];
}
lastVertexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 10) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 3, binding = 1, rgba8) uniform image2D diffuseAlbedoImage;
layout(set = 3, binding = 2, rgba8) uniform image2D specularAlbedoImage;
layout(set = 3, binding = 3, rgba16f) uniform image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg16f) uniform image2D motionVectorImage;
layout(set = 3, binding = 5, r16f) uniform image2D linearDepthImage;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRVertex vertices[];
}
vertexBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

layout(location = 0) rayPayloadInEXT MainRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

float sampleCloudShadow(vec3 sampleWorldPos,
                        float localY,
                        vec3 sunDir,
                        float gameTime,
                        float cloudMode) {
    float realism = clamp((cloudMode - CLOUD_MODE_EFFICIENT)
        / max(CLOUD_MODE_REALISTIC - CLOUD_MODE_EFFICIENT, 1.0), 0.0, 1.0);
    int lightSteps = realism > 0.5 ? 6 : 3;
    float occlusion = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= lightSteps) {
            break;
        }
        float t = (float(i) + 1.0) * mix(1.0, 1.35, realism);
        vec3 lightSamplePos = sampleWorldPos + sunDir * t;
        float lightSampleY = localY + sunDir.y * t;
        occlusion += sampleCloudDensity(lightSamplePos, lightSampleY, gameTime, cloudMode);
    }
    return exp(-occlusion * mix(0.95, 1.3, realism));
}

vec3 renderVolumetricCloud(vec3 worldPos,
                           vec3 localPos,
                           vec3 normal,
                           vec3 viewDir,
                           vec3 sunDir,
                           vec3 tint,
                           vec3 skyAmbient,
                           float rainGradient,
                           float gameTime,
                           float cloudMode) {
    float realism = clamp((cloudMode - CLOUD_MODE_EFFICIENT)
        / max(CLOUD_MODE_REALISTIC - CLOUD_MODE_EFFICIENT, 1.0), 0.0, 1.0);
    vec3 inwardDir = dot(normal, viewDir) > 0.0 ? -normal : normal;
    inwardDir = normalize(inwardDir);

    int marchSteps = realism > 0.5 ? 9 : 5;
    float stepLength = mix(0.7, 0.52, realism);
    float transmittance = 1.0;
    float accumulatedDensity = 0.0;
    float accumulatedLight = 0.0;
    for (int i = 0; i < 9; i++) {
        if (i >= marchSteps) {
            break;
        }
        float t = (float(i) + 0.5) * stepLength;
        vec3 samplePos = worldPos + inwardDir * t;
        float sampleY = localPos.y + inwardDir.y * t;
        float density = sampleCloudDensity(samplePos, sampleY, gameTime, cloudMode);
        if (density <= 1e-4) {
            continue;
        }

        float shadow = sampleCloudShadow(samplePos, sampleY, sunDir, gameTime, cloudMode);
        float forwardPhase = pow(max(dot(viewDir, sunDir), 0.0), mix(6.0, 14.0, realism));
        float silverLining = pow(max(dot(-inwardDir, sunDir), 0.0), mix(8.0, 18.0, realism));
        float localLight = shadow * mix(0.52, 0.82, realism)
            + forwardPhase * mix(0.08, 0.18, realism)
            + silverLining * mix(0.18, 0.42, realism);

        accumulatedLight += density * transmittance * localLight;
        accumulatedDensity += density * transmittance;
        transmittance *= exp(-density * mix(0.95, 1.25, realism));
        if (transmittance < 0.02) {
            break;
        }
    }

    float opacity = clamp(1.0 - transmittance, 0.0, 1.0);
    float dayFactor = smoothstep(-0.3, 0.35, sunDir.y);
    vec3 ambient = tint * skyAmbient * mix(0.18, 0.32, realism) * max(opacity, accumulatedDensity * 0.42);
    vec3 direct = tint * accumulatedLight * mix(0.75, 1.08, realism);
    vec3 silver = mix(vec3(0.0), vec3(1.0, 0.97, 0.92), dayFactor)
        * pow(max(dot(viewDir, -sunDir), 0.0), mix(10.0, 22.0, realism))
        * opacity * mix(0.16, 0.36, realism);
    vec3 rainyLift = mix(skyAmbient * 0.08, vec3(0.1, 0.11, 0.12) * opacity, 0.55);
    vec3 litCloud = ambient + direct + silver;
    return mix(litCloud, rainyLift + ambient * 0.85, rainGradient);
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint blasOffset = blasOffsets.offsets[instanceID];

    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[blasOffset + geometryID]);
    uint indexBaseID = 3 * gl_PrimitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1];
    uint i2 = indexBuffer.indices[indexBaseID + 2];

    VertexBuffer vertexBuffer = VertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
    PBRVertex v0 = vertexBuffer.vertices[i0];
    PBRVertex v1 = vertexBuffer.vertices[i1];
    PBRVertex v2 = vertexBuffer.vertices[i2];

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    uint coordinate = v0.coordinate;
    vec3 normal = baryCoords.x * v0.norm + baryCoords.y * v1.norm + baryCoords.z * v2.norm;
    if (coordinate == 1) {
        normal = normalize(mat3(worldUbo.cameraViewMatInv) * normal);
    } else {
        normal = normalize(normal);
    }
    uint useColorLayer = v0.useColorLayer;
    vec3 colorLayer;
    if (useColorLayer > 0) {
        colorLayer = (baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer).rgb;
    } else {
        colorLayer = vec3(1.0);
    }

    uint useTexture = v0.useTexture;
    vec2 textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;
    uint textureID = v0.textureID;

    vec4 albedoSample = vec4(1.0);
    if (useTexture > 0) { albedoSample = sampleTexture(textures[nonuniformEXT(textureID)], textureUV, false); }

    vec3 albedo = albedoSample.rgb;
    float alpha = albedoSample.a;
    vec3 tint = albedo * colorLayer;

    LabPBRMat mat;
    mat.albedo = tint;
    mat.f0 = vec3(0.04);
    mat.roughness = 1.0;
    mat.metallic = 0.0;
    mat.subSurface = 0.0;
    mat.transmission = 0.0;
    mat.ior = 1.5;
    mat.emission = 0.0;

    vec3 sunDir = normalize(skyUBO.sunDirection);
    float progress = skyUBO.rainGradient;
    vec3 skyAmbient = texture(skyFull, normalize(viewDir)).rgb;
    if (sunDir.y < 0) {
        sunDir = -sunDir;
    }

    mainRay.hitT = gl_HitTEXT;

    float cloudMode = skyUBO.pad1;
    if (cloudMode <= CLOUD_MODE_NATIVE + 0.25) {
        vec3 rayOrigin = worldPos;
        vec3 sampledLightDir = sunDir;

        // Clouds are mostly viewed from below; keep underside lit via two-sided + backlit response.
        float ndotl = dot(normal, sampledLightDir);
        float ndotv = dot(normal, viewDir);
        float frontLit = max(ndotl, 0.0);
        float wrappedLit = clamp((abs(ndotl) + 0.4) / 1.4, 0.0, 1.0);
        float backLit = max(-ndotl, 0.0) * max(-ndotv, 0.0);
        float cloudPhase = max(frontLit, max(wrappedLit * 0.6, backLit * 1.35));
        vec3 lightBRDF = tint * (INV_PI * cloudPhase);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.bounceIndex = rayBounce(mainRay);
        traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                    WORLD_MASK, // masks
                    0,          // sbtRecordOffset
                    0,          // sbtRecordStride
                    0,          // missIndex
                    rayOrigin, 0.001, sampledLightDir, 1000, 1);

        vec3 lightContribution = shadowRay.radiance;
        vec3 lightRadiance = lightContribution * mainRay.throughput * lightBRDF;
        lightRadiance *= alpha * 0.65;

        float dayFactor = smoothstep(-0.3, 0.3, sunDir.y);
        vec3 rainyRadiance = mix(skyAmbient * 0.12, vec3(0.08), dayFactor);
        vec3 wetCloudRadiance = lightRadiance * mix(0.2, 0.35, dayFactor) + rainyRadiance;
        mainRay.radiance += mix(lightRadiance, wetCloudRadiance, progress);
    } else {
        vec3 localPos = baryCoords.x * v0.pos + baryCoords.y * v1.pos + baryCoords.z * v2.pos;
        vec3 volumetricRadiance =
            renderVolumetricCloud(worldPos, localPos, normal, viewDir, sunDir, tint, skyAmbient,
                progress, worldUbo.gameTime, cloudMode);
        mainRay.radiance += volumetricRadiance * alpha;
    }

    mainRay.hitT = gl_HitTEXT;
    mainRay.normal = vec3(0.0);
    rayClearMaterial(mainRay);
    raySetNoisy(mainRay, false);
    raySetSkipFog(mainRay, true);
    mainRay.hasPrevScenePos = 0u;
    raySetStop(mainRay, true);
}
