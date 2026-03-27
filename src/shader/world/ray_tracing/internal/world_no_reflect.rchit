#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

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

layout(set = 1, binding = 9) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

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

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    float directLightStrength;
    float indirectLightStrength;
    float basicRadiance;
    uint pbrSamplingMode;
    uint transparentSplitMode;
    float farFieldStartDistanceChunks;
    uint farFieldMaterialMode;
}
pc;

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

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
    bool useFarFieldFlatMaterial =
        pc.farFieldMaterialMode != 0u &&
        dot(worldPos, worldPos) >= pow(pc.farFieldStartDistanceChunks * 16.0, 2.0);
    float farFieldBaseColorBias = farFieldBaseColorLodBias(worldPos, pc.farFieldStartDistanceChunks);
    float farFieldPbrBias = farFieldPbrDetailLodBias(worldPos, pc.farFieldStartDistanceChunks);
    uint coordinate = v0.coordinate;
    vec3 normal = baryCoords.x * v0.norm + baryCoords.y * v1.norm + baryCoords.z * v2.norm;
    if (coordinate == 1) {
        normal = normalize(mat3(worldUbo.cameraViewMatInv) * normal);
    } else {
        normal = normalize(normal);
    }

    uint useColorLayer = v0.useColorLayer;
    vec4 colorLayerValue;
    if (useColorLayer > 0) {
        colorLayerValue = baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer;
    } else {
        colorLayerValue = vec4(1.0);
    }
    vec3 colorLayer = colorLayerValue.rgb;

    uint useTexture = v0.useTexture;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    bool forceNoPbrMaterial = (v0.alphaMode & 0x100u) != 0u;
    uint alphaMode = v0.alphaMode & 0xFu;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    vec2 textureUV;
    if (useTexture > 0) {
        int specularTextureID = mapping.entries[textureID].specular;
        int normalTextureID = mapping.entries[textureID].normal;
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;
        vec2 atlasUvMin = min(v0.textureUV, min(v1.textureUV, v2.textureUV));
        vec2 atlasUvMax = max(v0.textureUV, max(v1.textureUV, v2.textureUV));

        // ray cone
        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        vec3 dposdu, dposdv;
        computedposduDv(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, dposdu, dposdv);
        float lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);
        float baseColorLod = lod + farFieldBaseColorBias;
        float detailLod = lod + farFieldPbrBias;

        albedoValue = sampleTexture(textures[nonuniformEXT(textureID)], textureUV, baseColorLod, false);
        albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
        if (useFarFieldFlatMaterial || forceNoPbrMaterial) {
            specularValue = vec4(0.0);
            normalValue = vec4(0.5, 0.5, 1.0, 0.0);
        } else {
            if (specularTextureID >= 0) {
                specularValue = sampleTexture(textures[nonuniformEXT(specularTextureID)], textureUV, detailLod, false);
            } else {
                specularValue = vec4(0.0);
            }
            if (normalTextureID >= 0) {
                normalValue = samplePBRTexture(textures[nonuniformEXT(normalTextureID)], textureUV, atlasUvMin,
                                               atlasUvMax, detailLod, pc.pbrSamplingMode);
            } else {
                normalValue = vec4(0.5, 0.5, 1.0, 0.0);
            }
        }
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.5, 0.5, 1.0, 0.0);
    }

    uint useGlint = v0.useGlint;
    uint glintTexture = v0.glintTexture;
    vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
    glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
    vec3 glint =
        useGlint > 0 && !useFarFieldFlatMaterial ? sampleTexture(textures[nonuniformEXT(glintTexture)], glintUV, false).rgb
                                                 : vec3(0.0);
    glint = glint * glint;

    uint useOverlay = v0.useOverlay;
    vec3 tint = albedoValue.rgb * colorLayer + glint;
    if (useOverlay > 0 && !useFarFieldFlatMaterial) {
        ivec2 overlayUV = v0.overlayUV;
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    if (forceNoPbrMaterial) {
        mat.roughness = max(mat.roughness, 0.92);
        mat.metallic = 0.0;
        mat.transmission = 0.0;
        mat.f0 = vec3(0.04);
        mat.emission = 0.0;
    }

    // add glowing radiance
    mainRay.radiance += 12 * tint * mat.emission * mainRay.throughput;
    mainRay.hitT = gl_HitTEXT;
    mainRay.normal = vec3(0.0);
    rayStoreMaterial(mainRay, albedoValue, mat.f0, mat.roughness, mat.metallic, mat.transmission, mat.ior, mat.emission);
    raySetNoisy(mainRay, false);
    mainRay.hasPrevScenePos = 0u;
    raySetStop(mainRay, true);
}
