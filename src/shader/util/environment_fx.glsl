#ifndef ENVIRONMENT_FX_GLSL
#define ENVIRONMENT_FX_GLSL

const float WATER_SURFACE_SENTINEL_MIN = 31.0;
const float WATER_SURFACE_SENTINEL_MAX = 31.5;

const float WATER_MODE_NATIVE_LIKE = 0.0;
const float WATER_MODE_SPARKLING = 1.0;

const float CLOUD_MODE_NATIVE = 0.0;
const float CLOUD_MODE_EFFICIENT = 1.0;
const float CLOUD_MODE_REALISTIC = 2.0;

float environmentHash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float environmentValueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = environmentHash13(i + vec3(0.0, 0.0, 0.0));
    float n100 = environmentHash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = environmentHash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = environmentHash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = environmentHash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = environmentHash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = environmentHash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = environmentHash13(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

float environmentFbm(vec3 p, int octaves) {
    float sum = 0.0;
    float amplitude = 0.5;
    vec3 q = p;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) {
            break;
        }
        sum += amplitude * environmentValueNoise(q);
        q = q * 2.03 + vec3(13.1, 7.7, 11.3);
        amplitude *= 0.5;
    }
    return sum;
}

bool decodeWaterSurfaceSentinel(inout float encodedEmission) {
    if (encodedEmission >= WATER_SURFACE_SENTINEL_MIN
        && encodedEmission <= WATER_SURFACE_SENTINEL_MAX) {
        encodedEmission = 0.0;
        return true;
    }
    return false;
}

vec3 applyWaterSurfaceNormal(vec3 baseNormal,
                             vec3 geoNormal,
                             vec3 worldPos,
                             float gameTime,
                             float waterMode) {
    vec3 resolvedGeoNormal = normalize(geoNormal);
    vec3 resolvedBaseNormal = normalize(baseNormal);
    if (abs(resolvedGeoNormal.y) < 0.35) {
        return resolvedBaseNormal;
    }

    float sparkleMix = step(0.5, waterMode);
    float time = gameTime * mix(0.55, 1.05, sparkleMix);
    vec2 p = worldPos.xz * mix(0.11, 0.16, sparkleMix);

    float dx =
        cos(p.x + time * 1.3) * 0.42
            + sin((p.x + p.y) * 0.65 - time * 1.1) * 0.24
            + (environmentFbm(vec3(p * 1.8, time * 0.35), sparkleMix > 0.5 ? 5 : 3) - 0.5)
            * mix(0.22, 0.42, sparkleMix);
    float dz =
        -sin(p.y * 1.1 - time * 0.9) * 0.34
            + cos((p.x - p.y) * 0.72 + time * 1.45) * 0.2
            + (environmentValueNoise(vec3(p.yx * 2.1, time * 0.29)) - 0.5)
            * mix(0.18, 0.38, sparkleMix);

    vec3 tangentX = normalize(vec3(1.0, 0.0, 0.0) - resolvedGeoNormal * resolvedGeoNormal.x);
    if (dot(tangentX, tangentX) <= 1e-6) {
        tangentX = normalize(cross(vec3(0.0, 0.0, 1.0), resolvedGeoNormal));
    }
    vec3 tangentZ = normalize(cross(resolvedGeoNormal, tangentX));

    float amplitude = mix(0.075, 0.22, sparkleMix);
    vec3 proceduralNormal =
        normalize(resolvedGeoNormal + tangentX * dx * amplitude + tangentZ * dz * amplitude);
    return normalize(mix(resolvedBaseNormal, proceduralNormal, mix(0.42, 0.82, sparkleMix)));
}

float cloudVerticalProfile(float localY) {
    float h = clamp(localY / 4.0, 0.0, 1.0);
    return smoothstep(0.04, 0.24, h) * (1.0 - smoothstep(0.62, 0.98, h));
}

float sampleCloudDensity(vec3 sampleWorldPos, float localY, float gameTime, float cloudMode) {
    float realism = clamp((cloudMode - CLOUD_MODE_EFFICIENT)
        / max(CLOUD_MODE_REALISTIC - CLOUD_MODE_EFFICIENT, 1.0), 0.0, 1.0);
    float vertical = cloudVerticalProfile(localY);
    if (vertical <= 0.0) {
        return 0.0;
    }

    float drift = gameTime * mix(0.015, 0.026, realism);
    vec3 coarseCoord = vec3(sampleWorldPos.x * 0.028 + drift, localY * 0.24,
        sampleWorldPos.z * 0.028 - drift * 0.7);
    vec3 detailCoord = vec3(sampleWorldPos.x * 0.09 - drift * 1.8, localY * 0.55 + drift * 0.5,
        sampleWorldPos.z * 0.09 + drift);

    float coarse = environmentFbm(coarseCoord, realism > 0.5 ? 5 : 3);
    float detail = environmentFbm(detailCoord, realism > 0.5 ? 4 : 2);
    float wisps = environmentValueNoise(detailCoord * 1.7 + vec3(4.2, 1.9, 7.3));

    float densityField = coarse * mix(0.92, 0.82, realism)
        + detail * mix(0.22, 0.36, realism)
        + wisps * mix(0.06, 0.14, realism);
    float threshold = mix(0.54, 0.46, realism);
    return smoothstep(threshold, 0.95, densityField) * vertical;
}

#endif
