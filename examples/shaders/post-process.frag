#version 460

layout(binding = 0) uniform UniformBufferObject {
    vec2 resolution;
} ubo;

layout(binding = 1) uniform sampler2D color_sampler;

layout(location = 0) in vec2 UV;
layout(location = 0) out vec4 fragColor;

const float VIGNETTE_INTENSITY = 0.8;
const float VIGNETTE_SMOOTHNESS = 0.3;

void main() {
    float aspect_ratio = ubo.resolution.x / ubo.resolution.y;
    vec2 coord = (UV - 0.5) * 1.0;
    coord.x *= aspect_ratio;
    float dist = length(coord);
    float vignette = 1.0 - smoothstep(VIGNETTE_SMOOTHNESS, 1.0, dist);
    vignette = mix(1.0, vignette, VIGNETTE_INTENSITY);

    vec4 color = texture(color_sampler, UV);
    fragColor = color * vignette;
}
