    / examples / shaders / raytrace . rmiss #L1-30
#version 460 core
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    payload = vec3(0.0); // чёрный фон при промахе
}
