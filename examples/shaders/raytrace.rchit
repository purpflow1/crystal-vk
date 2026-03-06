
    / examples / shaders / raytrace . rchit #L1-60
#version 460 core
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

// Наш треугольник задаёт цвет через атрибут вершины (в данном примере просто берём позицию)
hitAttributeEXT vec2 barycentrics;

void main() {
    // простой подход: берём позицию во входных атрибутах
    // (в реальном примере здесь бы читали буфер вершин)
    payload = vec3(1.0, 0.5, 0.2); // фиксированный цвет треугольника
}
