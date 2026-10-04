#version 450 core

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4 model;
    vec4 baseColor;
} objectUniforms;

layout(location = 0) in vec3 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    // цвет из ColorEdit умножается на интерполированный цвет вершин
    outColor = vec4(objectUniforms.baseColor.rgb * inColor,
                    objectUniforms.baseColor.a);
}
