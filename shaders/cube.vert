#version 450 core

// set = 0: данные кадра, общие для всех кубов
layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 projection;
} globalUniforms;

// set = 1: данные конкретного куба
layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4 model;
    vec4 baseColor;
} objectUniforms;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor; // процедурный цвет вершины

layout(location = 0) out vec3 outColor;

void main() {
    outColor = inColor;
    gl_Position = globalUniforms.projection *
                  objectUniforms.model *
                  vec4(inPosition, 1.0);
}
