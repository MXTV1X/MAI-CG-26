#include "application.hpp"
#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

const auto& context = graphics::internal::context;

using lab_math::Mat4;
using lab_math::Vec3;

constexpr uint32_t cube_count = 3;

constexpr float cube_half_size = 0.6f;

enum class ProjectionType {
    Perspective = 0,
    Orthographic = 1,
};

struct Vertex {
    float position[3];
    float color[3];
};

struct GlobalUniforms {
    float projection[16];
};

struct ObjectUniforms {
    float model[16];
    float baseColor[4];
};

struct GpuBuffer {
    VkBuffer buffer = nullptr;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
    size_t size = 0;
};

struct Mesh {
    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;

    uint32_t triangleIndexCount = 0;
    uint32_t edgeFirstIndex = 0;
    uint32_t edgeIndexCount = 0;
};

struct Cube {
    const char* name = "";

    Mesh mesh;

    Vec3 position{};
    Vec3 rotationDegrees{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    float color[3] = {1.0f, 1.0f, 1.0f};

    float phaseOffset = 0.0f;
    float animationMultiplier = 1.0f;
    bool animate = true;

    Mat4 modelMatrix{};

    VkDescriptorSet descriptorSet = nullptr;
    GpuBuffer uniformBuffer;
};

VkDescriptorSetLayout globalDescriptorSetLayout = nullptr;
VkDescriptorSetLayout objectDescriptorSetLayout = nullptr;
VkDescriptorPool descriptorPool = nullptr;
VkDescriptorSet globalDescriptorSet = nullptr;
GpuBuffer globalUniformBuffer;

VkPipelineLayout pipelineLayout = nullptr;
VkPipeline fillPipeline = nullptr;
VkPipeline edgePipeline = nullptr;

VkShaderModule cubeVertexShader = nullptr;
VkShaderModule cubeFragmentShader = nullptr;
VkShaderModule edgeVertexShader = nullptr;
VkShaderModule edgeFragmentShader = nullptr;

Cube cubes[cube_count];

ProjectionType projectionType = ProjectionType::Perspective;

float orthographicHeight = 8.5f;
float perspectiveFovDegrees = 55.0f;
float nearPlane = 0.1f;
float farPlane = 30.0f;

double previousWallTime = 0.0;
double animationTime = 0.0;
bool timeInitialized = false;

bool animationPlaying = true;
float animationSpeed = 1.0f;
float trajectoryRadius = 1.0f;
float trajectoryVerticalAmplitude = 0.8f;
float trajectoryDepthAmplitude = 0.7f;
float trajectoryVerticalFrequency = 2.0f;
float trajectoryDepthFrequency = 0.5f;
Vec3 animatedRotationSpeedDegrees{20.0f, 30.0f, 40.0f};

bool resetAnimationRequested = false;

bool showEdges = true;

bool createHostBuffer(size_t size, VkBufferUsageFlags usage, GpuBuffer& result) {
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocationInfo = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
                        &result.buffer, &result.allocation,
                        nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    if (vmaMapMemory(context.allocator, result.allocation, &result.mapped) != VK_SUCCESS) {
        std::cerr << "Failed to map buffer memory\n";
        return false;
    }
    result.size = size;

    return true;
}

bool loadShaderModule(const char* path, VkShaderModule& shaderModule) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open shader: " << path << '\n';
        return false;
    }

    const std::streamsize fileSize = file.tellg();

    std::vector<uint32_t> code(static_cast<size_t>(fileSize / sizeof(uint32_t)));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);
    file.close();

    const VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<size_t>(fileSize),
        .pCode = code.data(),
    };

    if (vkCreateShaderModule(context.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << '\n';
        return false;
    }

    return true;
}

constexpr uint32_t cube_vertex_count = 8;

constexpr uint32_t cube_triangle_indices[36] = {
    0, 1, 2,  0, 2, 3,
    4, 7, 6,  4, 6, 5,
    0, 4, 5,  0, 5, 1,
    3, 2, 6,  3, 6, 7,
    0, 3, 7,  0, 7, 4,
    1, 5, 6,  1, 6, 2,
};

constexpr uint32_t cube_edge_indices[24] = {
    0, 1,  1, 2,  2, 3,  3, 0,
    4, 5,  5, 6,  6, 7,  7, 4,
    0, 4,  1, 5,  2, 6,  3, 7,
};

std::vector<Vertex> makeCubeVertices() {
    std::vector<Vertex> vertices;
    vertices.reserve(cube_vertex_count);

    const float signs[cube_vertex_count][3] = {
        {-1.0f, -1.0f, -1.0f}, {+1.0f, -1.0f, -1.0f},
        {+1.0f, +1.0f, -1.0f}, {-1.0f, +1.0f, -1.0f},
        {-1.0f, -1.0f, +1.0f}, {+1.0f, -1.0f, +1.0f},
        {+1.0f, +1.0f, +1.0f}, {-1.0f, +1.0f, +1.0f},
    };

    for (const auto& s : signs) {
        Vertex v{};
        for (int axis = 0; axis < 3; ++axis) {
            v.position[axis] = s[axis] * cube_half_size;
            // локальная позиция [-h, h] -> цвет [0, 1]
            v.color[axis] = v.position[axis] / (2.0f * cube_half_size) + 0.5f;
        }
        vertices.push_back(v);
    }

    return vertices;
}

bool createMesh(Cube& cube) {
    const std::vector<Vertex> vertices = makeCubeVertices();

    std::vector<uint32_t> indices;
    indices.insert(indices.end(), std::begin(cube_triangle_indices),
                   std::end(cube_triangle_indices));
    const uint32_t edgeFirstIndex = static_cast<uint32_t>(indices.size());
    indices.insert(indices.end(), std::begin(cube_edge_indices),
                   std::end(cube_edge_indices));

    cube.mesh.triangleIndexCount = edgeFirstIndex;
    cube.mesh.edgeFirstIndex = edgeFirstIndex;
    cube.mesh.edgeIndexCount = static_cast<uint32_t>(indices.size()) - edgeFirstIndex;

    if (!createHostBuffer(sizeof(Vertex) * vertices.size(),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          cube.mesh.vertexBuffer)) {
        return false;
    }
    std::memcpy(cube.mesh.vertexBuffer.mapped, vertices.data(),
                sizeof(Vertex) * vertices.size());

    if (!createHostBuffer(sizeof(uint32_t) * indices.size(),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          cube.mesh.indexBuffer)) {
        return false;
    }
    std::memcpy(cube.mesh.indexBuffer.mapped, indices.data(),
                sizeof(uint32_t) * indices.size());

    return true;
}

bool createDescriptorLayouts() {
    const VkDescriptorSetLayoutBinding globalBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo globalLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &globalBinding,
    };

    if (vkCreateDescriptorSetLayout(context.device, &globalLayoutInfo, nullptr,
                                    &globalDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create global descriptor set layout\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding objectBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo objectLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &objectBinding,
    };

    if (vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr,
                                    &objectDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create object descriptor set layout\n";
        return false;
    }

    return true;
}

bool createDescriptorPoolAndSets() {
    constexpr uint32_t setCount = cube_count + 1;

    const VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = setCount,
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = setCount,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create application descriptor pool\n";
        return false;
    }

    if (!createHostBuffer(sizeof(GlobalUniforms),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          globalUniformBuffer)) {
        return false;
    }

    const VkDescriptorSetAllocateInfo globalAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &globalDescriptorSetLayout,
    };

    if (vkAllocateDescriptorSets(context.device, &globalAllocateInfo,
                                 &globalDescriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate global descriptor set\n";
        return false;
    }

    for (Cube& cube : cubes) {
        if (!createHostBuffer(sizeof(ObjectUniforms),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              cube.uniformBuffer)) {
            return false;
        }

        const VkDescriptorSetAllocateInfo objectAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &objectDescriptorSetLayout,
        };

        if (vkAllocateDescriptorSets(context.device, &objectAllocateInfo,
                                     &cube.descriptorSet) != VK_SUCCESS) {
            std::cerr << "Failed to allocate object descriptor set\n";
            return false;
        }
    }

    const VkDescriptorBufferInfo globalBufferInfo = {
        .buffer = globalUniformBuffer.buffer,
        .offset = 0,
        .range = sizeof(GlobalUniforms),
    };

    const VkWriteDescriptorSet globalWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = globalDescriptorSet,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &globalBufferInfo,
    };

    vkUpdateDescriptorSets(context.device, 1, &globalWrite, 0, nullptr);

    for (Cube& cube : cubes) {
        const VkDescriptorBufferInfo objectBufferInfo = {
            .buffer = cube.uniformBuffer.buffer,
            .offset = 0,
            .range = sizeof(ObjectUniforms),
        };

        const VkWriteDescriptorSet objectWrite = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = cube.descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &objectBufferInfo,
        };

        vkUpdateDescriptorSets(context.device, 1, &objectWrite, 0, nullptr);
    }

    return true;
}

VkPipelineColorBlendStateCreateInfo makeBlendState(VkPipelineColorBlendAttachmentState& attachment) {
    attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };

    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &attachment,
    };
}

bool createPipelines() {
    const VkDescriptorSetLayout setLayouts[2] = {
        globalDescriptorSetLayout,
        objectDescriptorSetLayout,
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = setLayouts,
    };

    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    if (!loadShaderModule("shaders/cube.vert.spv", cubeVertexShader) ||
        !loadShaderModule("shaders/cube.frag.spv", cubeFragmentShader) ||
        !loadShaderModule("shaders/edge.vert.spv", edgeVertexShader) ||
        !loadShaderModule("shaders/edge.frag.spv", edgeFragmentShader)) {
        return false;
    }

    const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription attributes[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertexInput = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo fillAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineInputAssemblyStateCreateInfo edgeAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo fillRaster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,
        .depthBiasConstantFactor = 1.0f,
        .depthBiasSlopeFactor = 1.0f,
        .lineWidth = 1.0f,
    };

    const VkPipelineRasterizationStateCreateInfo edgeRaster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampleState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo fillDepth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineDepthStencilStateCreateInfo edgeDepth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    const VkDynamicState dynamicStates[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    VkPipelineColorBlendAttachmentState fillAttachment{};
    const VkPipelineColorBlendStateCreateInfo fillBlend = makeBlendState(fillAttachment);

    VkPipelineColorBlendAttachmentState edgeAttachment{};
    const VkPipelineColorBlendStateCreateInfo edgeBlend = makeBlendState(edgeAttachment);

    const VkPipelineShaderStageCreateInfo fillStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = cubeVertexShader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = cubeFragmentShader,
            .pName = "main",
        },
    };

    const VkPipelineShaderStageCreateInfo edgeStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = edgeVertexShader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = edgeFragmentShader,
            .pName = "main",
        },
    };

    const VkGraphicsPipelineCreateInfo fillPipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = fillStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &fillAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &fillRaster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &fillDepth,
        .pColorBlendState = &fillBlend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, nullptr, 1,
                                  &fillPipelineInfo, nullptr,
                                  &fillPipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create cube fill pipeline\n";
        return false;
    }

    const VkGraphicsPipelineCreateInfo edgePipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = edgeStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &edgeAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &edgeRaster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &edgeDepth,
        .pColorBlendState = &edgeBlend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, nullptr, 1,
                                  &edgePipelineInfo, nullptr,
                                  &edgePipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create cube edge pipeline\n";
        return false;
    }

    return true;
}

Mat4 makeModelMatrix(const Cube& cube) {
    const float phase = static_cast<float>(
        animationTime * cube.animationMultiplier + cube.phaseOffset);

    Vec3 animatedOffset{};
    Vec3 animatedRotationDegrees{};

    if (cube.animate) {
        animatedOffset = {
            trajectoryRadius * std::cos(phase),
            trajectoryVerticalAmplitude *
                std::sin(trajectoryVerticalFrequency * phase + cube.phaseOffset * 0.5f),
            trajectoryDepthAmplitude *
                std::sin(trajectoryDepthFrequency * phase + cube.phaseOffset),
        };

        animatedRotationDegrees = {
            animatedRotationSpeedDegrees.x * static_cast<float>(animationTime),
            animatedRotationSpeedDegrees.y * static_cast<float>(animationTime),
            animatedRotationSpeedDegrees.z * static_cast<float>(animationTime),
        };
    }

    const Vec3 finalPosition = lab_math::add(cube.position, animatedOffset);
    const Vec3 finalRotationDegrees = lab_math::add(cube.rotationDegrees,
                                                    animatedRotationDegrees);

    const Mat4 S = lab_math::scaleMatrix(cube.scale);
    const Mat4 Rx = lab_math::rotateX(lab_math::radians(finalRotationDegrees.x));
    const Mat4 Ry = lab_math::rotateY(lab_math::radians(finalRotationDegrees.y));
    const Mat4 Rz = lab_math::rotateZ(lab_math::radians(finalRotationDegrees.z));
    const Mat4 T = lab_math::translate(finalPosition);

    const Mat4 RxS = lab_math::multiply(Rx, S);
    const Mat4 RyRxS = lab_math::multiply(Ry, RxS);
    const Mat4 RzRyRxS = lab_math::multiply(Rz, RyRxS);

    return lab_math::multiply(T, RzRyRxS);
}

Mat4 makeProjectionMatrix() {
    const float width = static_cast<float>(context.swapchain_extent.width);
    const float height = static_cast<float>(context.swapchain_extent.height);
    const float aspect = width / height;

    if (projectionType == ProjectionType::Orthographic) {
        const float halfHeight = orthographicHeight * 0.5f;
        const float halfWidth = halfHeight * aspect;

        return lab_math::orthographic(-halfWidth, halfWidth,
                                      -halfHeight, halfHeight,
                                      nearPlane, farPlane);
    }

    return lab_math::perspective(lab_math::radians(perspectiveFovDegrees),
                                 aspect, nearPlane, farPlane);
}

void drawProjectionUI() {
    if (!ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    int projection = static_cast<int>(projectionType);
    const char* projectionItems[] = {"Perspective", "Orthographic"};

    if (ImGui::Combo("Projection type", &projection, projectionItems, 2)) {
        projectionType = static_cast<ProjectionType>(projection);
    }

    if (projectionType == ProjectionType::Perspective) {
        ImGui::SliderFloat("FOV Y", &perspectiveFovDegrees, 25.0f, 100.0f, "%.1f deg");
    } else {
        ImGui::SliderFloat("Ortho area height", &orthographicHeight, 4.0f, 15.0f, "%.2f");
    }

    if (ImGui::SliderFloat("Near", &nearPlane, 0.01f, 10.0f, "%.2f")) {
        nearPlane = std::min(nearPlane, farPlane - 0.01f);
        nearPlane = std::max(nearPlane, 0.01f);
    }

    if (ImGui::SliderFloat("Far", &farPlane, 1.0f, 50.0f, "%.2f")) {
        farPlane = std::max(farPlane, nearPlane + 0.01f);
    }
}

void drawAnimationUI() {
    if (!ImGui::CollapsingHeader("Animation & Trajectory", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (ImGui::Button(animationPlaying ? "Pause" : "Play")) {
        animationPlaying = !animationPlaying;
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset time")) {
        resetAnimationRequested = true;
    }

    ImGui::SliderFloat("Speed", &animationSpeed, 0.0f, 3.0f, "%.2fx");
    ImGui::SliderFloat("Trajectory radius", &trajectoryRadius, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Y amplitude", &trajectoryVerticalAmplitude, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Z amplitude", &trajectoryDepthAmplitude, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Y frequency", &trajectoryVerticalFrequency, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Z frequency", &trajectoryDepthFrequency, 0.1f, 3.0f, "%.2f");
    ImGui::DragFloat3("Rotation speed", &animatedRotationSpeedDegrees.x,
                      1.0f, -180.0f, 180.0f, "%.1f deg/s");
}

void drawCubeUI(Cube& cube) {
    if (!ImGui::TreeNode(cube.name)) {
        return;
    }

    ImGui::Checkbox("Animate", &cube.animate);
    ImGui::DragFloat("Speed multiplier", &cube.animationMultiplier,
                     0.01f, 0.0f, 3.0f, "%.2f");
    ImGui::DragFloat("Phase offset", &cube.phaseOffset,
                     0.01f, -lab_math::PI, lab_math::PI, "%.2f rad");

    ImGui::Separator();

    ImGui::DragFloat3("Position", &cube.position.x, 0.01f, -10.0f, 10.0f, "%.2f");
    ImGui::DragFloat3("Rotation", &cube.rotationDegrees.x,
                      1.0f, -360.0f, 360.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", &cube.scale.x, 0.01f, 0.1f, 4.0f, "%.2f");
    ImGui::ColorEdit3("Cube color", cube.color);

    ImGui::TreePop();
}

void drawUI() {
    ImGui::Begin("Vulkan Lab 1 - Cubes");

    drawProjectionUI();
    drawAnimationUI();

    if (ImGui::CollapsingHeader("Cubes", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show edges", &showEdges);

        for (Cube& cube : cubes) {
            drawCubeUI(cube);
        }
    }

    ImGui::End();
}

void updateGlobalUniform() {
    GlobalUniforms global{};

    const Mat4 projection = makeProjectionMatrix();
    lab_math::toColumnMajor(projection, global.projection);

    std::memcpy(globalUniformBuffer.mapped, &global, sizeof(global));
}

void updateObjectUniform(Cube& cube) {
    ObjectUniforms object{};

    cube.modelMatrix = makeModelMatrix(cube);
    lab_math::toColumnMajor(cube.modelMatrix, object.model);

    object.baseColor[0] = cube.color[0];
    object.baseColor[1] = cube.color[1];
    object.baseColor[2] = cube.color[2];
    object.baseColor[3] = 1.0f;

    std::memcpy(cube.uniformBuffer.mapped, &object, sizeof(object));
}

void drawCube(const graphics::internal::FrameData& fd, const Cube& cube) {
    const VkDeviceSize vertexBufferOffset = 0;

    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                           &cube.mesh.vertexBuffer.buffer,
                           &vertexBufferOffset);

    vkCmdBindIndexBuffer(fd.command_buffer, cube.mesh.indexBuffer.buffer,
                         0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &globalDescriptorSet,
                            0, nullptr);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 1, 1, &cube.descriptorSet,
                            0, nullptr);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      fillPipeline);

    vkCmdDrawIndexed(fd.command_buffer, cube.mesh.triangleIndexCount,
                     1, 0, 0, 0);

    if (showEdges) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          edgePipeline);

        vkCmdDrawIndexed(fd.command_buffer, cube.mesh.edgeIndexCount,
                         1, cube.mesh.edgeFirstIndex, 0, 0);
    }
}

} // namespace

bool initialize() {
    cubes[0].name = "Cube 1";
    cubes[0].position = {-3.0f, 0.0f, 6.0f};
    cubes[0].color[0] = 1.0f;
    cubes[0].color[1] = 1.0f;
    cubes[0].color[2] = 1.0f;
    cubes[0].phaseOffset = 0.0f;

    cubes[1].name = "Cube 2";
    cubes[1].position = {0.0f, 0.0f, 6.5f};
    cubes[1].color[0] = 1.0f;
    cubes[1].color[1] = 0.80f;
    cubes[1].color[2] = 0.70f;
    cubes[1].phaseOffset = 2.0f * lab_math::PI / 3.0f;

    cubes[2].name = "Cube 3";
    cubes[2].position = {3.0f, 0.0f, 7.0f};
    cubes[2].color[0] = 0.70f;
    cubes[2].color[1] = 0.85f;
    cubes[2].color[2] = 1.0f;
    cubes[2].phaseOffset = 4.0f * lab_math::PI / 3.0f;

    for (Cube& cube : cubes) {
        if (!createMesh(cube)) {
            std::cerr << "Failed to create mesh for " << cube.name << '\n';
            return false;
        }
    }

    if (!createDescriptorLayouts()) {
        return false;
    }

    if (!createDescriptorPoolAndSets()) {
        return false;
    }

    if (!createPipelines()) {
        return false;
    }

    previousWallTime = 0.0;
    animationTime = 0.0;
    animationPlaying = true;
    timeInitialized = false;

    return true;
}

void shutdown() {
    vkQueueWaitIdle(context.graphics_queue);

    for (Cube& cube : cubes) {
        vmaDestroyBuffer(context.allocator, cube.uniformBuffer.buffer,
                         cube.uniformBuffer.allocation);
        vmaDestroyBuffer(context.allocator, cube.mesh.vertexBuffer.buffer,
                         cube.mesh.vertexBuffer.allocation);
        vmaDestroyBuffer(context.allocator, cube.mesh.indexBuffer.buffer,
                         cube.mesh.indexBuffer.allocation);
    }

    vmaDestroyBuffer(context.allocator, globalUniformBuffer.buffer,
                     globalUniformBuffer.allocation);

    vkDestroyPipeline(context.device, fillPipeline, nullptr);
    vkDestroyPipeline(context.device, edgePipeline, nullptr);

    vkDestroyShaderModule(context.device, cubeVertexShader, nullptr);
    vkDestroyShaderModule(context.device, cubeFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, edgeVertexShader, nullptr);
    vkDestroyShaderModule(context.device, edgeFragmentShader, nullptr);

    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, globalDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, objectDescriptorSetLayout, nullptr);
}

void update(double time) {
    if (!timeInitialized) {
        previousWallTime = time;
        timeInitialized = true;
    }

    const double delta = time - previousWallTime;
    previousWallTime = time;

    drawUI();

    if (resetAnimationRequested) {
        animationTime = 0.0;
        resetAnimationRequested = false;
    } else if (animationPlaying) {
        animationTime += delta * animationSpeed;
    }
}

void render(const graphics::internal::FrameData& fd) {
    updateGlobalUniform();
    for (Cube& cube : cubes) {
        updateObjectUniform(cube);
    }

    vkResetCommandBuffer(fd.command_buffer, 0);

    const VkCommandBufferBeginInfo commandBufferBegin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    if (vkBeginCommandBuffer(fd.command_buffer, &commandBufferBegin) != VK_SUCCESS) {
        std::cerr << "Failed to begin application command buffer\n";
        return;
    }

    const VkClearValue clearValues[] = {
        {
            .color = {
                .float32 = {0.035f, 0.04f, 0.055f, 1.0f},
            },
        },
        {
            .depthStencil = {1.0f, 0},
        },
    };

    const VkRenderPassBeginInfo renderPassBegin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent, },
        .clearValueCount = 2,
        .pClearValues = clearValues,
    };

    vkCmdBeginRenderPass(fd.command_buffer, &renderPassBegin,
                         VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = { .extent = context.swapchain_extent, };

    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    for (const Cube& cube : cubes) {
        drawCube(fd, cube);
    }

    vkCmdEndRenderPass(fd.command_buffer);

    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to end application command buffer\n";
    }
}

} // namespace application