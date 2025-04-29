#version 450

// 输入：匹配你的 Vertex 结构体 (pos + color)
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;

// 输出：传递颜色到片段着色器
layout(location = 0) out vec3 fragColor;

void main() {
    // 直接输出顶点位置（假设已经是裁剪空间坐标）
    gl_Position = vec4(inPos, 0.0, 1.0);
    
    // 传递颜色
    fragColor = inColor;
}