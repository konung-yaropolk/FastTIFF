// composite.vert.glsl - the 2D composite's fullscreen quad (glow backend).
out vec2 v_uv;
void main() {
    const vec2 POS[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0));
    const vec2 UV[6] = vec2[6](
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
    gl_Position = vec4(POS[gl_VertexID], 0.0, 1.0);
    v_uv = UV[gl_VertexID];
}
