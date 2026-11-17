// volume.vert.glsl - the volume ray-march's fullscreen quad (glow backend):
// each fragment gets its position in normalized device coordinates.
out vec2 v_ndc;
void main() {
    const vec2 P[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0));
    gl_Position = vec4(P[gl_VertexID], 0.0, 1.0);
    v_ndc = P[gl_VertexID];
}
