// blit.vert.glsl - scale a reduced-size render of the 3D view up to the
// view's own size (glow backend; the WGSL twin is `blit.wgsl`): one
// full-screen triangle.
out vec2 v_uv;
void main() {
    vec2 c = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0);
    v_uv = c;
}
