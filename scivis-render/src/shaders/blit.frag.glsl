// blit.frag.glsl - sample the reduced render bilinearly (see `blit.vert.glsl`).
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D src;
void main() {
    frag_color = textureLod(src, v_uv, 0.0);
}
