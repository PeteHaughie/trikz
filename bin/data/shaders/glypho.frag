#version 300 es
precision highp float;
precision highp int;

uniform sampler2D textTex;
uniform vec2 resolution;
uniform float time;
uniform float seed;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718


void main() {
    vec2 st = gl_FragCoord.xy / resolution;
	st.y = 1.0 - st.y;

    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    color = texture(textTex, st);
    color.rgb = color.rgb * color.a;

	fragColor = color;
}
