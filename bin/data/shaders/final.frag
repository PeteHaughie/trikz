#version 300 es
precision highp float;
precision highp int;

uniform sampler2D postTex;
uniform sampler2D watermark;
uniform vec2 resolution;
uniform float intensity;
uniform float saturation;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718

float map(float value, float inMin, float inMax, float outMin, float outMax) {
      return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec3 brightnessContrast(vec3 color) {
    float bright = map(intensity, -100.0, 100.0, -0.4, 0.4);
    float cont = 1.0;
    if ( intensity < 0.0) {
        cont = map(intensity, -100.0, 0.0, 0.5, 1.0);
    } else {
        cont = map(intensity, 0.0, 100.0, 1.0, 1.5);
    }

    color = (color - 0.5) * cont + 0.5 + bright;
    return color;
}

vec3 saturate(vec3 color) {
    float sat = map(saturation, -100.0, 100.0, -1.0, 1.0);
    float avg = (color.r + color.g + color.b) / 3.0;
    color -= (avg - color) * sat;
    return color;
}

vec4 addWatermark(vec4 color) {
    vec2 st = gl_FragCoord.xy / vec2(128, 32); // size of watermark image
    st.y = 1.0 - st.y;
    vec2 ipos = resolution / vec2(128, 32);
    vec2 offset = vec2(st.x - ipos.x + 1.05, st.y);
    vec4 text = texture(watermark, offset);
    vec4 shadow = 1.0 - texture(watermark, offset + vec2(-0.01, -0.01));
    color = color * shadow;
    color = max(color, text);
    return color;
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    uv.y = 1.0 - uv.y;

    vec4 color = texture(postTex, uv);
    color.a = 1.0;

    // brightness/contrast/saturation
    color.rgb = brightnessContrast(color.rgb);
    color.rgb = saturate(color.rgb);

    // watermark
    color = addWatermark(color);

    fragColor = vec4(color.rgb, 1.0);
}
