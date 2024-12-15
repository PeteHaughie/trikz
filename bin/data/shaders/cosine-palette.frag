#version 300 es
precision highp float;
precision highp int;

uniform sampler2D mixerTex;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform float offsetR;
uniform float offsetG;
uniform float offsetB;
uniform float phaseR;
uniform float phaseG;
uniform float phaseB;
uniform float ampR;
uniform float ampG;
uniform float ampB;
uniform float freq;
uniform bool cycleColors;
uniform int colorMode;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718

vec3 pal(float t) {
    vec3 a = vec3(offsetR, offsetG, offsetB) * 0.01;
    vec3 b = vec3(ampR, ampG, ampB) * 0.01;
    vec3 c = vec3(freq);
    vec3 d = vec3(phaseR, phaseG, phaseB) * 0.01;

    return a + b * cos(6.28318 * (c * t + d));
}

float luminance(vec3 color) {
    float l = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    return l;
}

// oklab palette stuff
// oklab stuff
//https://bottosson.github.io/posts/colorwrong/#what-can-we-do%3F
float srgb_from_linear_srgb(float x) {
    if (x >= 0.0031308) {
        return 1.055 * pow(x, 1.0/2.4) - 0.055;
    } else {
        return 12.92 * x;
    }
}

float linear_srgb_from_srgb(float x) {
    if (x >= 0.04045) {
        return pow((x + 0.055)/(1.0 + 0.055), 2.4);
    } else {
        return x / 12.92;
    }
}

vec3 srgb_from_linear_srgb(vec3 c) {
    return vec3(
        srgb_from_linear_srgb(c.x),
        srgb_from_linear_srgb(c.y),
        srgb_from_linear_srgb(c.z)
    );
}

vec3 linear_srgb_from_srgb(vec3 c) {
    return vec3(
        linear_srgb_from_srgb(c.x),
        linear_srgb_from_srgb(c.y),
        linear_srgb_from_srgb(c.z)
    );
}

//////////////////////////////////////////////////////////////////////
// oklab transform and inverse from
// https://bottosson.github.io/posts/oklab/

const mat3 fwdA = mat3(1.0, 1.0, 1.0,
                       0.3963377774, -0.1055613458, -0.0894841775,
                       0.2158037573, -0.0638541728, -1.2914855480);

const mat3 fwdB = mat3(4.0767245293, -1.2681437731, -0.0041119885,
                       -3.3072168827, 2.6093323231, -0.7034763098,
                       0.2307590544, -0.3411344290,  1.7068625689);

const mat3 invB = mat3(0.4121656120, 0.2118591070, 0.0883097947,
                       0.5362752080, 0.6807189584, 0.2818474174,
                       0.0514575653, 0.1074065790, 0.6302613616);

const mat3 invA = mat3(0.2104542553, 1.9779984951, 0.0259040371,
                       0.7936177850, -2.4285922050, 0.7827717662,
                       -0.0040720468, 0.4505937099, -0.8086757660);

vec3 oklab_from_linear_srgb(vec3 c) {
    vec3 lms = invB * c;

    return invA * (sign(lms)*pow(abs(lms), vec3(0.3333333333333)));
}

vec3 linear_srgb_from_oklab(vec3 c) {
    vec3 lms = fwdA * c;

    return fwdB * (lms * lms * lms);
}
// end oklab stuff

vec3 hsv2rgb(vec3 c) {
    c.x = mod(c.x, 1.0);
    c.y = clamp(c.y, 0.0, 1.0);
    c.z = clamp(c.z, 0.0, 1.0);

    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    uv.y = 1.0 - uv.y;

    vec4 color = vec4(0.0);

    color = texture(mixerTex, uv);
    float l = luminance(color.rgb) * 0.9; // prevent black and white from returning the same color
    if (cycleColors) {
        color.rgb = pal(l + time);
    } else {
        color.rgb = pal(l);
    }

    if (colorMode == 0) {
        // hsv -> rgb conversion
        color.rgb = hsv2rgb(color.rgb);
    } else if (colorMode == 1) {
        // oklab -> rgb conversion
        color.g = color.g * -.509 + .276;
        color.b = color.b * -.509 + .198;
        color.rgb = linear_srgb_from_oklab(color.rgb);
        color.rgb = srgb_from_linear_srgb(color.rgb);
    } else if (colorMode == 2) {
        // rgb, do nothing
    }

    fragColor = vec4(color.rgb, 1.0);
}
