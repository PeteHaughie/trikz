#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform float offsetX;
uniform float offsetY;
uniform float centerX;
uniform float centerY;
uniform float zoomAmt;
uniform float speed;
uniform float rotation;
uniform int paletteMode;
uniform vec3 paletteOffset;
uniform vec3 paletteAmp;
uniform vec3 paletteFreq;
uniform vec3 palettePhase;
uniform bool cycleColors;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y


float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 rotate2D(vec2 st, float rot) {
    rot = map(rot, 0.0, 360.0, 0.0, 2.0);
    float angle = rot * PI;
    // angle = PI * u_time * 2.0; // animate rotation
    st -= vec2(0.5 * aspectRatio, 0.5);
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st += vec2(0.5 * aspectRatio, 0.5);
    return st;
}

float offset(vec2 st) {
    return distance(st, vec2(0.5)) * 0.25;
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
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

vec3 pal(float t) {
    vec3 a = paletteOffset;
    vec3 b = paletteAmp;
    vec3 c = paletteFreq;
    vec3 d = palettePhase;

    t = abs(t);

    vec3 color = a + b * cos(6.28318 * (c * t + d));

    // convert to rgb if palette is in hsv or oklab mode
    // 1 = hsv, 2 = oklab, 3 = rgb
    if (paletteMode == 1) {
        color = hsv2rgb(color);
    } else if (paletteMode == 2) {
        color.g = color.g * -.509 + .276;
        color.b = color.b * -.509 + .198;
        color = linear_srgb_from_oklab(color);
        color = srgb_from_linear_srgb(color);
    } 

    return color;
}

// from http://nuclear.mutantstargoat.com/articles/sdr_fract/
float julia(vec2 st, float blendy) {
    
    float zoom = map(zoomAmt, 0.0, 100.0, 2.0, 0.5);
    vec2 z;
    float speedy = map(speed, 0.0, 100.0, 0.0, 1.0);
    float s = mix(speedy * 0.05, speedy * 0.125, speedy);
    float _offsetX = map(offsetX, 0.0, 100.0, 0.0, 0.5);
    float _offsetY = map(offsetY, 0.0, 100.0, 0.0, 1.0);
    vec2 c = vec2(sin(time * TAU) * s + _offsetX, cos(time * TAU) * s + _offsetY);

    st = rotate2D(st, rotation);
    st = (st - vec2(0.5 * aspectRatio, 0.5)) * zoom;

    z.x = st.x + map(centerX, -100.0, 100.0, 1.0, -1.0);
    z.y = st.y + map(centerY, -100.0, 100.0, 1.0, -1.0);

    int iter;
    for (int i=0; i<100; i++) {
        iter = i;
        float x = (z.x * z.x - z.y * z.y) + c.x;
        float y = (z.y * z.x + z.x * z.y) + c.y;

        if((x * x + y * y) > 4.0) break;
        z.x = x;
        z.y = y;
    }

    return float(iter) / 100.0;
}

void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution.y;	

    float blend = periodicFunction(time - offset(st));

    float d = julia(st, blend);
    if (cycleColors) {
        color.rgb = pal(d + time);
    } else {
        color.rgb = pal(d);
    }

    fragColor = color;
}
