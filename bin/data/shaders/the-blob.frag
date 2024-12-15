#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform float scale;
uniform float cutoff;
uniform float loopScale;
uniform float loopAmp;
uniform int loopOffset;
uniform vec3 color1;
uniform vec3 color2;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

// PCG PRNG from https://github.com/riccardoscalco/glsl-pcg-prng, MIT license
uvec3 pcg(uvec3 v) {
	v = v * uint(1664525) + uint(1013904223);

	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;

	v ^= v >> uint(16);

	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;

	return v;
}

vec3 prng (vec3 p) {
	return vec3(pcg(uvec3(p))) / float(uint(0xffffffff));
}
// end PCG PRNG

float circles(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return dist * freq;
}

float rings(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return cos(dist * PI * freq);
}

float diamonds(vec2 st, float freq) {
    st = gl_FragCoord.xy / resolution.y;
    st -= vec2(0.5 * aspectRatio, 0.5);
    st *= freq;
    return (cos(st.x * PI) + cos(st.y * PI));
}

float shape(vec2 st, int sides, float blend) {
    // Remap st to -1 to 1
    st.y = st.y * 2.0 - 1.0;
    st.x = st.x * 2.0 - aspectRatio;

    // Angle and radius
    float a = atan(st.x, st.y) + PI;
    float r = TAU / float(sides);

    // Shaping function
    float d = cos(floor(0.5 + a / r) * r - a) * length(st);

    d *= blend;
    return d;
}

// copy of noise generation functions with no seed or periodic function (for looping)

float f(vec2 st) {
    return prng(vec3(floor(st), seed)).x;
}

// from https://observablehq.com/@riccardoscalco/glsl-bicubic-interpolation
float bicubic(vec2 p, float freq) {
    float x = p.x;
    float y = p.y;
    float x1 = floor(x);
    float y1 = floor(y);
    float x2 = x1 + 1.;
    float y2 = y1 + 1.;
    float f11 = f(vec2(x1, y1));
    float f12 = f(vec2(x1, y2));
    float f21 = f(vec2(x2, y1));
    float f22 = f(vec2(x2, y2));
    float f11x = (f(vec2(x1 + 1., y1)) - f(vec2(x1 - 1., y1))) / 2.;
    float f12x = (f(vec2(x1 + 1., y2)) - f(vec2(x1 - 1., y2))) / 2.;
    float f21x = (f(vec2(x2 + 1., y1)) - f(vec2(x2 - 1., y1))) / 2.;
    float f22x = (f(vec2(x2 + 1., y2)) - f(vec2(x2 - 1., y2))) / 2.;
    float f11y = (f(vec2(x1, y1 + 1.)) - f(vec2(x1, y1 - 1.))) / 2.;
    float f12y = (f(vec2(x1, y2 + 1.)) - f(vec2(x1, y2 - 1.))) / 2.;
    float f21y = (f(vec2(x2, y1 + 1.)) - f(vec2(x2, y1 - 1.))) / 2.;
    float f22y = (f(vec2(x2, y2 + 1.)) - f(vec2(x2, y2 - 1.))) / 2.;
    float f11xy = (f(vec2(x1 + 1., y1 + 1.)) - f(vec2(x1 + 1., y1 - 1.)) - f(vec2(x1 - 1., y1 + 1.)) + f(vec2(x1 - 1., y1 - 1.))) / 4.;
    float f12xy = (f(vec2(x1 + 1., y2 + 1.)) - f(vec2(x1 + 1., y2 - 1.)) - f(vec2(x1 - 1., y2 + 1.)) + f(vec2(x1 - 1., y2 - 1.))) / 4.;
    float f21xy = (f(vec2(x2 + 1., y1 + 1.)) - f(vec2(x2 + 1., y1 - 1.)) - f(vec2(x2 - 1., y1 + 1.)) + f(vec2(x2 - 1., y1 - 1.))) / 4.;
    float f22xy = (f(vec2(x2 + 1., y2 + 1.)) - f(vec2(x2 + 1., y2 - 1.)) - f(vec2(x2 - 1., y2 + 1.)) + f(vec2(x2 - 1., y2 - 1.))) / 4.;
    mat4 Q = mat4(f11, f21, f11x, f21x, f12, f22, f12x, f22x, f11y, f21y, f11xy, f21xy, f12y, f22y, f12xy, f22xy);
    mat4 S = mat4(1., 0., 0., 0., 0., 0., 1., 0., -3., 3., -2., -1., 2., -2., 1., 1.);
    mat4 T = mat4(1., 0., -3., 2., 0., 0., 3., -2., 0., 1., -2., 1., 0., 0., -1., 1.);
    mat4 A = T * Q * S;
    float t = fract(p.x);
    float u = fract(p.y);
    vec4 tv = vec4(1., t, t * t, t * t * t);
    vec4 uv = vec4(1., u, u * u, u * u * u);
    return dot(tv * A, uv);
}

float noise(vec2 st, float freq, int interp) {
    st -= vec2(0.5 * aspectRatio, 0.5);
    st *= freq;
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    i += floor(seed + 4.0);
    float a = prng(vec3(i, vec2(0.0))).x;
    float b = prng(vec3(i + vec2(1.0, 0.0), 0.0)).x;
    float c = prng(vec3(i + vec2(0.0, 1.0), 0.0)).x;
    float d = prng(vec3(i + vec2(1.0, 1.0), 0.0)).x;

    vec2 u = f;
    if (interp == 0) {
        // Constant
        return a;
    } else if (interp == 1) {
        // Linear
        u = f;
    } else if (interp == 2) {
        // Cosine
        u = smoothstep(0.1, 0.9, f);
    } else if (interp == 3) {
        // Bicubic
        //return bicubic(st, freq);
        return bicubic(gl_FragCoord.xy/resolution.y * freq, freq);
    } else {
        // Cosine fallback
        u = smoothstep(0.1, 0.9, f);
    }

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    //func = tan(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

float offset(vec2 st, float freq) {
    if (loopOffset == 10) {
        // circle
        return circles(st, freq);
    } else if (loopOffset == 20) {
        // triangle
        return shape(st, 3, freq * 0.5);
    } else if (loopOffset == 30) {
        // diamond
        return (abs(st.x - 0.5 * aspectRatio) + abs(st.y - 0.5)) * freq * 0.5;
    } else if (loopOffset == 40) {
        // square
        return shape(st, 4, freq * 0.5);
    } else if (loopOffset == 50) {
        // pentagon
        return shape(st, 5, freq * 0.5);
    } else if (loopOffset == 60) {
        // hexagon
        return shape(st, 6, freq * 0.5);
    } else if (loopOffset == 70) {
        // heptagon
        return shape(st, 7, freq * 0.5);
    } else if (loopOffset == 80) {
        // octagon
        return shape(st, 8, freq * 0.5);
    } else if (loopOffset == 90) {
        // nonagon
        return shape(st, 9, freq * 0.5);
    } else if (loopOffset == 100) {
        // decagon
        return shape(st, 10, freq * 0.5);
    } else if (loopOffset == 110) {
        // hendecagon
        return shape(st, 11, freq * 0.5);
    } else if (loopOffset == 120) {
        // dodecagon
        return shape(st, 12, freq * 0.5);
    } else if (loopOffset == 200) {
        // horizontal scan
        return st.x * freq * 0.5;
    } else if (loopOffset == 210) {
        // vertical scan
        return st.y * freq * 0.5;
    } else if (loopOffset == 300) {
        // constant
        return 1.0 - noise(st, freq, 0);
    } else if (loopOffset == 310) {
        // linear
        return 1.0 - noise(st, freq, 1);
    } else if (loopOffset == 320) {
        // cosine
        return 1.0 - noise(st, freq, 2);
    } else if (loopOffset == 330) {
        // bicubic
        return 1.0 - noise(st, freq, 3);
    } else if (loopOffset == 400) {
        // rings
        return 1.0 - rings(st, freq);
    } else if (loopOffset == 410) {
        // sine
        return 1.0 - diamonds(st, freq);
    }
}

// from https://www.iquilezles.org/www/articles/smoothvoronoi/smoothvoronoi.htm
// see also https://www.shadertoy.com/view/XscyRX

float smoothVoronoi(vec2 st, float freq) {
    st *= freq;
    vec2 ip = floor(st);
    st -= ip;
    float d = 1.0, res = 0.0;

    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            vec2 b = vec2(i, j);
            vec2 v = b - st + cos(prng(vec3(ip + b, 0.0)).xy * TAU) * 0.5;
            d = max(dot(v, v), 1e-4);
            res += 1.0 / pow(d, 8.0);
        }
    }
    return pow(1.0 / res, 1.0 / 16.0);
}

vec3 applyColors(float d) {
    if (d <= 0.5) {
        return color1;
    } else {
        return color2;
    }
}

void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution.y;

    float lf = map(loopScale, 1.0, 100.0, 6.0, 1.0);
    float o = offset(st, lf);
    float amp = map(abs(loopAmp), 0.0, 100.0, 0.0, 1.0);
    float blend = 1.0;
    if (loopAmp > 0.0) {
        blend = periodicFunction(time - o) * amp;
    } else {
        blend = periodicFunction(time + o) * amp;
    }

    float freq = map(scale, 1.0, 100.0, 20.0, 1.0);

    float c = map(cutoff, 0.0, 100.0, 0.0, 1.0);

    st.x += blend * 0.025 * c;
    st.y += periodicFunction(time - o + 0.5) * amp * 0.025 * c;
    float d = smoothVoronoi(st, freq);

    c = mix(
            mix(
                0.125,  // maintain a minimum dot size
                o * c,  // cutoff slider value, with offset variation
                c
            ),
            mix(
                0.125,  // maintain a minimum dot size
                (c + 0.25) * blend,
                amp
            ),
            amp
    );

    d = step(c, d);

    color.rgb = applyColors(d);

    fragColor = color;
}
