#version 300 es
// bit fields, inspired by https://twitter.com/aemkei/status/1378106731386040322
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform int formula;
uniform int colorScheme;
uniform float n;
uniform int interp;
uniform float scale;
uniform float rotation;
uniform float offset;
uniform float loopAmp;
out vec4 fragColor;

const int BIT_COUNT = 8;

#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y


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

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 rotate2D(vec2 st, float rot) {
    rot = map(rot, 0.0, 360.0, 0.0, 1.0);
    float angle = rot * TAU;
    // angle = PI * u_time * 2.0; // animate rotation
    st -= vec2(0.5 * aspectRatio, 0.5);
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st += vec2(0.5 * aspectRatio, 0.5);
    return st;
}

// periodic function for looping
float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

float constant(vec2 st, float xFreq, float yFreq, float s) {
    float x = st.x * xFreq;
    float y = st.y * yFreq;

    x += s;

    float scaledTime = periodicFunction(
            prng(vec3(floor(vec2(x + 40.0, y)), 0.0)).x - time
        ) * map(abs(loopAmp), 0.0, 100.0, 0.0, 0.333);

    return periodicFunction(prng(vec3(floor(vec2(x, y)), 0.0)).x - scaledTime);
}

float value(vec2 st, float xFreq, float yFreq, float s) {
    float scaledTime = 1.0;

    float x1y1 = constant(st, xFreq, yFreq, s);

    if (interp == 0) {
        return x1y1;
    }

    // Neighbor Distance
    float ndX = 1.0 / xFreq;
    float ndY = 1.0 / yFreq;

    float x1y2 = constant(vec2(st.x, st.y + ndY), xFreq, yFreq, s);
    float x2y1 = constant(vec2(st.x + ndX, st.y), xFreq, yFreq, s);
    float x2y2 = constant(vec2(st.x + ndX, st.y + ndY), xFreq, yFreq, s);

    vec2 uv = vec2(st.x * xFreq, st.y * yFreq);

    float a = mix(x1y1, x2y1, fract(uv.x));
    float b = mix(x1y2, x2y2, fract(uv.x));

    return mix(a, b, fract(uv.y));
}

// bitwise operations https://gist.github.com/mattatz/70b96f8c57d4ba1ad2cd
int modi(int x, int y) {
    return x - y * (x / y);
}

int or(int a, int b) {
    int result = 0;
    int n = 1;

    for(int i = 0; i < BIT_COUNT; i++) {
        if ((modi(a, 2) == 1) || (modi(b, 2) == 1)) {
            result += n;
        }
        a = a / 2;
        b = b / 2;
        n = n * 2;
        if(!(a > 0 || b > 0)) {
            break;
        }
    }
    return result;
}

int and(int a, int b) {
    int result = 0;
    int n = 1;

    for(int i = 0; i < BIT_COUNT; i++) {
        if ((modi(a, 2) == 1) && (modi(b, 2) == 1)) {
            result += n;
        }

        a = a / 2;
        b = b / 2;
        n = n * 2;

        if(!(a > 0 && b > 0)) {
            break;
        }
    }
    return result;
}

int not2(int a) {
    int result = 0;
    int n = 1;

    for(int i = 0; i < BIT_COUNT; i++) {
        if (modi(a, 2) == 0) {
            result += n;
        }
        a = a / 2;
        n = n * 2;
    }
    return result;
}

int xor(int a, int b) {
    return or(a, b) - and(a, b);
}

float or(float a, float b) {
    return float(or(int(a), int(b)));
}

float and(float a, float b) {
    return float(and(int(a), int(b)));
}

float not3(float a) {
    return float(not2(int(a)));
}

float xor(float a, float b) {
    return float(xor(int(a), int(b)));
}
// end bitwise operations

float bitValue(vec2 st, float freq, vec2 _offset, float nForColor) {
    float blendy = nForColor + periodicFunction(value(st, freq * 0.01, freq * 0.01, nForColor) * 0.1) * 100.0;

    float v = 1.0;

    if (formula == 0) {
        // alien
        v = mod(xor(_offset.x + st.x * freq, _offset.y + st.y * freq), blendy);
    } else if (formula == 1) {
        // sierpinski
        v = mod(or(_offset.x + st.x * freq, _offset.y + st.y * freq), blendy);
    } else if (formula == 2) {
        // circular
        v = mod((_offset.x + st.x * freq) * (_offset.y + st.y * freq), blendy);
    } else if (formula == 3) {
        // steps
        v = float(xor(_offset.x + st.x * freq, _offset.y + st.y * freq) < blendy);
    } else if (formula == 4) {
        // beams
        v = mod(_offset.x + st.x * freq * blendy, _offset.y + st.y * freq);
    } else if (formula == 5) {
        // perspective
        v = mod(((st.x * freq - 0.5) * 0.25), st.y * freq - 0.5);
    }

    return v > 1.0 ? 0.0 : 1.0;
}


void main() {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    vec2 st = gl_FragCoord.xy;

    st = rotate2D(st, rotation) / scale;

    float freq = map(scale, 1.0, 100.0, scale, 8.0);
    vec2 _offset = vec2(
        sin(map(offset, 0.0, 100.0, 0.0, TAU)) * scale * 10.0,
        cos(map(offset, 0.0, 100.0, 0.0, TAU)) * scale * 10.0
    ) + 1000.0;

    if (colorScheme == 0) {
        // blue
        color.b = bitValue(st, freq, _offset, n);
    } else if (colorScheme == 1) {
        // cyan
        color.gb = vec2(bitValue(st, freq, _offset, n));
    } else if (colorScheme == 2) {
        // green
        color.g = bitValue(st, freq, _offset, n);
    } else if (colorScheme == 3) {
        // magenta
        color.br = vec2(bitValue(st, freq, _offset, n));
    } else if (colorScheme == 4) {
        // red
        color.r = bitValue(st, freq, _offset, n);
    } else if (colorScheme == 5) {
        // white
        color.rgb = vec3(bitValue(st, freq, _offset, n));
    } else if (colorScheme == 6) {
        // yellow
        color.rg = vec2(bitValue(st, freq, _offset, n));
    } else if (colorScheme == 10) {
        // blue green
        color.b = bitValue(st, freq, _offset, n);
        color.g = bitValue(st, freq, _offset, n + 1.0);
    } else if (colorScheme == 11) {
        // blue red
        color.b = bitValue(st, freq, _offset, n);
        color.r = bitValue(st, freq, _offset, n + 1.0);
    } else if (colorScheme == 12) {
        // blue yellow
        color.b = bitValue(st, freq, _offset, n);
        color.rg = vec2(bitValue(st, freq, _offset, n + 1.0));
    } else if (colorScheme == 13) {
        // green magenta
        color.g = bitValue(st, freq, _offset, n);
        color.rb = vec2(bitValue(st, freq, _offset, n + 1.0));
    } else if (colorScheme == 14) {
        // green red
        color.g = bitValue(st, freq, _offset, n);
        color.r = bitValue(st, freq, _offset, n + 1.0);
    } else if (colorScheme == 15) {
        // red cyan
        color.r = bitValue(st, freq, _offset, n);
        color.bg = vec2(bitValue(st, freq, _offset, n + 1.0));
    } else if (colorScheme == 20) {
        // rgb
        color.r = bitValue(st, freq, _offset, n);
        color.g = bitValue(st, freq, _offset, n + 1.0);
        color.b = bitValue(st, freq, _offset, n + 2.0);
    }


    fragColor = color;
}
