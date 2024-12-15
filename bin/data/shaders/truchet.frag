#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform int patternIndex;
uniform float scale;
uniform float lineWidth;
uniform float softness;
uniform float speed;
uniform float rotation;
uniform vec3 color1;
uniform vec3 color2;
out vec4 fragColor;


#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y

float map(float value, float inMin, float inMax, float outMin, float outMax) {
      return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

// periodic function for looping
float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
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

// truchet modified from Art of Code
// https://www.youtube.com/watch?v=2R7h76GoIJM&list=RDCMUCcAlTqd9zID6aNX3TzwxJXg&index=10

float hash21(vec2 p) {
    // returns random number between 0 and 1
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float truchetLines(vec2 uv, float freq) {
    // uvs inside of grid cells
    uv *= freq;
    vec2 gv = fract(uv) - 0.5;

    // get unique value for each tile
    vec2 id = floor(uv);
    id.x += seed;
    id.y *= 1.0 + periodicFunction(time * speed * 0.0000001) * 0.1;
    float n = hash21(id);

    // flip about half of the tiles
    if (n < 0.5) {
        gv.x *= -1.0;
    }

    // create two lines 
    float dist = abs(abs(gv.x + gv.y) - 0.5);
    float soft = softness * 0.001;
    float mask = smoothstep(soft, -soft, dist - lineWidth * 0.005);
    return mask;
}

float truchetCurves(vec2 uv, float freq) {
    // uvs inside of grid cells
    uv *= freq;
    vec2 gv = fract(uv) - 0.5;

    // get unique value for each tile
    vec2 id = floor(uv);
    id.x += seed;
    id.y *= 1.0 + periodicFunction(time * speed * 0.0000001) * 0.1;
    float n = hash21(id);

    // flip about half of the tiles
    if (n < 0.5) {
        gv.x *= -1.0;
    }

    // can use global UVs or other factor to modify width of truchet
    vec2 UV = gl_FragCoord.xy / resolution;
    float width = lineWidth * 0.005; // * length(UV - 0.5);

    // create two curves
    vec2 cuv = gv - 0.5 * sign(gv.x + gv.y + 0.001);
    float dist = length(cuv);
    float soft = softness * 0.001; // was 0.01 - -0.01
    float mask = smoothstep(soft, -soft, abs(dist - 0.5) - width);
    
    return mask;
}

float truchet(vec2 uv, float freq, int index) {
    float d = 0.0;
    if (index == 0) {
        d = truchetCurves(uv, freq);
    } else if (index == 1) {
        d = truchetLines(uv, freq);
    }
    return d;
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.y;
    vec3 color = vec3(0.0);

    float freq = map(scale, 2.0, 100.0, 100.0, 2.0);
    uv = rotate2D(uv, rotation);
    float d = truchet(uv, freq, patternIndex);
    color = mix(color1, color2, d);

    // create red outlines for debugging purposes
    /*
    if (gv.x > 0.48 || gv.y > 0.48) {
        color = vec3(1.0, 0.0, 0.0);
    }
    */


    fragColor = vec4(color, 1.0);
}