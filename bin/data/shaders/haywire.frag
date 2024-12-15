#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform int noiseType;
uniform float noiseAmp;
uniform float scale;
uniform float skewAmt;
uniform float rotation;
uniform vec3 color1;
uniform vec3 color2;
out vec4 fragColor;

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


float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(vec2 st, float freq) {
    st *= freq;
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    i += floor(seed);
    float a = prng(vec3(i, 0.0)).x;
    float b = prng(vec3(i + vec2(1.0, 0.0), 0.0)).x;
    float c = prng(vec3(i + vec2(0.0, 1.0), 0.0)).x;
    float d = prng(vec3(i + vec2(1.0, 1.0), 0.0)).x;

    vec2 u = smoothstep(0.1, 0.9, f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

vec3 applyColors(float d) {
    if (d <= 0.5) {
        return color1;
    } else {
        return color2;
    }
}

// periodic function for looping
float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

vec2 rotate2D(vec2 st, float rot) {
    rot = map(rot, 0.0, 360.0, 0.0, 2.0);
    float angle = rot * PI;
    // angle = PI * u_time * 2.0; // animate rotation
    st -= 0.5;
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st += 0.5;
    return st;
}

vec2 skew(vec2 st) {
    st = rotate2D(st, rotation);
    float x = map(skewAmt, -100.0, 100.0, -10.0, 10.0);
    vec2 uv = st;
    uv.x += (st.y * x);
    return uv;
}

float chess(vec2 st, float freq, float blend) {
    st = skew(st);
    st.x -= time / freq;
    st *= freq;
    st = fract(st);
    st -= 0.5;
    float d = step(st.x * st.y, 0.0);
    return d;
}

// from https://thebookofshaders.com/edit.php#09/lines-wave.frag
vec2 wave(vec2 st, float freq, float amp, float blend) {
    st.y += time;
    amp = amp * blend;
    st.y += cos(st.x * freq) * amp;
    return st;
}

// from https://thebookofshaders.com/edit.php#09/lines-wave.frag
float line(vec2 st, float width) {
    return step(width, 1.0 - smoothstep(0.0, 1.0, abs(sin(st.y * PI))));
}

// from https://thebookofshaders.com/edit.php#09/lines-wave.frag
float wavy(vec2 st, float freq, float blend) {
    st *= freq;
    st = skew(st);
    st = wave(st, PI, 0.5, blend);
    return line(st, 0.5);
}

// from https://www.shadertoy.com/view/3t2BWW
float zigzag(vec2 st, float freq, float blend) {
    //freq.y *= 0.02 * blend;
    vec3 color = vec3(0.0);
    //st.y += sin(st.x * 5.0 + time * 3.3) * 0.05 * cos(st.x) * 0.3;
    // move UV over time
    //st.x += time * 0.09;
    //st.y += time * 0.04;
    st = skew(st);

    // Make a grid
    vec2 gv = fract(st * freq);

    // Remap GV from [0..1] to [-1..1]
    gv.x = gv.x * 2.0 - 1.0;
    gv.x = abs(gv.x);

    // Distort uv.y
    gv.y += gv.x * 0.4 * blend;
    gv.y -= 0.1;

    // Define the colors of the bands
    vec3 col1 = vec3(0.3882, 0.102, 0.502);
    vec3 col2 = vec3(0.6588, 0.0784, 0.5137);
    vec3 col3 = vec3(1.0, 0.2, 0.7);

    col1 = vec3(0.15);
    col2 = vec3(0.35);
    col3 = vec3(0.75);
    
    if (gv.y <= 0.19)
        color += col1 * (1.0 - smoothstep(0.0, 0.19, gv.y) * 0.5);

    if (gv.y > 0.19 && gv.y <= 0.53)
        color += col2 * (1.0 - smoothstep(0.34, 0.53, gv.y) * 0.5);

    if (gv.y > 0.53 && gv.y <= 0.86)
        color += col3 * (1.0 - smoothstep(0.67, 0.86, gv.y) * 0.5);

    if (gv.y > 0.86)
        color += col1 * (1.0 - smoothstep(1.0, 1.19, gv.y) * 0.5);

    if (gv.y > 1.19 && gv.y <= 1.53)
        color = col2 * (1.0 - smoothstep(1.34, 1.53, gv.y) * 0.5);

    float avg = (color.r + color.g + color.b) / 3.0;
    return avg;
}

float stripes(vec2 st, float freq, float blend) {
    st = skew(st);
    st.x -= time / freq;
    st *= freq;
    float d = step(0.5, fract(st.x));// * 0.5;
    return d;
}

float concentric(vec2 st, float freq, float blend) {
    float rot = map(rotation, 0.0, 360.0, 0.0, 2.0);
    float angle = rot * PI;
    st -= vec2(0.5 * aspectRatio, 0.5);
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st += vec2(0.5 * aspectRatio, 0.5);
    float x = map(skewAmt, -100.0, 100.0, -10.0, 10.0);
    st.x += (st.y * x);
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return cos((dist - (time / freq)) * freq * TAU);
}

float circle(vec2 st, float radius) {
    st = vec2(0.5) - st;
    radius *= 0.75;
    return 1.0 - smoothstep(radius - (radius * 0.05), radius + (radius * 0.05), dot(st, st) * PI);
}

float circles(vec2 st, float freq, float blend) {
    st *= freq;
    st = skew(st);
    return circle(fract(st), 0.5 * blend);
}

float shape(vec2 st, int sides, float blend) {
    // Remap st to -1 to 1
    st = st * 2.0 - 1.0;

    // Angle and radius
    float a = atan(st.x, st.y) + PI;
    float r = TAU / float(sides);
    
    // Shaping function
    float d = cos(floor(0.5 + a / r) * r - a) * length(st);

    //d *= blend;
    d += blend * 0.25;
    return d;
}

float squares(vec2 st, float freq, float blend) {
    st *= freq;
    st = skew(st);
    return shape(fract(st), 4, blend);
}

float hexagons(vec2 st, float freq, float blend) {
    st *= freq;
    st = skew(st);
    return shape(fract(st), 6, blend);
}

float hearts(vec2 st, float freq, float blend) {
    blend *= 0.5;
    st *= freq;
    st = skew(st);
    st = fract(st);

    st -= vec2(0.5, 0.65);
    float r = length(st) * 8.0;
    st = normalize(st);
    return r - 
        ((st.y * pow(abs(st.x), 0.75) - 0.25) / 
        (st.y + 1.5) - (2.0) * st.y + 1.26 + blend);
}

float grid(vec2 st, float freq, float res, float blend) {
    st = skew(st);
    st.y += time / freq / res;
    st *= freq * 4.0;
    vec2 grid = fract(st * res);
    return (step(res, grid.x) * step(res, grid.y));
}

float generate(vec2 st, int ntype, float freq, float seed, float blend) {
    float d = 0.0;
    if (ntype == 0) {
        d = chess(st, freq, blend);
    } else if (ntype == 1) {
        d = circles(st, freq, blend);
    } else if (ntype == 2) {
        d = grid(st, freq, 0.1, blend);
    } else if (ntype == 3) {
        d = hearts(st, freq, blend);
    } else if (ntype == 4) {
        d = hexagons(st, freq, blend); 
    } else if (ntype == 5) {
        d = concentric(st, freq, blend);  
    } else if (ntype == 6) {
        d = squares(st, freq, blend);
    } else if (ntype == 7) {
        d = stripes(st, freq, blend);
    } else if (ntype == 8) {
        d = wavy(st, freq, blend);
    } else if (ntype == 9) {
        d = zigzag(st, freq, blend);
    }
    return d;
}

void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);

    vec2 st = gl_FragCoord.xy / resolution.y;

    //st.x = smoothstep(0.0, 1.0, st.x / st.y);


    //st = st * st.x + st * st.y + st * 0.5;
    //st = st + st.y;

/*
(1 - t) * a + t * b
  const Vector2D texture_coords =
    triangle_screen.v1.texture_coords * u +
    triangle_screen.v2.texture_coords * v +
    triangle_screen.v3.texture_coords * w;
*/


    float blendy = periodicFunction(time);
    float freq = map(scale, 1.0, 100.0, 100.0, 1.0);
    float d = generate(st, noiseType, freq, 1.0 / seed, blendy);
    //d = step(0.5, d) * 0.875;

    //color.rgb = pal(d);
    color.rgb = applyColors(d);

    fragColor = color;
}
