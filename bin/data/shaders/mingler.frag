#version 300 es
precision highp float;
precision highp int;

uniform sampler2D noiseTex;
uniform sampler2D synthTex1;
uniform sampler2D synthTex2;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform bool wrap;
uniform float mixScale;
uniform float mixAmp;
uniform int mixOffset;
uniform float mixAmt;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y


float ridge(float h, float peak) {
    h = abs(h);     // create creases
    h = peak - h;   // invert so creases are at top
    h = h * h;      // sharpen creases
    return h;
}

float map(float value, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec3 blend(vec3 color1, vec3 color2, float blend) {
    vec3 color = color1;
    float mixVal = map(mixAmt, -100.0, 100.0, 0.0, 1.0);
    vec3 middle = mix(color1, color2, blend);

    if (mixVal == 0.5) {
        color = middle;
    } else if (mixVal < 0.5) {
        color = mix(color1, middle, min(mixVal * 2.0, blend));
    } else if (mixVal > 0.5) {
        color = mix(middle, color2, max((mixVal - 0.5) * 2.0, blend));
    }

    return color;
}

float rings(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return cos(dist * PI * freq);
}

float circles(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return dist * freq;
}

float diamonds(vec2 st, float freq) {
    st = gl_FragCoord.xy / resolution.y;
    st -= vec2(0.5 * aspectRatio, 0.5);
    //st *= freq;
    return (cos(st.x * PI) + cos(st.y * PI)) * freq;
}

float shape(vec2 st, int sides, float blend) {
    // Remap st to -1 to 1
    st.y = 1.0 - st.y;  // right side up shapes
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

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// periodic function for looping
float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

// texture-based bicubic for better performance
// from http://www.java-gaming.org/index.php?topic=35123.0
vec4 cubic(float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

float textureBicubic(vec2 texCoords, float freq, float _seed, float blend) {
    texCoords.x *= aspectRatio;
    texCoords *= freq * freq / resolution * 0.5;

    vec2 texSize = vec2(512.0);
    vec2 invTexSize = 1.0 / texSize;

    texCoords = texCoords * texSize - 0.5;
    // Map to avoid image crease
    texCoords.x = mod(texCoords.x + _seed, texSize.x);

    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2(-0.5, 1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    float sample0 = texture(noiseTex, offset.xz).r;
    float sample1 = texture(noiseTex, offset.yz).r;
    float sample2 = texture(noiseTex, offset.xw).r;
    float sample3 = texture(noiseTex, offset.yw).r;

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    float temp = mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
    return periodicFunction(temp * 4.0 - blend);
}
// end texture-based bicubic

// simplex via https://github.com/ashima/webgl-noise/blob/master/src/noise2D.glsl
vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
    return mod289(((x*34.0)+1.0)*x);
}

float simplexValue(vec2 st, float freq, float s, float blend) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0

    vec2 uv = st * freq;
    st.x *= aspectRatio;
    uv.x += s;

    // First corner
    vec2 i  = floor(uv + dot(uv, C.yy) );
    vec2 x0 = uv -   i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		  + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

    // Compute final noise value at P
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;

    float v = 130.0 * dot(m, g);

    return periodicFunction(map(v, -1.0, 1.0, 0.0, 1.0) - blend);
}

// end simplex

// value noise functions c/o py-noisemaker
// https://github.com/aayars/py-noisemaker/blob/master/noisemaker/value.py
float constant(vec2 st, float freq) {
    float x = st.x * freq;
    float y = st.y * freq;
    if (wrap) {
        x = mod(x, freq);
        y = mod(y, freq);
    }
    x += seed;

    float scaledTime = periodicFunction(
            random(floor(vec2(x + 40.0, y))) - time
        ) * map(abs(mixAmp), 0.0, 100.0, 0.0, 0.333);

    return periodicFunction(random(floor(vec2(x, y))) - scaledTime);
}

float blendBicubic(float a, float b, float c, float d, float amount) {
    // http://paulbourke.net/miscellaneous/interpolation/
    float amountSqr = amount * amount;

    float a0 = d - c - a + b;
    float a1 = a - b - a0;
    float a2 = c - a;
    float a3 = b;

    return a0 * amount * amountSqr + a1 * amountSqr + a2 * amount + a3;
}

float blendLinearOrCosine(float a, float b, float amount, int interp) {
    if (interp == 1) {
        return mix(a, b, amount);
    }

    return mix(a, b, smoothstep(0.0, 1.0, amount));
}

float bicubicValue(vec2 st, float freq) {
    // Neighbor Distance
    float ndX = 1.0 / freq;
    float ndY = 1.0 / freq;

    float u0 = st.x - ndX;
    float u1 = st.x;
    float u2 = st.x + ndX;
    float u3 = st.x + ndX + ndX;

    float v0 = st.y - ndY;
    float v1 = st.y;
    float v2 = st.y + ndY;
    float v3 = st.y + ndY + ndY;

    float x0y0 = constant(vec2(u0, v0), freq);
    float x0y1 = constant(vec2(u0, v1), freq);
    float x0y2 = constant(vec2(u0, v2), freq);
    float x0y3 = constant(vec2(u0, v3), freq);

    float x1y0 = constant(vec2(u1, v0), freq);
    float x1y1 = constant(st, freq);
    float x1y2 = constant(vec2(u1, v2), freq);
    float x1y3 = constant(vec2(u1, v3), freq);

    float x2y0 = constant(vec2(u2, v0), freq);
    float x2y1 = constant(vec2(u2, v1), freq);
    float x2y2 = constant(vec2(u2, v2), freq);
    float x2y3 = constant(vec2(u2, v3), freq);

    float x3y0 = constant(vec2(u3, v0), freq);
    float x3y1 = constant(vec2(u3, v1), freq);
    float x3y2 = constant(vec2(u3, v2), freq);
    float x3y3 = constant(vec2(u3, v3), freq);

    vec2 uv = st * freq;

    float y0 = blendBicubic(x0y0, x1y0, x2y0, x3y0, fract(uv.x));
    float y1 = blendBicubic(x0y1, x1y1, x2y1, x3y1, fract(uv.x));
    float y2 = blendBicubic(x0y2, x1y2, x2y2, x3y2, fract(uv.x));
    float y3 = blendBicubic(x0y3, x1y3, x2y3, x3y3, fract(uv.x));

    return blendBicubic(y0, y1, y2, y3, fract(uv.y));
}

float value(vec2 st, float freq, int interp) {
    vec2 st2 = st - vec2(0.5 * aspectRatio, 0.5);
    float scaledTime = 1.0;
    float d = 0.0;

    if (interp == 3) {
        d = bicubicValue(st, freq);
    } else if (interp == 4) {
        // scaledTime = textureBicubic(st, freq, seed + 40.0, time) * mixAmp * 0.01;
        d = textureBicubic(st, freq, seed, scaledTime);
    } else if (interp == 10) {
        scaledTime = simplexValue(st, freq, seed + 40.0, time) * mixAmp * 0.01;
        d = simplexValue(st, freq, seed, scaledTime);
    } else {
        float x1y1 = constant(st, freq);

        if (interp == 0) {
            d = x1y1;
        } else {

            // Neighbor Distance
            float ndX = 1.0 / freq;
            float ndY = 1.0 / freq;

            float x1y2 = constant(vec2(st.x, st.y + ndY), freq);
            float x2y1 = constant(vec2(st.x + ndX, st.y), freq);
            float x2y2 = constant(vec2(st.x + ndX, st.y + ndY), freq);

            vec2 uv = st * freq;

            float a = blendLinearOrCosine(x1y1, x2y1, fract(uv.x), interp);
            float b = blendLinearOrCosine(x1y2, x2y2, fract(uv.x), interp);

            d = blendLinearOrCosine(a, b, fract(uv.y), interp);
        }
    }
    return d;
}

float offset(vec2 st, float freq) {
    float d = 0.0;
    if (mixOffset == 10) {
        // circle
        d = circles(st, freq);
    } else if (mixOffset == 20) {
        // triangle
        d = shape(st, 3, freq * 0.5);
    } else if (mixOffset == 30) {
        // diamond
        d = (abs(st.x - 0.5 * aspectRatio) + abs(st.y - 0.5)) * freq * 0.5;
    } else if (mixOffset == 40) {
        // square
        d = shape(st, 4, freq * 0.5);
    } else if (mixOffset == 50) {
        // pentagon
        d = shape(st, 5, freq * 0.5);
    } else if (mixOffset == 60) {
        // hexagon
        d = shape(st, 6, freq * 0.5);
    } else if (mixOffset == 70) {
        // heptagon
        d = shape(st, 7, freq * 0.5);
    } else if (mixOffset == 80) {
        // octagon
        d = shape(st, 8, freq * 0.5);
    } else if (mixOffset == 90) {
        // nonagon
        d = shape(st, 9, freq * 0.5);
    } else if (mixOffset == 100) {
        // decagon
        d = shape(st, 10, freq * 0.5);
    } else if (mixOffset == 110) {
        // hendecagon
        d = shape(st, 11, freq * 0.5);
    } else if (mixOffset == 120) {
        // dodecagon
        d = shape(st, 12, freq * 0.5);
    } else if (mixOffset == 200) {
        // horizontal scan
        d = st.x * freq * 0.5;
    } else if (mixOffset == 210) {
        // vertical scan
        d = st.y * freq * 0.5;
    } else if (mixOffset == 300) {
        // constant
        d = 1.0 - value(st, freq, 0);
    } else if (mixOffset == 310) {
        // linear
        d = 1.0 - value(st, freq, 1);
    } else if (mixOffset == 320) {
        // cosine
        d = 1.0 - value(st, freq, 2);
    } else if (mixOffset == 330) {
        // bicubic
        d = 1.0 - value(st, freq, 3);
    } else if (mixOffset == 340) {
        // budget
        d = 1.0 - value(st, freq, 4);
    } else if (mixOffset == 350) {
        // simplex
        d = 1.0 - value(st, freq, 10);
    } else if (mixOffset == 400) {
        // rings
        d = 1.0 - rings(st, freq);
    } else if (mixOffset == 410) {
        // sine
        d = 1.0 - diamonds(st, freq);
    }
    
    return d;
}

void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution;
    st.y = 1.0 - st.y;

    vec4 color1 = texture(synthTex1, st);
    vec4 color2 = texture(synthTex2, st);

    float freq = 1.0;
    if (mixOffset == 340) {
        // budget
        freq = map(mixScale, 1.0, 100.0, 12.0, 4.0);
    } else if (mixOffset == 350) {
        freq = map(mixScale, 1.0, 100.0, 12.0, 0.5);
    } else {
        freq = map(mixScale, 1.0, 100.0, 20.0, 2.0);
    }
    if (mixOffset >= 300 && mixOffset < 340 && wrap) {
        freq = floor(freq);  // for seamless noise
        freq *= 2.0;
    }

    st.x *= aspectRatio;
    float t = 1.0;
    if (mixAmp < 1.0) {
        t = time + offset(st, freq);
    } else {
        t = time - offset(st, freq);
    }
    float blendy = periodicFunction(t) * map(abs(mixAmp), 0.0, 100.0, 0.0, 1.0);

    color.rgb = blend(color1.rgb, color2.rgb, blendy);

    fragColor = color;
}
