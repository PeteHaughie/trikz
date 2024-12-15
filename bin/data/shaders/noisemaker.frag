#version 300 es
precision highp float;
precision highp int;

uniform sampler2D noiseTex;
uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform float noiseScale;
uniform int interp;
uniform float refractAmt;
uniform float loopScale;
uniform float loopAmp;
uniform int loopOffset;
uniform float hueRange;
uniform float hueRotation;
uniform bool wrap;
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
    return prng(vec3(st, 0.0)).x;
}

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = cos(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

float constant(vec2 st, float freq, float s, float blend) {
    float x = st.x * freq;
    float y = st.y * freq;
    if (wrap) {
        x = mod(x, freq);
        y = mod(y, freq);
    }
    x += s;

    return periodicFunction(random(floor(vec2(x, y))) - blend);
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

float bicubicValue(vec2 st, float freq, float s, float blend) {
    // Neighbor Distance
    float nd = 1.0 / freq;

    float u0 = st.x - nd;
    float u1 = st.x;
    float u2 = st.x + nd;
    float u3 = st.x + nd + nd;

    float v0 = st.y - nd;
    float v1 = st.y;
    float v2 = st.y + nd;
    float v3 = st.y + nd + nd;

    float x0y0 = constant(vec2(u0, v0), freq, s, blend);
    float x0y1 = constant(vec2(u0, v1), freq, s, blend);
    float x0y2 = constant(vec2(u0, v2), freq, s, blend);
    float x0y3 = constant(vec2(u0, v3), freq, s, blend);

    float x1y0 = constant(vec2(u1, v0), freq, s, blend);
    float x1y1 = constant(st, freq, s, blend);
    float x1y2 = constant(vec2(u1, v2), freq, s, blend);
    float x1y3 = constant(vec2(u1, v3), freq, s, blend);

    float x2y0 = constant(vec2(u2, v0), freq, s, blend);
    float x2y1 = constant(vec2(u2, v1), freq, s, blend);
    float x2y2 = constant(vec2(u2, v2), freq, s, blend);
    float x2y3 = constant(vec2(u2, v3), freq, s, blend);

    float x3y0 = constant(vec2(u3, v0), freq, s, blend);
    float x3y1 = constant(vec2(u3, v1), freq, s, blend);
    float x3y2 = constant(vec2(u3, v2), freq, s, blend);
    float x3y3 = constant(vec2(u3, v3), freq, s, blend);

    vec2 uv = st * freq;

    float y0 = blendBicubic(x0y0, x1y0, x2y0, x3y0, fract(uv.x));
    float y1 = blendBicubic(x0y1, x1y1, x2y1, x3y1, fract(uv.x));
    float y2 = blendBicubic(x0y2, x1y2, x2y2, x3y2, fract(uv.x));
    float y3 = blendBicubic(x0y3, x1y3, x2y3, x3y3, fract(uv.x));

    return blendBicubic(y0, y1, y2, y3, fract(uv.y));
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

float value(vec2 st, float freq, float s, float blend) {
    if (interp == 3) {
        return bicubicValue(st, freq, s, blend);
    } else if (interp == 4) {
        // Bicubic, cheaper texture-based
        return textureBicubic(st, freq, s, blend);
    } else if (interp == 10) {
        return simplexValue(st, freq, s, blend);
    }

    float x1y1 = constant(st, freq, s, blend);

    if (interp == 0) {
        st -= vec2(0.5 * aspectRatio, 0.5);
        return constant(st, freq, s, blend);
    }

    // Neighbor Distance
    float nd = 1.0 / freq;

    float x1y2 = constant(vec2(st.x, st.y + nd), freq, s, blend);
    float x2y1 = constant(vec2(st.x + nd, st.y), freq, s, blend);
    float x2y2 = constant(vec2(st.x + nd, st.y + nd), freq, s, blend);

    vec2 uv = st * freq;

    float a = blendLinearOrCosine(x1y1, x2y1, fract(uv.x), interp);
    float b = blendLinearOrCosine(x1y2, x2y2, fract(uv.x), interp);

    return blendLinearOrCosine(a, b, fract(uv.y), interp);
}

//////////////////////////////////////////////////////////////////////

float circles(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return dist * freq;
}

float rings(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return cos(dist * PI * freq);
}

float concentric(vec2 st, float freq) {
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
        // noise
        return value(st, freq, seed + 50.0, 0.0);
    } else if (loopOffset == 400) {
        // rings
        return 1.0 - rings(st, freq);
    } else if (loopOffset == 410) {
        // sine
        return 1.0 - diamonds(st, freq);
    }
}

// from https://gist.github.com/yiwenl/745bfea7f04c456e0101
vec3 hsv2rgb(vec3 c) {
    c.x = mod(c.x, 1.0);
    c.y = clamp(c.y, 0.0, 1.0);
    c.z = clamp(c.z, 0.0, 1.0);

    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 rgb2hsv(vec3 rgb) {
    float Cmax = max(rgb.r, max(rgb.g, rgb.b));
    float Cmin = min(rgb.r, min(rgb.g, rgb.b));
    float delta = Cmax - Cmin;

    vec3 hsv = vec3(0., 0., Cmax);

    if (Cmax > Cmin) {
        hsv.y = delta / Cmax;

        if (rgb.r == Cmax)
            hsv.x = (rgb.g - rgb.b) / delta;
        else {
            if (rgb.g == Cmax)
                hsv.x = 2. + (rgb.b - rgb.r) / delta;
            else
                hsv.x = 4. + (rgb.r - rgb.g) / delta;
        }
        hsv.x = fract(hsv.x / 6.);
    }
    return hsv;
}

vec3 generate_octave(vec2 st, float freq, float s, float blend) {
    vec3 color = vec3(1.0);
    color.r = value(st, freq, s, blend) * (hueRange * 0.01) + (hueRotation * 0.025);// + time;
    color.g = value(st, freq, s + 10.0, blend);
    color.b = value(st, freq, s + 20.0, blend);
    return hsv2rgb(color.rgb);
}

vec3 multires(vec2 st, float freq, float s, float blend) {
    vec3 octave1 = generate_octave(st, freq, s, blend);
    vec3 hsv = rgb2hsv(octave1);
    vec3 warp1 = generate_octave(vec2(st.x + cos(hsv.b), st.y + sin(hsv.b)), freq, s, blend);
    octave1 = mix(octave1, warp1, map(refractAmt, 0.0, 100.0, 0.0, 1.0));
    return octave1;
}

void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution.y;

    float freq = 1.0;
    float lf = 1.0;

    if (interp == 10) {
        // simplex
        freq = map(noiseScale, 1.0, 100.0, 6.0, 0.5);
        lf = map(loopScale, 1.0, 100.0, 6.0, 0.5);
    } else if (interp == 4) {
        // budget interp
        freq = map(noiseScale, 1.0, 100.0, 12.0, 2.0);
        lf = map(loopScale, 1.0, 100.0, 12.0, 2.0);
    } else {
        // everything else
        freq = map(noiseScale, 1.0, 100.0, 20.0, 3.0);
        lf = map(loopScale, 1.0, 100.0, 12.0, 3.0);
    }

    if (interp != 4 && interp != 10 && wrap) {
        freq = floor(freq);
        if (loopOffset == 300) {
            lf = floor(lf);
        }
    }

    float t = 1.0;
    if (loopAmp < 0.0) {
        t = time + offset(st, lf);
    } else {
        t = time - offset(st, lf);
    }
    float blend = periodicFunction(t) * abs(loopAmp) * 0.01;

    color.rgb = multires(st, freq, seed, blend);

    fragColor = color;
}