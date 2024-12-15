#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform int metric;
uniform float scale;
uniform float loopScale;
uniform float loopAmp;
uniform int loopOffset;
uniform int paletteMode;
uniform vec3 paletteOffset;
uniform vec3 paletteAmp;
uniform vec3 paletteFreq;
uniform vec3 palettePhase;
uniform int colorMode;
uniform bool cycleColors;
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
    //float dist = length(st - 0.5);
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return dist * freq;
}

float rings(vec2 st, float freq) {
    //float dist = length(st - 0.5);
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
    //st = st * 2.0 - 1.0;
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

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
// copy of noise generation functions with no seed or periodic function (for looping)

float f(vec2 st) {
    return prng(vec3(floor(st), seed)).x;
}

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
    //st = gl_FragCoord.xy / resolution.y - vec2(0.5 * aspectRatio, 0.5);
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

float smoothVoronoi(vec2 st, float blend, float freq) {
    blend = 0.9 + abs(blend);
    st *= freq;
    vec2 ip = floor(st);
    st -= ip;
    float d = 1.0, res = 0.0;

    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            vec2 b = vec2(i, j);
            vec2 v = b - st + cos(prng(vec3(ip + b, 0.0)).xy * TAU + blend) * 0.5;
            d = max(dot(v, v), 1e-4);
            res += 1.0 / pow(d, 8.0);
        }
    }
    return pow(1.0 / res, 1.0 / 16.0);
}

// from http://thebookofshaders.com/12/
float cellnoise(vec2 st, float blend, float freq) {
    if (metric == 5) {
        return smoothVoronoi(st, blend, freq);
    }

    vec3 color = vec3(0.0);

    // Scale 
    st *= freq;
    
    // Tile the space
    vec2 i_st = floor(st);
    vec2 f_st = fract(st);

    float m_dist = 1.0;  // minimum distance

    int nth = 2; // neighbor? not being used
    
    for (int y= -1; y <= 1; y++) {
        for (int x= -1; x <= 1; x++) {
            // Neighbor place in the grid
            vec2 neighbor = vec2(float(x), float(y));
            
            // Wrap edges
            vec2 wrapNeighbor = i_st + neighbor;
            if (wrapNeighbor.x < 0.0) {
                wrapNeighbor.x = freq - 1.0;
            }
            if (wrapNeighbor.x >= freq * aspectRatio) {
                wrapNeighbor.x = 0.0;
            }
            if (wrapNeighbor.y < 0.0) {
                wrapNeighbor.y = freq - 1.0;
            }
            if (wrapNeighbor.y >= freq) {
                wrapNeighbor.y = 0.0;
            }
            
            // Random position from current + neighbor place in the grid
            vec2 point = prng(vec3(wrapNeighbor, seed)).xy;

            // animate
            //point *= blend;
            // Animate the point
            //point = 0.5 + 0.5*sin(time + 6.2831*point);

            point = 0.5 + 0.5*sin(blend + 6.2831*point);
            
            // Vector between the pixel and the point
            vec2 diff = neighbor + point - f_st;

            float dist = 0.0;
            vec2 pointy = neighbor + point;

            if (metric == 0) {
                // Euclidean
                dist = length(diff);
            } else if (metric == 1) {
                // Manhattan
                dist = abs(pointy.x - f_st.x) + abs(pointy.y - f_st.y);
            } else if (metric == 2) {
                // hexagon
                dist = max(max(abs(diff.x) - diff.y * -0.5, -1.0 * diff.y), max(abs(diff.x) - diff.y * 0.5, 1.0 * diff.y));
            } else if (metric == 3) {
                // octagon
                dist = max((abs(pointy.x - f_st.x) + abs(pointy.y - f_st.y)) / sqrt(2.0), max(abs(pointy.x - f_st.x), abs(pointy.y - f_st.y)));
            } else if (metric == 4) {
                // Chebychev
                dist = max(abs(pointy.x - f_st.x), abs(pointy.y - f_st.y));
            } else if (metric == 6) {
                // Triangle
                //dist = max(abs(diff.x) - diff.y * 0.5, diff.y);
                dist = max(abs(diff.x * aspectRatio) - diff.y * -0.5 * aspectRatio, -1.0 * diff.y);
                // rounded triangle - p = diff
                //dist = (dot(diff, diff) * 4.0 * 0.25 + 0.75) * max(abs(diff.x) * 0.866025 + diff.y * 0.5, -diff.y);
            }

            // Keep the closer distance
            m_dist = min(m_dist, dist);
            //m_dist = min(m_dist, m_dist * dist);

        }
    }
    
    // Draw the min distance (distance field)
    //color += m_dist;
    return m_dist;
    //return shaper(m_dist, blend);

    // Draw cell center
    //color += 1.-step(.02, m_dist);
    
    // Draw grid
    //color.r += step(.98, f_st.x) + step(.98, f_st.y);
}



void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution.y;	
    

    float lf = map(loopScale, 1.0, 100.0, 6.0, 1.0);
	float t = 1.0;
	if (loopAmp < 0.0) {
	    t = time + offset(st, lf);
	} else {
		t = time - offset(st, lf);
	}
    float blend = periodicFunction(t) * map(abs(loopAmp), 0.0, 100.0, 0.0, 2.0);

    float d = cellnoise(st, blend, map(scale, 1.0, 100.0, 20.0, 1.0));

    if (colorMode == 0) {
        color.rgb = vec3(d);
    } else if (colorMode == 1) {
        color.rgb = vec3(1.0 - d);
    } else if (colorMode == 2) {
        if (cycleColors) {
            color.rgb = pal(d + time);
        } else {
            color.rgb = pal(d);
        }
    }

    fragColor = color;
}
