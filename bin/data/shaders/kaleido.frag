#version 300 es
precision highp float;
precision highp int;

uniform sampler2D mixerTex;
uniform vec2 resolution;
uniform float time;
uniform bool wrap;
uniform float seed;
uniform float loopFreq;
uniform float loopAmp;
uniform float loopScale;
uniform int loopOffset;
uniform float kaleido; 
uniform int metric;
uniform int direction;
uniform int kernel;
uniform float effectWidth;
uniform float intensity;
uniform float saturation;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y

// convolution kernels
float emboss[9];
float sharpen[9];
float blur[9];
float edge[9];
float edge2[9];

void loadKernels() {
	// kernels can be declared outside of function but values must be set inside function
	// emboss kernel
	emboss[0] = -2.0; emboss[1] = -1.0; emboss[2] = 0.0;
	emboss[3] = -1.0; emboss[4] = 1.0; emboss[5] = 1.0;
	emboss[6] = 0.0; emboss[7] = 1.0; emboss[8] = 2.0;

	// sharpen kernel
	sharpen[0] = -1.0; sharpen[1] = 0.0; sharpen[2] = -1.0;
	sharpen[3] = 0.0; sharpen[4] = 5.0; sharpen[5] = 0.0;
	sharpen[6] = -1.0; sharpen[7] = 0.0; sharpen[8] = -1.0;

	// gaussian blur kernel
	blur[0] = 1.0; blur[1] = 2.0; blur[2] = 1.0;
	blur[3] = 2.0; blur[4] = 4.0; blur[5] = 2.0;
	blur[6] = 1.0; blur[7] = 2.0; blur[8] = 1.0;

	// edge detect kernel
	edge[0] = -1.0; edge[1] = -1.0; edge[2] = -1.0;
	edge[3] = -1.0; edge[4] = 8.0; edge[5] = -1.0;
	edge[6] = -1.0; edge[7] = -1.0; edge[8] = -1.0;

	// edge detect kernel 2
	edge2[0] = -1.0; edge2[1] = 0.0; edge2[2] = -1.0;
	edge2[3] = 0.0; edge2[4] = 4.0; edge2[5] = 0.0;
	edge2[6] = -1.0; edge2[7] = 0.0; edge2[8] = -1.0;
}

float circles(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return dist * freq;
}

float rings(vec2 st, float freq) {
    float dist = length(st - vec2(0.5 * aspectRatio, 0.5));
    return cos(dist * PI * freq);
}

float diamonds(vec2 st, float freq) {
    st.x -= 0.5 * aspectRatio;
    st *= freq;
    return (sin(st.x * PI) + sin(st.y * PI));
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

float prng2(vec2 p) {
    vec3 p2 = vec3(p, 0.0);
    return float(pcg(uvec3(p2)).x) / float(uint(0xffffffff));
}
// end PCG PRNG

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

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

    float scaledTime = periodicFunction(prng2(floor(vec2(x + 40.0, y))) - time) * map(abs(loopAmp), 0.0, 100.0, 0.0, 0.333);

    return periodicFunction(prng2(floor(vec2(x, y))) - scaledTime);
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
    st -= vec2(0.5 * aspectRatio, 0.5);
    if (interp == 3) {
        return bicubicValue(st, freq);
    }

    float x1y1 = constant(st, freq);

    if (interp == 0) {
        return x1y1;
    }

    // Neighbor Distance
    float ndX = 1.0 / freq;
    float ndY = 1.0 / freq;

    float x1y2 = constant(vec2(st.x, st.y + ndY), freq);
    float x2y1 = constant(vec2(st.x + ndX, st.y), freq);
    float x2y2 = constant(vec2(st.x + ndX, st.y + ndY), freq);

    vec2 uv = st * freq;

    float a = blendLinearOrCosine(x1y1, x2y1, fract(uv.x), interp);
    float b = blendLinearOrCosine(x1y2, x2y2, fract(uv.x), interp);

    return blendLinearOrCosine(a, b, fract(uv.y), interp);
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

vec3 convolve(vec2 uv, float kernel[9], bool divide) {
    vec2 steps = 1.0 / resolution; // 1.0 / width = 1 texel
    vec2 offset[9];
    offset[0] = vec2(-steps.x, -steps.y);   // top left
    offset[1] = vec2(0.0, -steps.y);        // top middle
    offset[2] = vec2(steps.x, -steps.y);    // top right
    offset[3] = vec2(-steps.x, 0.0);        // middle left
    offset[4] = vec2(0.0, 0.0);             //middle
    offset[5] = vec2(steps.x, 0.0);         //middle right
    offset[6] = vec2(-steps.x, steps.y);    //bottom left
    offset[7] = vec2(0.0, steps.y);         //bottom middle
    offset[8] = vec2(steps.x, steps.y);     //bottom right

    float kernelWeight = 0.0;
    vec3 conv = vec3(0.0);

    for(int i = 0; i < 9; i++){
        //sample a 3x3 grid of pixels
        vec3 color = texture(mixerTex, uv + offset[i] * effectWidth).rgb;

        // multiply the color by the kernel value and add it to our conv total
        conv += color * kernel[i];

        // keep a running tally of the kernel weights
        kernelWeight += kernel[i];
    }

    // normalize the convolution by dividing by the kernel weight
    if (divide) {
        conv.rgb /= kernelWeight;
    }

    return clamp(conv.rgb, 0.0, 1.0);
}

vec3 desaturate(vec3 color) {
	float avg = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
	return vec3(avg);
}

vec3 derivatives(vec3 color, vec2 uv, bool divide) {
	// use: desaturate, get deriv_x and deriv_y and calculate dist between, then multiply by color
	vec3 dcolor = desaturate(color);

	float deriv_x[9];
	deriv_x[0] = 0.0; deriv_x[1] = 0.0; deriv_x[2] = 0.0;
	deriv_x[3] = 0.0; deriv_x[4] = 1.0; deriv_x[5] = -1.0;
	deriv_x[6] = 0.0; deriv_x[7] = 0.0; deriv_x[8] = 0.0;

	float deriv_y[9];
	deriv_y[0] = 0.0; deriv_y[1] = 0.0; deriv_y[2] = 0.0;
	deriv_y[3] = 0.0; deriv_y[4] = 1.0; deriv_y[5] = 0.0;
	deriv_y[6] = 0.0; deriv_y[7] = -1.0; deriv_y[8] = 0.0;

	vec3 s1 = convolve(uv, deriv_x, divide);
	vec3 s2 = convolve(uv, deriv_y, divide);
	float dist = distance(s1, s2);
	return color *= dist;
}

vec3 sobel(vec3 color, vec2 uv) {
	// use: desaturate, get sobel_x and sobel_y and calculate dist between, then multiply by color
	vec3 dcolor = desaturate(color);
	
	float sobel_x[9];
	sobel_x[0] = 1.0; sobel_x[1] = 0.0; sobel_x[2] = -1.0;
	sobel_x[3] = 2.0; sobel_x[4] = 0.0; sobel_x[5] = -2.0;
	sobel_x[6] = 1.0; sobel_x[7] = 0.0; sobel_x[8] = -1.0;

	float sobel_y[9];
	sobel_y[0] = 1.0; sobel_y[1] = 2.0; sobel_y[2] = 1.0;
	sobel_y[3] = 0.0; sobel_y[4] = 0.0; sobel_y[5] = 0.0;
	sobel_y[6] = -1.0; sobel_y[7] = -2.0; sobel_y[8] = -1.0;

	vec3 s1 = convolve(uv, sobel_x, false);
	vec3 s2 = convolve(uv, sobel_y, false);
	float dist = distance(s1, s2);
	return color *= dist;
}

vec3 outline(vec3 color, vec2 uv) {
    // use: desaturate, get sobel_x and sobel_y and calculate dist between, then multiply by color
    vec3 dcolor = desaturate(color);

    float sobel_x[9];
    sobel_x[0] = 1.0; sobel_x[1] = 0.0; sobel_x[2] = -1.0;
    sobel_x[3] = 2.0; sobel_x[4] = 0.0; sobel_x[5] = -2.0;
    sobel_x[6] = 1.0; sobel_x[7] = 0.0; sobel_x[8] = -1.0;

    float sobel_y[9];
    sobel_y[0] = 1.0; sobel_y[1] = 2.0; sobel_y[2] = 1.0;
    sobel_y[3] = 0.0; sobel_y[4] = 0.0; sobel_y[5] = 0.0;
    sobel_y[6] = -1.0; sobel_y[7] = -2.0; sobel_y[8] = -1.0;

    vec3 s1 = convolve(uv, sobel_x, false);
    vec3 s2 = convolve(uv, sobel_y, false);
    float dist = distance(s1, s2);

    vec3 outcolor = color - dist;
    return max(outcolor, 0.0);
}

vec3 shadow(vec3 color, vec2 uv) {
    float sobel_x[9];
    sobel_x[0] = 1.0; sobel_x[1] = 0.0; sobel_x[2] = -1.0;
    sobel_x[3] = 2.0; sobel_x[4] = 0.0; sobel_x[5] = -2.0;
    sobel_x[6] = 1.0; sobel_x[7] = 0.0; sobel_x[8] = -1.0;

    float sobel_y[9];
    sobel_y[0] = 1.0; sobel_y[1] = 2.0; sobel_y[2] = 1.0;
    sobel_y[3] = 0.0; sobel_y[4] = 0.0; sobel_y[5] = 0.0;
    sobel_y[6] = -1.0; sobel_y[7] = -2.0; sobel_y[8] = -1.0;

    color = rgb2hsv(color);

    vec3 x = convolve(uv, sobel_x, false);
    vec3 y = convolve(uv, sobel_y, false);

    float shade = distance(x, y);
    float highlight = shade * shade;
    shade = (1.0 - ((1.0 - color.z) * (1.0 - highlight))) * shade;

    // should be effectAmt
    float alpha = 0.75;
    color = vec3(color.x, color.y, mix(color.z, shade, alpha));
    return hsv2rgb(color);
}

vec3 convolution(int kernel, vec3 color, vec2 uv) {
    vec3 color1 = vec3(0.0);

    if (kernel == 0) {
        return color;
    } else if (kernel == 1) {
        color1 = convolve(uv, blur, true);
    } else if (kernel == 2) {
        // deriv divide
        color1 = derivatives(color, uv, true);
    } else if (kernel == 120) {
        // deriv
        color1 = clamp(derivatives(color, uv, false) * 2.5, 0.0, 1.0);
    } else if (kernel == 3) {
        color1 = convolve(uv, edge2, true);
        color1 = color * color1;
    } else if (kernel == 4) {
        color1 = convolve(uv, emboss, false);
    } else if (kernel == 5) {
        color1 = outline(color, uv);
    } else if (kernel == 6) {
        color1 = shadow(color, uv);
    } else if (kernel == 7) {
        color1 = convolve(uv, sharpen, false);
    } else if (kernel == 8) {
        color1 = sobel(color, uv);
    }

    return color1;
}

float shape(vec2 st, int sides, float blend) {
	if (sides < 2) {
		return distance(st, vec2(0.5));
	}
    // Remap st to -1 to 1
	st.y = 1.0 - st.y;
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

vec3 posterize(vec3 color, float lev) {
    if (lev == 0.0) {
        return color;
    } else if (lev == 1.0) {
        lev = 2.0;
    }

    color = clamp(color, 0.0, 0.99); // avoids speckles
    color = color * lev;
    color = floor(color) + 0.5;
    color = color / lev;
    return color;
}

vec3 pixellate(vec2 uv, float size) {
	float dx = size * (1.0 / resolution.x);
	float dy = size * (1.0 / resolution.y);
	vec2 coord = vec2(dx * floor(uv.x / dx), dy * floor(uv.y / dy));
	return texture(mixerTex, coord).rgb;
}

float getMetric(vec2 st) {
    vec2 diff = vec2(0.5 * aspectRatio, 0.5) - st;
    float r = 1.0;

    if (metric == 0) {
        // euclidean
        r = length(st - vec2(0.5 * aspectRatio, 0.5));
    } else if (metric == 1) {
		// manhattan
        r = abs(diff.x) + abs(diff.y);
    } else if (metric == 2) {
		// hexagon
        r = max(max(abs(diff.x) - diff.y * -0.5, -1.0 * diff.y), max(abs(diff.x) - diff.y * 0.5, 1.0 * diff.y));
    } else if (metric == 3) {
        // octagon
        r = max((abs(diff.x) + abs(diff.y)) / sqrt(2.0), max(abs(diff.x), abs(diff.y)));
    } else if (metric == 4) {
        // chebychev
        r = max(abs(diff.x), abs(diff.y));
    } else if (metric == 5) {
        // triangle
        r = max(abs(diff.x) - (diff.y) * -0.5, -1.0 * (diff.y));
    }

    return r;
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
        return 1.0 - value(st, freq, 0);
    } else if (loopOffset == 310) {
        // linear
        return 1.0 - value(st, freq, 1);
    } else if (loopOffset == 320) {
        // cosine
        return 1.0 - value(st, freq, 2);
    } else if (loopOffset == 330) {
        // bicubic
        return 1.0 - value(st, freq, 3);
    } else if (loopOffset == 400) {
        // rings
        return 1.0 - rings(st, freq);
    } else if (loopOffset == 410) {
        // sine
        return 1.0 - diamonds(st, freq);
    }
}

vec2 kaleidoscope(vec2 st, float sides, float blendy) {
	
	// distance metric
	float r = getMetric(st) + blendy;

    // cartesian to polar coordinates
    st = st - vec2(0.5 * aspectRatio, 0.5);
	float a = atan(st.y, st.x);

	float dir = time;
	if (direction == 1) {
		dir *= -1.0;
	} else if (direction == 2) {
		dir = 1.0;
	}
	// Repeat side according to angle
	//float ma = mod(a - radians(360.0 / sides * dir), TAU/sides);
	float ma = mod(a + radians(90.0) - radians(360.0 / sides * dir), TAU/sides);
	ma = abs(ma - PI/sides);

	// polar to cartesian coordinates
	st = r * vec2(cos(ma), sin(ma));
	st = fract(st);
	return st;
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.y;
	uv.y = 1.0 - uv.y;

	vec4 color = vec4(0.0);
    loadKernels();

    float lf = map(loopScale, 1.0, 100.0, 6.0, 1.0);
    if (wrap) {
        lf = floor(lf);
    }

    float t = time + offset(uv, lf) * loopAmp * 0.01;
	float blendy = periodicFunction(t) * map(abs(loopAmp), 0.0, 100.0, 0.0, 2.0);

	uv = kaleidoscope(uv, kaleido, blendy);
	color = texture(mixerTex, uv);
	
    if (effectWidth != 0.0 && kernel != 0) {
        if (kernel == 10) {
		    color.rgb = pixellate(uv, effectWidth * 4.0);
        } else if (kernel == 110) {
            color.rgb = posterize(color.rgb, floor(map(effectWidth, 0.0, 10.0, 0.0, 20.0)));
        } else {
		    color.rgb = convolution(kernel, color.rgb, uv);
        }
    }

	// brightness/contrast/saturation
	color.rgb = brightnessContrast(color.rgb);
	color.rgb = saturate(color.rgb);

	fragColor = vec4(color.rgb, 1.0);
}
