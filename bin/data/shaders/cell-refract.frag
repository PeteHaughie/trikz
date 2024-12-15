#version 300 es
precision highp float;
precision highp int;

uniform sampler2D mixerTex;
uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform int metric;
uniform float scale;
uniform float refractAmt;
uniform float loopScale;
uniform float loopAmp;
uniform int loopOffset;
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

	// should be effectWidth
	float alpha = 0.75;
	color = vec3(color.x, color.y, mix(color.z, shade, alpha));
	return hsv2rgb(color);
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
    st -= vec2(0.5 * aspectRatio, 0.5);
    st *= freq;
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    i += floor(seed);
    float a = prng(vec3(i, 0.0)).x;
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
        // horizontal
        return st.x * freq * 0.5;
    } else if (loopOffset == 210) {
        // vertical
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
    float d = 0.0;
    if (metric == 5) {
        d = smoothVoronoi(st, blend, freq);
    } else {
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
                    dist = max(abs(diff.x) - diff.y * -0.5 * aspectRatio, -1.0 * diff.y);
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
        d = m_dist;
        //return shaper(m_dist, blend);

        // Draw cell center
        //color += 1.-step(.02, m_dist);
        
        // Draw grid
        //color.r += step(.98, f_st.x) + step(.98, f_st.y);
    }
    return d;
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
    if (size <= 1.0) {
        return texture(mixerTex, uv).rgb;
    }

    float dx = size * (1.0 / resolution.x);
    float dy = size * (1.0 / resolution.y);
    vec2 coord = vec2(dx * floor(uv.x / dx), dy * floor(uv.y / dy));
    return texture(mixerTex, coord).rgb;
}


void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);

    vec2 st = gl_FragCoord.xy / resolution.y;
    st.y = 1.0 - st.y;

    loadKernels();
    float lf = map(loopScale, 1.0, 100.0, 6.0, 1.0);
    /*
    float t = 1.0;
    if (loopAmp < 0.0) {
        t = time + offset(st, lf);
    } else {
        t = time - offset(st, lf);
    }
    */
    float t = time + offset(st, lf) * loopAmp * 0.01;
    float blend = periodicFunction(t) * map(abs(loopAmp), 0.0, 100.0, 0.0, 2.0);
    
    //vec4 orig = texture(mixerTex, st);

    float d = cellnoise(st, blend, map(scale, 1.0, 100.0, 20.0, 1.0));
    //float sum = (orig.r + orig.g + orig.b) / 3.0;

    float ref = map(refractAmt, 0.0, 100.0, 0.0, 1.0);

    float xOffset = sin(d * ref * TAU);
    float yOffset = cos(d * ref * TAU);
    xOffset = map(xOffset, -1.0, 1.0, -0.25, 0.25) - 0.25 * ref;
    yOffset = map(yOffset, -1.0, 1.0, -0.25, 0.25) - 0.25 + 0.25 * ref;
    st.x = mod(st.x + xOffset, 1.0);
    st.y = mod(st.y + yOffset, 1.0);

    color = texture(mixerTex, st);

    if (effectWidth != 0.0 && kernel != 0) {
        if (kernel == 100) {
            color.rgb = pixellate(st, effectWidth * 4.0);
        } else if (kernel == 110) {
            color.rgb = posterize(color.rgb, floor(map(effectWidth, 0.0, 10.0, 0.0, 20.0)));
        } else {
            color.rgb = convolution(kernel, color.rgb, st);
        }
    }

    // brightness/contrast/saturation
	color.rgb = brightnessContrast(color.rgb);
	color.rgb = saturate(color.rgb);
    
    fragColor = color;
}
