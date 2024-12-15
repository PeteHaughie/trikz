#version 300 es
precision highp float;
precision highp int;

uniform sampler2D mixerTex;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform bool aspectLens;
uniform float xChonk;
uniform float yChonk;
uniform float loopAmp;
uniform float glitchiness;
uniform float scanlinesAmt;
uniform float vignetteAmt;
uniform float aberrationAmt;
uniform float distortion;
uniform int kernel;
uniform float kaleido; 
uniform float pixelSize; 
uniform float levels; 
uniform bool invert;
uniform float intensity;
uniform float saturation;
out vec4 fragColor;

// convolution kernels
float emboss[9];
float sharpen[9];
float blur[9];
float edge[9];
float edge2[9];

#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y


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

float f(vec2 st) {
    return prng(vec3(floor(st), seed)).x;
}

float bicubic(vec2 p) {
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

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  	return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
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

vec3 posterize(vec3 color, float lev) {
	if (lev == 0.0) {
		return color;
	}

    color = clamp(color, 0.0, 0.99); // avoids speckles
    color = color * lev;
    color = floor(color) + 0.5;
    color = color / lev;
    return color;
}

vec3 scanlines(vec3 color, vec2 st) {
    float centerDistance = length(0.5 - st) * PI * 0.5;

    float noise = periodicFunction(bicubic(st * 4.0) - time) * map(scanlinesAmt, 0.0, 100.0, 0.0, 0.5);

    float hatch = (sin(mix(st.y, st.y + noise, pow(centerDistance, 8.0)) * resolution.y * 1.5) + 1.0) * 0.5;

    return mix(color, color * hatch, map(scanlinesAmt, 0.0, 100.0, 0.0, 0.5));
}

vec3 invertColor(vec3 color) {
	return 1.0 - color;
}

vec3 desaturate(vec3 color) {
	float avg = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
	return vec3(avg);
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

vec3 convolve(vec2 uv, float kernel[9], bool divide) {
	vec2 stepSize = vec2(1.0); // 1.0 / width = 1 texel
	vec2 steps = vec2(stepSize.x / resolution.x, stepSize.y / resolution.y);
	vec2 offset[9];
	offset[0] = vec2(-steps.x, -steps.y); 	// top left
	offset[1] = vec2(0.0, -steps.y); 		// top middle
	offset[2] = vec2(steps.x, -steps.y); 	// top right
	offset[3] = vec2(-steps.x, 0.0); 		// middle left
	offset[4] = vec2(0.0, 0.0); 			//middle
	offset[5] = vec2(steps.x, 0.0);			//middle right
	offset[6] = vec2(-steps.x, steps.y); 	//bottom left
	offset[7] = vec2(0.0, steps.y); 		//bottom middle
	offset[8] = vec2(steps.x, steps.y); 	//bottom right

	float kernelWeight = 0.0;
	vec3 conv = vec3(0.0);

	for(int i = 0; i < 9; i++){
		//sample a 3x3 grid of pixels
		vec3 color = texture(mixerTex, uv + offset[i]).rgb;

		// multiply the color by the kernel value and add it to our conv total
		conv += color * kernel[i];

		// keep a running tally of the kernel weights
		kernelWeight += kernel[i];
	}

	// normalize the convolution by dividing by the kernel weight
	if (divide) {
		conv.rgb /= kernelWeight;
	}
	
	if (levels > 0.0) {
		conv.rgb = posterize(conv.rgb, levels);
	}
	
	return clamp(conv.rgb, 0.0, 1.0);
}

vec3 derivatives(vec3 color, vec2 uv) {
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

	vec3 s1 = convolve(uv, deriv_x, true);
	vec3 s2 = convolve(uv, deriv_y, true);
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
	vec3 edges = convolve(uv, edge, true);
	//edges = 1.0 - edges;
	vec3 poster = posterize(color, 2.0);
	return poster + edges;
}


vec3 bloom(vec3 color, vec2 uv, float factor) {
	vec3 blur = convolve(uv, blur, true);
	float avg = dot(blur, vec3(0.33333));
	factor = clamp(avg * factor, 0.0, 1.0);
	return mix(color, blur, factor);
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
	int index = kernel;
	if (index == 0) {
		return color;
	} else if (index == 1) {
		color1 = convolve(uv, emboss, false);
	} else if (index == 2) {
		color1 = convolve(uv, sharpen, false);
	} else if (index == 3) {
		color1 = convolve(uv, blur, true);
	} else if (index == 4) {
		color1 = convolve(uv, edge2, true);
		color1 = color * color1;
	} else if (index == 5) {
		color1 = derivatives(color, uv);
	} else if (index == 6) {
		color1 = sobel(color, uv);
	} else if (index == 7) {
		color1 = shadow(color, uv);
	}

	return color1;
}



float offsets(vec2 st) {
	return prng(vec3(floor(st), 0.0)).x;
}

vec3 glitch(vec2 st) {
    vec2 freq = vec2(1.0);
    freq.x *= map(xChonk, 1.0, 100.0, 50.0, 1.0);
    freq.y *= map(yChonk, 1.0, 100.0, 50.0, 1.0);

    freq *= vec2(periodicFunction(prng(vec3(floor(st * freq), 0.0)).x - time));

    float g = map(glitchiness, 0.0, 100.0, 0.0, 1.0);

    // get drift value from somewhere far away
    float xDrift = prng(vec3(floor(st * freq) + 10.0, 0.0)).x * g;
    float yDrift = prng(vec3(floor(st * freq) - 10.0, 0.0)).x * g;

    float sparseness = map(glitchiness, 0.0, 100.0, 8.0, 2.0);

    // clamp for sparseness
	float rand = prng(vec3(floor(st * freq), 0.0)).x;
    float xOffset = clamp((periodicFunction(rand + xDrift - time)
        - periodicFunction(xDrift - time) * sparseness) * 4.0, 0.0, 1.0);

    float yOffset = clamp((periodicFunction(rand + yDrift - time)
        - periodicFunction(yDrift - time) * sparseness) * 4.0, 0.0, 1.0);

    float refract = g * .125;

    st.x = mod(st.x + sin(xOffset * TAU) * refract, 1.0);
    st.y = mod(st.y + sin(yOffset * TAU) * refract, 1.0);

    // aberration and lensing, borrowed from lens
    vec2 diff = vec2(0.5 - st);
	if (aspectLens) {
		diff = vec2(0.5 * aspectRatio, 0.5) - vec2(st.x * aspectRatio, st.y);
	}
    float centerDist = length(diff);

    float distort = 0.0;
    float zoom = 1.0;
    if (distortion < 0.0) {
        distort = map(distortion, -100.0, 0.0, -0.5, 0.0);
        zoom = map(distortion, -100.0, 0.0, 0.01, 0.0);
    } else {
        distort = map(distortion, 0.0, 100.0, 0.0, 0.5);
        zoom = map(distortion, 0.0, 100.0, 0.0, -0.25);
    }

    vec2 lensedCoords = fract((st - diff * zoom) - diff * centerDist * centerDist * distort);

    float aberrationOffset = map(aberrationAmt, 0.0, 100.0, 0.0, 0.05) * centerDist * PI * 0.5;

    float redOffset = mix(clamp(lensedCoords.x + aberrationOffset, 0.0, 1.0), lensedCoords.x, lensedCoords.x);
    vec4 red = texture(mixerTex, vec2(redOffset, lensedCoords.y));

    vec4 green = texture(mixerTex, lensedCoords);

    float blueOffset = mix(lensedCoords.x, clamp(lensedCoords.x - aberrationOffset, 0.0, 1.0), lensedCoords.x);
    vec4 blue = texture(mixerTex, vec2(blueOffset, lensedCoords.y));

    return vec3(red.r, green.g, blue.b);
}

void main() {
	vec2 uv = gl_FragCoord.xy / resolution;

	uv.y = 1.0 - uv.y;

	vec4 color = vec4(0.0);
	loadKernels();

	float blendy = periodicFunction(time - offsets(uv));

	vec4 origcolor = texture(mixerTex, uv);
	color.rgb = origcolor.rgb;

	// color.rgb = posterize(color.rgb, levels);
	// color.rgb = convolution(kernel, color.rgb, uv);

	//color.rgb = outline(color.rgb, uv);
	//color.rgb = bloom(color.rgb, uv, 20.0); // not working?

	color.rgb = glitch(uv);
	color.rgb = scanlines(color.rgb, uv);

	// invert
	if (invert == true) {
		color.rgb = invertColor(color.rgb);
	}

	// brightness/contrast/saturation
	color.rgb = brightnessContrast(color.rgb);
	color.rgb = saturate(color.rgb);
    
	// circle mask
	//color.rgb = color.rgb * smoothstep(0.10, 0.11, 1.0 - length(uv * 2.0 - 1.0));

	// vignette
	if (vignetteAmt < 0.0) {
		color.rgb = mix(color.rgb * 1.0 - pow(length(0.5 - uv) * 1.125, 2.0), color.rgb, map(vignetteAmt, -100.0, 0.0, 0.0, 1.0));
	} else {
		color.rgb = mix(color.rgb, 1.0 - (1.0 - color.rgb * 1.0 - pow(length(0.5 - uv) * 1.125, 2.0)), map(vignetteAmt, 0.0, 100.0, 0.0, 1.0));
	}

	// texture passthrough
	//color.rgb = origcolor.rgb;

	// outline
	//color.rgb = outline(color.rgb, uv);

	fragColor = vec4(color.rgb, 1.0);
}
