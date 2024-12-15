#version 300 es
precision highp float;
precision highp int;

uniform sampler2D mixerTex;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform int effect;
uniform float effectAmt;
uniform float scaleAmt;
uniform float rotation;
uniform float offsetX;
uniform float offsetY;
uniform int flip;
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

float random(vec2 p) {
    vec3 p2 = vec3(p, 0.0);
    return float(pcg(uvec3(p2)).x) / float(uint(0xffffffff));
}


float map(float value, float inMin, float inMax, float outMin, float outMax) {
      return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 rotate2D(vec2 st, float rot) {
    st.x *= aspectRatio;
    rot = map(rot, 0.0, 360.0, 0.0, 2.0);
    float angle = rot * PI;
    // angle = PI * u_time * 2.0; // animate rotation
    st -= vec2(0.5 * aspectRatio, 0.5);
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st += vec2(0.5 * aspectRatio, 0.5);
    st.x /= aspectRatio;
    return st;
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
    vec2 steps = 1.0 / resolution; // 1.0 / width = 1 texel
    vec2 offset[9];
    offset[0] = vec2(-steps.x, -steps.y);     // top left
    offset[1] = vec2(0.0, -steps.y);         // top middle
    offset[2] = vec2(steps.x, -steps.y);     // top right
    offset[3] = vec2(-steps.x, 0.0);         // middle left
    offset[4] = vec2(0.0, 0.0);             //middle
    offset[5] = vec2(steps.x, 0.0);            //middle right
    offset[6] = vec2(-steps.x, steps.y);     //bottom left
    offset[7] = vec2(0.0, steps.y);         //bottom middle
    offset[8] = vec2(steps.x, steps.y);     //bottom right

    float kernelWeight = 0.0;
    vec3 conv = vec3(0.0);

    for(int i = 0; i < 9; i++){
        //sample a 3x3 grid of pixels
        vec3 color = texture(mixerTex, uv + offset[i] * floor(map(effectAmt, 0.0, 100.0, 0.0, 20.0))).rgb;

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
    } else if (kernel == 9) {
        // lit edge
        color1 = convolve(uv, edge2, true);
        color1 = max(color, color1);
    }

    return color1;
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

float offsets(vec2 st) {
    return distance(st, vec2(0.5));
}

float f(vec2 st) {
    return random(floor(st));
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    uv.y = 1.0 - uv.y;

    vec4 color = vec4(0.0);

    float scale = 100.0 / scaleAmt; // 25 - 400 maps to 100 / 25 (4) to 100 / 400 (0.25)

    if (scale == 0.0) {
        scale = 1.0;
    }
    uv = rotate2D(uv, rotation) * scale;

    // no
    vec2 imageSize = resolution;

    // need to subtract 50% of image width and height
    // mid center
    uv.x -= ceil((resolution.x / imageSize.x * scale * 0.5) - (0.5 - (1.0 / imageSize.x * scale)));
    uv.y += ceil((resolution.y / imageSize.y * scale * 0.5) + (0.5 - (1.0 / imageSize.y * scale)) - (scale));

    uv.x -= map(offsetX, -100.0, 100.0, -resolution.x / imageSize.x * scale, resolution.x / imageSize.x * scale) * 1.5;
    uv.y -= map(offsetY, -100.0, 100.0, -resolution.y / imageSize.y * scale, resolution.y / imageSize.y * scale) * 1.5;

    uv = fract(uv);

    if (flip == 1) {
       // flip both
       uv.x = 1.0 - uv.x;
       uv.y = 1.0 - uv.y;
    } else if (flip == 2) {
       // flip h
       uv.x = 1.0 - uv.x;
    } else if (flip == 3) {
       // flip v
       uv.y = 1.0 - uv.y;
    } else if (flip == 11) {
       // mirror lr
       if (uv.x > 0.5) {
           uv.x = 1.0 - uv.x;
       }
    } else if (flip == 12) {
       // mirror rl
       if (uv.x < 0.5) {
           uv.x = 1.0 - uv.x;
       }
    } else if (flip == 13) {
       // mirror ud
       if (uv.y > 0.5) {
           uv.y = 1.0 - uv.y;
       }
    } else if (flip == 14) {
       // mirror du
       if (uv.y < 0.5) {
           uv.y = 1.0 - uv.y;
       }
    } else if (flip == 15) {
       // mirror lr ud
       if (uv.x > 0.5) {
           uv.x = 1.0 - uv.x;
       }
       if (uv.y > 0.5) {
           uv.y = 1.0 - uv.y;
       }
    } else if (flip == 16) {
       // mirror lr du
       if (uv.x > 0.5) {
           uv.x = 1.0 - uv.x;
       }
       if (uv.y < 0.5) {
           uv.y = 1.0 - uv.y;
       }
    } else if (flip == 17) {
       // mirror rl ud
       if (uv.x < 0.5) {
           uv.x = 1.0 - uv.x;
       }
       if (uv.y > 0.5) {
           uv.y = 1.0 - uv.y;
       }
    } else if (flip == 18) {
       // mirror rl du
       if (uv.x < 0.5) {
           uv.x = 1.0 - uv.x;
       }
       if (uv.y < 0.5) {
           uv.y = 1.0 - uv.y;
       }
    }

    //
    loadKernels();

    float blendy = periodicFunction(time - offsets(uv));

    vec2 origUV = uv;
    vec4 origcolor = texture(mixerTex, uv);
    color.rgb = origcolor.rgb;

    if (effectAmt != 0.0 && effect != 0) {
        if (effect == 100) {
            color.rgb = pixellate(uv, floor(map(effectAmt, 0.0, 100.0, 0.0, 20.0)));
        } else if (effect == 110) {
            color.rgb = posterize(color.rgb, floor(map(effectAmt, 0.0, 100.0, 0.0, 20.0)));
        } else {
            color.rgb = convolution(effect, color.rgb, uv);
        }
    }

    if (invert == true) {
        color.rgb = invertColor(color.rgb);
    }

    // brightness/contrast/saturation
    color.rgb = brightnessContrast(color.rgb);
    color.rgb = saturate(color.rgb);

    fragColor = vec4(color.rgb, 1.0);
}
