#version 300 es
precision highp float;
precision highp int;

uniform sampler2D synthTex1;
uniform sampler2D synthTex2;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform int blendMode;
uniform float cutoff;
uniform bool flip;
out vec4 fragColor;

#define PI 3.14159265359
#define TAU 6.28318530718


float map(float value, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float ridge(float h, float peak) {
    h = abs(h);     // create creases
    h = peak - h;   // invert so creases are at top
    h = h * h;      // sharpen creases
    return h;
}

float desaturate(vec3 color) {
    //return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    return (color.r + color.g + color.b) / 3.0;
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

vec3 blend(vec3 color1, vec3 color2) {
    vec3 color = vec3(0.0);
    float cut = cutoff * 0.01;

    if (blendMode == 0) {
        // a -> b black
        float c = step(cut, desaturate(color2));
        if (flip) { c = 1.0 - c; }
        color = mix(color1, vec3(0.0), c);
    } else if (blendMode == 1) {
        // a -> b color black
        vec3 c = step(cut, color2);
        if (flip) { c = 1.0 - c; }
        color = mix(color1, vec3(0.0), c);
    } else if (blendMode == 2) {
        // a -> b hue
        float c = rgb2hsv(color2).r;
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c * cut);
    } else if (blendMode == 3) {
        // a -> b saturation
        float c = rgb2hsv(color2).g;
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c * cut);
    } else if (blendMode == 4) {
        // a -> b value
        float c = rgb2hsv(color2).b;
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c * cut);
    } else if (blendMode == 5) {
        // b -> a black
        float c = step(cut, desaturate(color1));
        if (flip) { c = 1.0 - c; }
        color = mix(color2, vec3(0.0), c);
    } else if (blendMode == 6) {
        // b -> a color black
        vec3 c = step(cut, color1);
        if (flip) { c = 1.0 - c; }
        color = mix(color2, vec3(0.0), c);
    } else if (blendMode == 7) {
        // b -> a hue
        float c = rgb2hsv(color1).r;
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c * cut);
    } else if (blendMode == 8) {
        // b -> a saturation
        float c = rgb2hsv(color1).g;
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c * cut);
    } else if (blendMode == 9) {
        // b -> a value
        float c = rgb2hsv(color1).b;
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c * cut);
    } else if (blendMode == 10) {
        // mix
        if (flip) { cut = 1.0 - cut; }
        color = mix(color1, color2, cut);
    } else if (blendMode == 11) {
        // psychedelic
        vec3 c = step(cut, mix(color1, color2, 0.5));
        if (flip) { c = 1.0 - c; }
        color = mix(color1, color2, c);
    } else if (blendMode == 12) {
        // psychedelic 2
        if (flip) { cut = 1.0 - cut; }
        color = smoothstep(color1, color2, vec3(cut));
    }

    return color;
}


void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution;
    st.y = 1.0 - st.y;

    vec4 color1 = texture(synthTex1, st);
    vec4 color2 = texture(synthTex2, st);

    color.rgb = blend(color1.rgb, color2.rgb);

    fragColor = color;
}
