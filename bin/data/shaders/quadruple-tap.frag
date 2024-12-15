#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;

uniform vec4 color1;
uniform vec4 color2;
uniform vec4 color3;
uniform vec4 color4;
uniform bool animate;
out vec4 fragColor;

#define TAU 6.28318530718


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

void main() {
    vec2 st = gl_FragCoord.xy / resolution.y;    

    vec4 x0 = vec4(1.0);
    vec4 x1 = vec4(1.0);

    if (animate) {
        vec3 c1 = rgb2hsv(color1.rgb);
        vec3 c2 = rgb2hsv(color2.rgb);
        vec3 c3 = rgb2hsv(color3.rgb);
        vec3 c4 = rgb2hsv(color4.rgb);

        c1[0] += (sin(time * TAU) + 1.0) * 0.05;
        c2[0] += (sin((0.25 - time) * TAU) + 1.0) * 0.05;
        c3[0] += (sin((0.5 - time) * TAU) + 1.0) * 0.05;
        c4[0] += (sin((0.75 + time) * TAU) + 1.0) * 0.05;

        c1 = hsv2rgb(c1);
        c2 = hsv2rgb(c2);
        c3 = hsv2rgb(c3);
        c4 = hsv2rgb(c4);

        x0.rgb = mix(c1, c2, st.x);
        x1.rgb = mix(c3, c4, st.x);
    } else {
        x0 = mix(color1, color2, st.x);
        x1 = mix(color3, color4, st.x);
    }

    fragColor = mix(x0, x1, 1.0 - st.y);
}
