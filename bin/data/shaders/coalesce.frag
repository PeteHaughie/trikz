#version 300 es
precision highp float;
precision highp int;

uniform sampler2D synthTex1;
uniform sampler2D synthTex2;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform int blendMode;
uniform float mixAmt;
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

float blendOverlay(float a, float b) {
    return a < 0.5 ? (2.0 * a * b) : (1.0 - 2.0 * (1.0 - a) * (1.0 - b));
}

float blendSoftLight(float base, float blend) {
	return (blend<0.5)?(2.0*base*blend+base*base*(1.0-2.0*blend)):(sqrt(base)*(2.0*blend-1.0)+2.0*base*(1.0-blend));
}

vec3 blend(vec4 color1, vec4 color2, int mode, float factor) {
    // if only one noise is enabled, return that noise
    
    vec4 color;
    vec4 middle;

    float amt = map(mixAmt, -100.0, 100.0, 0.0, 1.0);

    if (mode == 0) {
        // add
        middle = min(color1 + color2, 1.0);
    } else if (mode == 1) {
        // alpha
        if (mixAmt < 0.0) {
            return mix(color1,
                       color2 * vec4(1.0 - color1.a) + color1 * vec4(color1.a),
                       map(mixAmt, -100.0, 0.0, 0.0, 1.0)).rgb;
        } else {
            return mix(color1 * vec4(1.0 - color2.a) + color2 * vec4(color2.a),
                       color2,
                       map(mixAmt, 0.0, 100.0, 0.0, 1.0)).rgb;
        }
    } else if (mode == 2) {
        // color burn
        middle = (color2 == vec4(0.0)) ? color2 : max((1.0 - ((1.0 - color1) / color2)),  vec4(0.0));
    } else if (mode == 3) {
        // color dodge
        middle = (color2 == vec4(1.0)) ? color2 : min(color1 / (1.0 - color2), vec4(1.0));
    } else if (mode == 4) {
        // darken
        middle = min(color1, color2);
    } else if (mode == 5) {
        // difference
        middle = abs(color1 - color2);
    } else if (mode == 6) {
        // exclusion
        middle = color1 + color2 - 2.0 * color1 * color2;  
    } else if (mode == 7) {
        // glow
        middle = (color2 == vec4(1.0)) ? color2 : min(color1 * color1 / (1.0 - color2), vec4(1.0));
    } else if (mode == 8) {
        // hard light
        middle = vec4(blendOverlay(color2.r, color1.r), blendOverlay(color2.g, color1.g), blendOverlay(color2.b, color1.b), mix(color1.a, color2.a, 0.5));
    } else if (mode == 9) {
        // lighten
        middle = max(color1, color2);
    } else if (mode == 10) {
        // mix
        middle = mix(color1, color2, 0.5);
    } else if (mode == 11) {
        // multiply
        middle = color1 * color2;
    } else if (mode == 12) {
        // negation
        middle = vec4(1.0) - abs(vec4(1.0) - color1 - color2);
    } else if (mode == 13) {
        // overlay
        middle = vec4(blendOverlay(color1.r, color2.r), blendOverlay(color1.g, color2.g), blendOverlay(color1.b, color2.b), mix(color1.a, color2.a, 0.5));
    } else if (mode == 14) {
        // phoenix
        middle = min(color1, color2) - max(color1, color2) + vec4(1.0);
    } else if (mode == 15) {
        // reflect
        middle = (color1 == vec4(1.0)) ? color1 : min(color2 * color2 / (1.0 - color1), vec4(1.0));
    } else if (mode == 16) {
        // screen
        middle = 1.0 - ((1.0 - color1) * (1.0 - color2));
    } else if (mode == 17) {
        // soft light
        middle = vec4(blendSoftLight(color1.r, color2.r), blendSoftLight(color1.g, color2.g), blendSoftLight(color1.b, color2.b), mix(color1.a, color2.a, 0.5));
    } else if (mode == 18) {
        // subtract
        middle = max(color1 + color2 - 1.0, 0.0);
    }

    if (factor == 0.5) {
        color = middle;
    } else if (factor < 0.5) {
        factor = map(amt, 0.0, 0.5, 0.0, 1.0);
        color = mix(color1, middle, factor);
    } else if (factor > 0.5) {
        factor = map(amt, 0.5, 1.0, 0.0, 1.0);
        color = mix(middle, color2, factor);
    }

    return color.rgb;
}





void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution;
    st.y = 1.0 - st.y;

    vec4 color1 = texture(synthTex1, st);
    vec4 color2 = texture(synthTex2, st);

    color.rgb = blend(color1, color2, blendMode, mixAmt);

    fragColor = color;
}
