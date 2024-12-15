#version 300 es
precision highp float;
precision highp int;

uniform sampler2D camTex;
uniform vec2 camSize;
uniform vec2 resolution;
uniform float time;
uniform float seed;
uniform int position;
uniform float rotation;
uniform float scaleAmt;
uniform float offsetX;
uniform float offsetY;
uniform int tiling;
uniform float hueRotation;
uniform float hueRange;
uniform float intensity;
out vec4 fragColor;


#define PI 3.14159265359
#define TAU 6.28318530718

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 rotate2D(vec2 st, float rot) {
    rot = map(rot, 0.0, 360.0, 0.0, 1.0);
    float angle = rot * TAU * -1.0;
    // angle = PI * u_time * 2.0; // animate rotation

    // extra code here to handle aspect ratio of camera output
    float aspect = camSize.x / camSize.y;
    st -= 0.5;
    st.x *= aspect;
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st.x *= 1.0 / aspect;
    st += 0.5;
    return st;
}

vec3 brightnessContrast(vec3 color) {
	float bright = map(intensity, -100.0, 100.0, -0.5, 0.5);
	float cont = map(intensity, -100.0, 100.0, 0.5, 1.5);
    color = (color - 0.5) * cont + 0.5 + bright;
    return color;
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

vec2 tile(vec2 st) {
    if (tiling == 0) {
        // no tiling
        return st;
    } else if (tiling == 1) {
        // tile both
        return fract(st);
    } else if (tiling == 2) {
        // horiz only
        return vec2(fract(st.x), st.y);
    } else if (tiling == 3) {
        // vert only
        return vec2(st.x, fract(st.y));
    }
}

vec4 getImage(vec2 st) {
    st = gl_FragCoord.xy / camSize;
    st = rotate2D(st, rotation);
	st.y = 1.0 - st.y;
    
    float scale = 100.0 / scaleAmt; // 25 - 400 maps to 100 / 25 (4) to 100 / 400 (0.25)
    
    if (scale == 0.0) {
        scale = 1.0;
    }
    st *= scale;
    
    // need to subtract 50% of image width and height 
    if (position == 0) {
        // top left
        st.y += (resolution.y / camSize.y * scale) - (scale - (1.0 / camSize.y * scale));
    } else if (position == 1) {
        // top center
        st.x -= (resolution.x / camSize.x * scale * 0.5) - (0.5 - (1.0 / camSize.x * scale));
        st.y += (resolution.y / camSize.y * scale) - (scale - (1.0 / camSize.y * scale));
    } else if (position == 2) {
        // top right
        st.x -= (resolution.x / camSize.x * scale) - (1.0 - (1.0 / camSize.x * scale));
        st.y += (resolution.y / camSize.y * scale) - (scale - (1.0 / camSize.y * scale));
    } else if (position == 3) {
        // mid left
        st.y += (resolution.y / camSize.y * scale * 0.5) + (0.5 - (1.0 / camSize.y * scale)) - (scale);
    } else if (position == 4) {
        // mid center
        st.x -= (resolution.x / camSize.x * scale * 0.5) - (0.5 - (1.0 / camSize.x * scale));
        st.y += (resolution.y / camSize.y * scale * 0.5) + (0.5 - (1.0 / camSize.y * scale)) - (scale);
    } else if (position == 5) {
        // mid right
        st.x -= (resolution.x / camSize.x * scale) - (1.0 - (1.0 / camSize.x * scale));
        st.y += (resolution.y / camSize.y * scale * 0.5) + (0.5 - (1.0 / camSize.y * scale)) - (scale);
    } else if (position == 6) {
        // bottom left
        st.y += 1.0 - (scale - (1.0 / camSize.y * scale));
    } else if (position == 7) {
        // bottom center
        st.x -= (resolution.x / camSize.x * scale * 0.5) - (0.5 - (1.0 / camSize.x * scale));
        st.y += 1.0 - (scale - (1.0 / camSize.y * scale));
    } else if (position == 8) {
        // bottom right
        st.x -= (resolution.x / camSize.x * scale) - (1.0 - (1.0 / camSize.x * scale));
        st.y += 1.0 - (scale - (1.0 / camSize.y * scale));
    }
    st.x -= map(offsetX, -100.0, 100.0, -resolution.x / camSize.x * scale, resolution.x / camSize.x * scale) * 1.5;
    st.y -= map(offsetY, -100.0, 100.0, -resolution.y / camSize.y * scale, resolution.y / camSize.y * scale) * 1.5;

    st = tile(st);
    vec4 text = texture(camTex, st);
    
    if (st.x < 0.0 || st.x > 1.0 || st.y < 0.0 || st.y > 1.0) {
        // don't draw texture if out of coordinate bounds
        return vec4(0.0);
    } else if (text.a == 0.0) {
        //return vec4(0.0);
    }

    // premultiply texture alpha
    text.rgb = text.rgb * text.a;
    
    return text;
}


void main() {
	vec2 st = gl_FragCoord.xy / resolution;	
	st.y = 1.0 - st.y;

    vec4 color = getImage(st);

	fragColor = color;
}
