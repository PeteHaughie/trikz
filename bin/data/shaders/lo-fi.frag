#version 300 es
precision highp float;
precision highp int;

uniform sampler2D mixerTex;
uniform float time;
uniform float seed;
uniform bool aspectLens;
uniform float freq;
uniform vec2 resolution;
uniform int effectType;
uniform float effectAmt;
uniform float dotsAmt;
uniform float scanlinesAmt;
uniform float fabricSize;
uniform float snowAmt;
uniform float vignetteAmt;
uniform float distortion;
uniform float intensity;
uniform float saturation;
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

float random(vec2 p) {
    vec3 p2 = vec3(p, 0.0);
    return float(pcg(uvec3(p2)).x) / float(uint(0xffffffff));
}

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float noise(vec2 st, float freq) {
    st *= freq;
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = smoothstep(0.1, 0.9, f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float f(vec2 st) {
    return random(floor(st));
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

vec3 fabric(vec3 color, vec2 st) {
    float hatch = sin(st.y * PI * fabricSize);
    hatch *= sin(st.x * PI * 200.0);
    color *= hatch * 0.25 + 0.75;
    return color;
}

vec3 snow(vec4 color, vec2 st) {
    st = gl_FragCoord.xy;
    float amt = snowAmt / 100.0;
    float noise = prng(vec3(st, time * 1000.0)).x;

    float mask;
    float maskNoise = prng(vec3(st + 10.0, time * 1000.0)).x;
    float maskNoiseSparse = clamp(maskNoise - 0.93875, 0.0, 0.06125) * 16.0;

    if (amt < .5) {
        mask = mix(0.0, maskNoiseSparse, amt * 2.0);
    } else {
        mask = mix(maskNoiseSparse, maskNoise * maskNoise, map(amt, 0.5, 1.0, 0.0, 1.0));

        if (amt > .75) {
            mask = mix(mask, 1.0, map(amt, 0.75, 1.0, 0.0, 1.0));
        }
    }

    return mix(color.rgb, vec3(noise), mask);
}

vec3 tracking(vec4 color, vec2 st) {
	// aka scanline error
	if (effectAmt == 0.0) {
		return color.rgb;
	}

	float amt = clamp(sin((st.y + time) * TAU) - map(effectAmt, 0.0, 100.0, 1.0, 0.5), 0.0, 0.5) * 2.0;

    float noise = prng(vec3(gl_FragCoord.xy, time * 1000.0)).x;
	float limiter = prng(vec3(gl_FragCoord.xy + 10.0, time * 1000.0)).x * amt;
	
	st.x = mod(st.x - pow(limiter, 2.0), 1.0);

	return mix(texture(mixerTex, st).rgb, vec3(noise), pow(amt, map(effectAmt, 0.0, 100.0, 8.0, 1.0)));
}

vec3 pixellate(vec2 uv, float size) {
	float dx = size * (1.0 / resolution.x);
	float dy = size * (1.0 / resolution.y);
	vec2 coord = vec2(dx * floor(uv.x / dx), dy * floor(uv.y / dy));
	return texture(mixerTex, coord).rgb;
}

vec3 subpixel(vec2 st, float scale) {
	scale = map(scale, 0.0, 100.0, 0.0, 10.0);

	vec3 orig = pixellate(st, 4.0 * scale);
    vec3 color = orig;

    st *= resolution;
    st = floor(st);

    float m = mod(st.x, 4.0 * scale);

    if (mod(st.y, 4.0 * scale) <= 1.0 * scale) {
        color *= vec3(0.0);
    } else if (m <= 1.0 * scale) {
        color *= vec3(1.0, 0.0, 0.0);
    } else if (m <= 2.0 * scale) {
        color *= vec3(0.0, 1.0, 0.0);
    } else if (m <= 3.0 * scale) {
        color *= vec3(0.0, 0.0, 1.0);
    } else {
        color *= vec3(0.0);
    }

    float factor = clamp(scale * 0.25, 0.0, 1.0); 
    return mix(orig, color, factor);
}

// periodic function for looping


float offset(vec2 st) {
	//return noise(st, 10.0) * 0.1;
    return distance(st, vec2(0.5));
}

float periodicFunction(float p) {
    float x = TAU * p;
    float func = sin(x);
    return map(func, -1.0, 1.0, 0.0, 1.0);
}

float concentric(vec2 st, float freq) {
    float dist = length(st - 0.5);
    return cos(dist * PI * freq);
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

vec3 saturate(vec4 color) {
	float sat = map(saturation, -100.0, 100.0, -1.0, 1.0);
    float avg = (color.r + color.g + color.b) / 3.0;
    color.rgb -= (avg - color.rgb) * sat;
    return color.rgb;
}

vec3 scanlines(vec3 color, vec2 st) {
    float centerDistance = length(0.5 - st) * PI * 0.5;

    float noise = periodicFunction(bicubic(st * 4.0) - time) * map(scanlinesAmt, 0.0, 100.0, 0.0, 0.5);

    float hatch = (sin(mix(st.y, st.y + noise, pow(centerDistance, 8.0)) * resolution.y * 1.5) + 1.0) * 0.5;

    return mix(color, color * hatch, map(scanlinesAmt, 0.0, 100.0, 0.0, 0.5));
}

vec2 distortionCoords(vec2 st) {
    // lensing, borrowed from lens
    vec2 diff = vec2(0.5) - st;
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

    // return displaced coords
    return fract((st - diff * zoom) - diff * centerDist * centerDist * distort);
}

vec4 aberration(vec2 st) {
    // aberration, borrowed from lens
    vec2 diff = vec2(0.5) - st;
    if (aspectLens) {
        diff = vec2(0.5 * aspectRatio, 0.5) - vec2(st.x * aspectRatio, st.y);
    }
    
    float centerDist = length(diff);

    float aberrationOffset = map(effectAmt, 0.0, 100.0, 0.0, 0.05) * centerDist * PI * 0.5;

    float redOffset = mix(clamp(st.x + aberrationOffset, 0.0, 1.0), st.x, st.x);
    vec4 red = texture(mixerTex, vec2(redOffset, st.y));

    vec4 green = texture(mixerTex, st);

    float blueOffset = mix(st.x, clamp(st.x - aberrationOffset, 0.0, 1.0), st.x);
    vec4 blue = texture(mixerTex, vec2(blueOffset, st.y));

    return vec4(red.r, green.g, blue.b, 1.0);
}


// from https://github.com/spite/Wagner/blob/master/fragment-shaders/cga-fs.glsl
// cga

vec3 cga(vec4 color, vec2 st) {
	float amt = map(effectAmt, 0.0, 100.0, 0.0, 5.0);
    if (amt < 0.01) {
        return color.rgb;
    }
    float pixelDensity = amt;
	float size = 2. * pixelDensity;
	float dSize = 2. * size;

	float amount = resolution.x / size;
	float d = 1.0 / amount;
	float ar = resolution.x / resolution.y;
	float sx = floor( st.x / d ) * d;
	d = ar / amount;
	float sy = floor( st.y / d ) * d;

	vec4 base = texture( mixerTex, vec2( sx, sy ) );

	float lum = .2126 * base.r + .7152 * base.g + .0722 * base.b;
	float o = floor( 6. * lum );

	vec3 c1;
	vec3 c2;
	
	vec3 black = vec3( 0. );
	vec3 light = vec3( 85., 255., 255. ) / 255.;
	vec3 dark = vec3( 254., 84., 255. ) / 255.;
	vec3 white = vec3( 1. );

	/*dark = vec3( 89., 255., 17. ) / 255.;
	light = vec3( 255., 87., 80. ) / 255.;
	white = vec3( 255., 255., 0. ) / 255.;*/

	/*light = vec3( 85., 255., 255. ) / 255.;
	dark = vec3( 255., 86., 80. ) / 255.;*/

	if( o == 0. ) { c1 = black; c2 = c1; }
	if( o == 1. ) { c1 = black; c2 = dark; }
	if( o == 2. ) { c1 = dark;  c2 = c1; }
	if( o == 3. ) { c1 = dark;  c2 = light; }
	if( o == 4. ) { c1 = light; c2 = c1; }
	if( o == 5. ) { c1 = light; c2 = white; }
	if( o == 6. ) { c1 = white; c2 = c1; }

	if( mod( gl_FragCoord.x, dSize ) > size ) {
		if( mod( gl_FragCoord.y, dSize ) > size ) {
			base.rgb = c1;
		} else {
			base.rgb = c2;	
		}
	} else {
		if( mod( gl_FragCoord.y, dSize ) > size ) {
			base.rgb = c2;
		} else {
			base.rgb = c1;		
		}
	}

	return base.rgb;

}
// end cga


// https://github.com/spite/Wagner/blob/master/fragment-shaders/dot-screen-fs.glsl
// dot screen
/**
 * @author alteredq / http://alteredqualia.com/
 *
 * Dot screen shader
 * based on glfx.js sepia shader
 * https://github.com/evanw/glfx.js
 */



float pattern(vec2 st) {
    vec2 center = .5 * resolution;
    float angle = 1.57;
    float scale = dotsAmt;

	float s = sin( angle ), c = cos( angle );
	vec2 tex = st * resolution - center;
	vec2 point = vec2( c * tex.x - s * tex.y, s * tex.x + c * tex.y ) * scale;
	return ( sin( point.x ) * sin( point.y ) ) * 4.0;
}

vec3 dots(vec4 color, vec2 st) {	
    if (dotsAmt < 1.0) {
        return color.rgb;
    }
	float average = (color.r + color.g + color.b) / 3.0;
	return vec3(average * 10.0 - 5.0 + pattern(st)) * color.rgb;
}
// end dots

vec3 effect(vec2 uv, float blend) {
    // Aberration and distortion are always on
    vec4 color = vec4(1.0);

	if (effectType == 0) {
        color = texture(mixerTex, uv);
    } else if (effectType == 1) {
        color = aberration(uv);
	} else if (effectType == 2) {
        // cga
		color.rgb = cga(color, uv);
	} else if (effectType == 3) {
        // pixels
		color.rgb = pixellate(uv, effectAmt);
	} else if (effectType == 4) {
		// subpixel
		color.rgb = subpixel(uv, effectAmt);
	} else if (effectType == 5) {
		// tracking
		color.rgb = tracking(color, uv);
    }

	return color.rgb;
}

void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
	
    vec2 st = gl_FragCoord.xy / resolution;
	st.y = 1.0 - st.y;

    float blend = periodicFunction(time - offset(st));

    vec2 uv = distortionCoords(st);

    color = texture(mixerTex, uv);


    if (effectAmt > 0.0) {
	    color.rgb = effect(uv, blend);
    }
	color.rgb = dots(color, uv);

	if (scanlinesAmt > 0.0) {
		//color.rgb = scanlines(color.rgb, vTexCoord);
        color.rgb = scanlines(color.rgb, uv);
	}

	if (fabricSize > 0.0) {
		//color.rgb = fabric(color.rgb, vTexCoord + blend * 0.05);
        color.rgb = fabric(color.rgb, uv + blend * 0.05);
	}

    color.rgb = snow(color, uv);
    
    color.rgb = brightnessContrast(color.rgb);
    color.rgb = saturate(color);

    if (vignetteAmt < 0.0) {
        color.rgb = mix(color.rgb * 1.0 - pow(length(0.5 - uv) * 1.125, 2.0), color.rgb, map(vignetteAmt, -100.0, 0.0, 0.0, 1.0));
    } else {
        color.rgb = mix(color.rgb, 1.0 - (1.0 - color.rgb * 1.0 - pow(length(0.5 - uv) * 1.125, 2.0)), map(vignetteAmt, 0.0, 100.0, 0.0, 1.0));
    } 

    fragColor = color;
}
