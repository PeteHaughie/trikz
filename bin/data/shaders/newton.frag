#version 300 es
precision highp float;
precision highp int;

uniform float time;
uniform float seed;
uniform vec2 resolution;
uniform float offset;
uniform float centerX;
uniform float centerY;
uniform float zoomAmt;
uniform float speed;
uniform float rotation;
uniform int symmetry; // 2 3 4 5
uniform int paletteMode;
uniform vec3 paletteOffset;
uniform vec3 paletteAmp;
uniform vec3 paletteFreq;
uniform vec3 palettePhase;
uniform bool cycleColors;
out vec4 fragColor;


#define PI 3.14159265359
#define TAU 6.28318530718
#define aspectRatio resolution.x / resolution.y

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 rotate2D(vec2 st, float rot) {
    rot = map(rot, 0.0, 360.0, 0.0, 2.0);
    float angle = rot * PI;
    // angle = PI * u_time * 2.0; // animate rotation
    st -= vec2(0.5 * aspectRatio, 0.5);
    st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
    st += vec2(0.5 * aspectRatio, 0.5);
    return st;
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

// newton 

#define NUM_STEPS   40
#define X_OFFSET    0.5

#define NEWTON      0
#define HALLEY      1
#define HOUSEHOLDER 2
#define METHOD      NEWTON

//const float PI = 3.1415926535897932384626433832795;
const float E = 2.7182818284590452353602874713527;

#define complex vec2

const complex pi = complex(PI, 0.0);
const complex e = complex(E, 0.0);
const complex i = complex(0.0, 1.0);
const complex c_nan = complex(10000.0, 20000.0);

bool error;

#define c_abs(a) length(a)

complex c_mul(complex a, complex b)
{
    return complex(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

complex c_div(complex a, complex b)
{
    float denominator = dot(b,b);
    if (denominator == 0.0) {
        error = true;
        return c_nan;
    }
    return complex(dot(a,b)/denominator, (a.y*b.x-a.x*b.y)/denominator);
}

float c_arg(complex a)
{
    if (a == complex(0, 0)) {
        error = true;
    }
    return atan(a.y, a.x);
}

complex c_exp(complex z)
{
    return complex(exp(z.x)*cos(z.y), exp(z.x)*sin(z.y));
}

complex c_pow(complex z, complex w)
{
    /*
    z = r*e^(i*t)
    w = c+di

    z^w = r^c * e^(-d*t) * e^(i*c*t + i*d*log(r))
    */

    float r = c_abs(z);
    float theta = c_arg(z);

    float r_ = pow(r, w.x);
    float theta_ = theta * w.x;

    if (w.y != 0.0)
    {
        r_ *= exp(-w.y*theta);
        theta_ += w.y*log(r);
    }

    return complex(r_*cos(theta_), r_*sin(theta_));
}

complex c_sqrt(complex z)
{
    float r = c_abs(z);

    float a = sqrt((r + z.x)/2.0);
    float b = sqrt((r - z.x)/2.0);

    if (z.y < 0.0) b = -b;

    return complex(a, b);
}

complex c_inv(complex z)
{
    return c_div(complex(1.0, 0), z);
}

complex c_log(complex z)
{
    return complex(log(c_abs(z)), c_arg(z));
}

float cosh2(float x)
{
    float e = exp(x);
    return 0.5 * (e + 1.0/e);
}

float sinh2(float x)
{
    float e = exp(x);
    return 0.5 * (e - 1.0/e);
}

complex c_sin(complex z)
{
    return complex(sin(z.x) * cosh2(z.y), cos(z.x) * sinh2(z.y));
}

complex c_cos(complex z)
{
    return complex(cos(z.x) * cosh2(z.y), -sin(z.x) * sinh2(z.y));
}

complex c_sinh2(complex z)
{
    return complex(cos(z.y) * sinh2(z.x), sin(z.y) * cosh2(z.x));
}

complex c_cosh2(complex z)
{
    return complex(cos(z.y) * cosh2(z.x), sin(z.y) * sinh2(z.x));
}

complex c_tanh(complex z)
{
    return c_div(
        c_exp(z) - c_exp(-z),
        c_exp(z) + c_exp(-z)
    );
}

complex c_tan(complex z)
{
    return c_mul(complex(0, -1), c_tanh(c_mul(complex(0, 1), z)));
}

complex c_cot(complex z)
{
    return c_inv(c_tan(z));
}

complex c_sec(complex z)
{
    return c_inv(c_cos(z));
}

complex c_csc(complex z)
{
    return c_inv(c_sin(z));
}

complex c_cis(complex z)
{
    return c_cos(z) + c_mul(i, c_sin(z));
}

complex c_asin(complex z)
{
    return c_mul(
        complex(0, -1.0),
        c_log(
            c_mul(complex(0, 1.0), z) + c_sqrt(complex(1.0, 0) - c_mul(z, z))
        ));
}

complex c_acos(complex z)
{
    return complex(PI/2.0, 0) - c_asin(z);
}

complex c_atan(complex z)
{
    return c_mul(
        complex(0, 0.5),
              c_log(complex(1.0, 0) - c_mul(complex(0, 1.0), z))
            - c_log(complex(1.0, 0) + c_mul(complex(0, 1.0), z))
        );
}

complex c_acot(complex z)
{
    return c_atan(c_inv(z));
}


complex c_asec(complex z)
{
    return c_acos(c_inv(z));
}

complex c_acsc(complex z)
{
    return c_asin(c_inv(z));
}




complex m;
uniform complex t;
uniform complex f;

bool isbad(float v) {
    if (!(v == 0.0 || v < 0.0 || 0.0 < v)) return true;
    if (v >= 100000.0) return true;

    return false;
}

bool isbad(complex z) {
    return isbad(z.x) || isbad(z.y) || z == c_nan;
}

float newton(vec2 st) {
    complex z, z_;

    float speedy = map(speed, 0.0, 100.0, 0.0, 1.0);
    float s = mix(speedy * 0.05, speedy * 0.125, speedy);

    float off = map(offset, 0.0, 100.0, 0.0, 1.0);
    float _offsetX = map(sin(off * TAU), -1.0, 1.0, 0.125, 1.5);
    float _offsetY = map(cos(off * TAU), -1.0, 1.0, 0.125, 1.5);

    vec2 cen = vec2(sin(time * TAU) * s + _offsetX, cos(time * TAU) * s + _offsetY);

    st = rotate2D(st, rotation);
    st = (st - .5) * map(zoomAmt, 0.0, 1.0, 1.0, 0.1);

    z.x = st.x + map(centerX, -100.0, 100.0, 1.0, -1.0);
    z.y = st.y + map(centerY, -100.0, 100.0, 1.0, -1.0);

    int steps;
    for (int step=0; step<NUM_STEPS; step++) {
        steps = step;
        error = false;
#if METHOD == NEWTON
        complex f = c_pow(z, complex(3.0, 0.0)) + complex(1.0, 0.0);
        complex df = c_mul(complex(3.0, 0.0), c_pow(z, complex(2.0, 0.0)));

        complex delta = c_div(f, df);
/*
#elif METHOD == HALLEY
        complex f_ = (%%f%%);
        complex df_ = (%%df%%);
        complex ddf_ = (%%ddf%%);

        complex delta = c_div(
            c_mul(complex(2.0, 0.0), c_mul(f_, df_)),
            c_mul(complex(2.0, 0.0), c_mul(df_, df_)) - c_mul(f_, ddf_)
        );

#else
        complex f_ = c_sin(z);
        complex df_ = c_cos(z);
        complex ddf_ = c_sqrt(z);

        complex delta = c_mul(
            c_div(f_, df_),
            complex(1.0, 0.0) + c_div(
                c_mul(f_, ddf_),
                c_mul(complex(2.0, 0.0), c_mul(df_, df_))
            )
        );
*/
#endif

        z_ = z - delta * cen;
        //z_ = z - delta;

        if (isbad(delta) || error) {
            return 0.0;
        }

        if (distance(z, z_) < 0.001)
            break;

        z = z_;
    }
    /*

    float hue, saturation, v;

    // Choose a hue with the same angle as the argument 
    hue = 0.5 + c_arg(z)/(PI*2.0);

    // Saturate roots the closer to 0 they are 
    saturation = 0.59/(sqrt(c_abs(z)));

    // Make roots close to 0 white 
    if (c_abs(z) < 0.1)
        saturation = 0.0;

    // Darken based on the number of steps taken 
    v = 0.95 * max(1.0-float(steps)*0.025, 0.00);

    // Make huge roots black 
    if (c_abs(z) > 100.0)
        v = 0.0;

    vec3 c = hsv2rgb(vec3(hue, saturation, v));
    */
    return float(steps) / 40.0;
}

// end newton


// newton 2

bool isNan(float val) {
    return (val <= 0.0 || 0.0 <= val) ? false : true;
}

 bool isInf(float val) {
    return (val != 0.0 && val * 2.0 == val) ? true : false;
}

//---------------------------------------------------------
// Shader:   Newton5Fractal.glsl  by tholzer
// Newton fractal with 3 or 4 or 5  symmetry (change #define NEWTON #).
// Press mouse button to change formula constants.
//           v1.0  2015-04-14
//           v1.1  2017-04-06  define NEWTON added
// tags:     2d, attractor, newton, fractal, complex, number
// info:     http://en.wikipedia.org/wiki/Newton_fractal
//---------------------------------------------------------

// from https://discourse.processing.org/t/newton-fractal-shader-sketch/24231

vec2 cinv(vec2 a) { 
    return vec2(a.x, -a.y) / dot(a, a); 
}

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); 
}

vec2 cdiv(vec2 a, vec2 b) {
    return cmul(a, cinv(b));
}

#define ITER 40

float newton2(vec2 st) {
    float speedy = map(speed, 0.0, 100.0, 0.0, 1.0);
    float s = mix(speedy * 0.05, speedy * 0.125, speedy);

    float off = map(offset, 0.0, 100.0, 0.0, 1.0);
    float _offsetX = map(sin(off * TAU), -1.0, 1.0, 0.125, 1.5);
    float _offsetY = map(cos(off * TAU), -1.0, 1.0, 0.125, 1.5);

    vec2 cen = vec2(sin(time * TAU) * s + _offsetX, cos(time * TAU) * s + _offsetY);

    st = rotate2D(st, rotation);

    st -= vec2(0.5 * aspectRatio, 0.5);
    st *= map(zoomAmt, 0.0, 100.0, 1.0, 0.1);

    vec2 z, z_;
    z.x = st.x + map(centerX, -100.0, 100.0, 1.0, -1.0);
    z.y = st.y + map(centerY, -100.0, 100.0, 1.0, -1.0);

    int iter = 0;

    for (int i = 0; i < ITER; i++) {
        iter = i;
        vec2 z2 = cmul(z, z);
        vec2 z3 = cmul(z2, z);
        vec2 z4 = cmul(z2, z2);
        vec2 z5 = cmul(z3, z2);

        if (symmetry == 2) {
            z_ = z - cdiv(z3 - 1.0, 3.0 * z2) * cen;  // original: z^3 - 1  / ( 3*z^2)
            //z_ = z - cdiv(z3 - 1.0 + cen.x, (3.0 + cen.y) * z2);  // original: z^3 - 1  / ( 3*z^2)
        } else if (symmetry == 3) {
            z_ = z - cdiv(z3 - 0.5, 0.5 * z2) * cen * 0.125;  // z^3 - my / (mx*z^2)
            //z_ = z - cdiv(z3 - 0.5 + cen.x, (0.5 + cen.y) * z2); // z^3 - my / (mx*z^2)
        } else if (symmetry == 4) {
            z_ = z - cdiv(z4 - 0.5, 0.5 * z3) * cen * 0.125;  // z^4 - my / (mx*z^3)
            //z_ = z - cdiv(z4 - 0.5 + cen.x, (0.5 + cen.y) * z3);  // z^4 - my / (mx*z^3)
        } else {
            z_ = z - cdiv(z5 - 0.5, 0.5 * z4) * cen * 0.125;  // z^5 - my / (mx*z^4)
            //z_ = z - cdiv(z5 - 0.5 + cen.x, (0.5 + cen.y) * z4);  // z^5 - my / (mx*z^4)
        }

        // alternate sin(z) implementation
        //z_ = z - cdiv(sin(z), cos(z)) * cen * 0.125;

        /*
        #if NEWTON==2
            z -= cdiv(z3 - 1.0, 3.0 * z2);                      // original: z^3 - 1  / ( 3*z^2)
        #elif NEWTON==3
            z -= cdiv(z3 - 0.5+0.05*iMouse.y, (0.5+0.01*iMouse.x) * z2);  // z^3 - my / (mx*z^2)
        #elif NEWTON==4
            z -= cdiv(z4 - 0.5+0.05*iMouse.y, (0.5+0.01*iMouse.x) * z3);  // z^4 - my / (mx*z^3)
        #elif NEWTON==5
            z -= cdiv(z5 - 0.5+0.05*iMouse.y, (0.5+0.1*iMouse.x) * z4);  // z^5 - my / (mx*z^4)
        #endif
        */
        
        if ( distance(z.x, z_.x) < 0.001 && distance(z.y, z_.y) < 0.001 ) {
            break;
        }

        z = z_;   
    }

    if (z.x == 0.0 || z.y == 0.0) {
        return 0.0;
    }

    return float(iter) / float(ITER);
}

// end newton 2


void main() {
    vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
    vec2 st = gl_FragCoord.xy / resolution.y;	

    //float blend = periodicFunction(time - offset(st));

    //float d = newton(st);
    //color.rgb = pal(d + time);

    float d = newton2(st);
    if (cycleColors) {
        color.rgb = pal(d + time);
    } else {
        color.rgb = pal(d);
    }


    fragColor = color;
}
