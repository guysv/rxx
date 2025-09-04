uniform sampler2D tex;
uniform mat4 ortho;
uniform mat4 transform;
uniform uint lt_tfw;

out vec4 f_color;

vec3 linearTosRGB(vec3 linear) {
    vec3 lower = linear * 12.92;
    vec3 higher = 1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055;

    // Use smoothstep for a smoother transition
    vec3 transition = smoothstep(vec3(0.0031308 - 0.00001), vec3(0.0031308 + 0.00001), linear);
    
    return mix(lower, higher, transition);
}

vec3 sRGBToLinear(vec3 c) {
    // inverse of your linearTosRGB()
    bvec3 cut = lessThanEqual(c, vec3(0.04045));
    vec3 lower = c / 12.92;
    vec3 higher = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(higher, lower, vec3(cut));
}

void main() {
	// ivec2 tex_size = textureSize(tex, 0);
    // Calculate UV coordinates from vertex ID
	ivec2 uv = ivec2(gl_VertexID % int(lt_tfw), gl_VertexID / int(lt_tfw));
    
    // Sample texture at UV coordinates
	vec4 texel = texelFetch(tex, uv, 0);
    texel = vec4(linearTosRGB(texel.rgb), texel.a); // Convert to linear space
	if (texel.a <= 0) {
		f_color = vec4(0.0);
		gl_Position = vec4(-2.0);
		return;
	}

	ivec3 texel256 = ivec3(round(texel.rgb * vec3(255.0)));
	ivec2 imtex_uv = ivec2(
		texel256.r + (texel256.b >> 4) * 256 + 1,
		texel256.g + (texel256.b & 15) * 256 + 1
	);
    
    // Use the color value for gl_Position (like lookuptex.frag but in reverse)
    gl_Position = ortho * transform * vec4(imtex_uv, 0.0, 1.0);
    
    // Set f_color to UV coordinates
	vec3 intended_srgb = vec3(vec2(uv) / 255.0, 0.0);
    f_color = vec4(sRGBToLinear(intended_srgb), 1.0);
}
