uniform sampler2D tex;
uniform sampler2D ltex;
uniform vec2 ltexreg; // lookup texture region normalization vector

in  vec2  f_uv;
in  vec4  f_color;
in  float f_opacity;

out vec4  fragColor;

vec3 linearTosRGB(vec3 linear) {
    vec3 lower = linear * 12.92;
    vec3 higher = 1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055;

    // Use smoothstep for a smoother transition
    vec3 transition = smoothstep(vec3(0.0031308 - 0.00001), vec3(0.0031308 + 0.00001), linear);
    
    return mix(lower, higher, transition);
}

void main() {
    // Convert normalized UV coordinates to pixel coordinates for texelFetch
    ivec2 tex_size = textureSize(tex, 0);
    ivec2 pixel_coord = ivec2(f_uv * vec2(tex_size));
    
    // Clamp to texture bounds to prevent out-of-bounds access
    pixel_coord = clamp(pixel_coord, ivec2(0), tex_size - ivec2(1));
    
    vec4 texel = texelFetch(tex, pixel_coord, 0);
    texel = vec4(linearTosRGB(texel.rgb), texel.a); // Convert to linear space
    texel.rg = texel.rg * ltexreg;
    texel.rg = texel.rg + vec2(0.001953125, 0.001953125);
    
    if (texel.a > 0.0) { // Non-transparent pixel
        // Convert lookup texture coordinates to pixel coordinates for texelFetch
        ivec2 ltex_size = textureSize(ltex, 0);
        ivec2 ltex_pixel_coord = ivec2(texel.rg * vec2(ltex_size));
        
        // Clamp to lookup texture bounds
        ltex_pixel_coord = clamp(ltex_pixel_coord, ivec2(0), ltex_size - ivec2(1));
        
        vec4 lt_texel = texelFetch(ltex, ltex_pixel_coord, 0);
        fragColor = vec4(
            mix(lt_texel.rgb, f_color.rgb, f_color.a),
            lt_texel.a * f_opacity
        );
    } else {
        fragColor = vec4(
            mix(texel.rgb, f_color.rgb, f_color.a),
            texel.a * f_opacity
        );
    }
}
