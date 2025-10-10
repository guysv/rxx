uniform sampler2D tex;
uniform sampler2D ltex;
uniform sampler2D ltexim;
uniform uint lt_tfw;
// uniform vec2 ltexreg; // lookup texture region normalization vector

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

    vec4 imtexel = texelFetch(tex, pixel_coord, 0);
    vec4 imtexel_raw = imtexel;
    imtexel = vec4(linearTosRGB(imtexel.rgb), imtexel.a); // Convert to linear space
    // imtexel.rgb = imtexel.rgb + vec3(0.001953125, 0.001953125, 0.001953125);
    if (imtexel.a > 0.0) {
        ivec3 imtexel256 = ivec3(round(imtexel.rgb * vec3(255)));
        ivec2 imtex_uv = ivec2(
            imtexel256.r + (imtexel256.b >> 4) * 256,
            imtexel256.g + (imtexel256.b & 15) * 256
        );
        vec4 texel = texelFetch(ltexim, imtex_uv, 0);
        texel = vec4(linearTosRGB(texel.rgb), texel.a); // Convert to linear space
        // texel.rg = texel.rg * ltexreg;
        texel.rg = texel.rg + vec2(0.001953125, 0.001953125);
        
        if (texel.a > 0.0) { // Non-transparent pixel
            // Convert lookup texture coordinates to pixel coordinates for texelFetch
            ivec2 ltex_size = textureSize(ltex, 0);
            ivec2 ltex_pixel_coord = ivec2(floor(texel.rg * vec2(255)));
            
            // Clamp to lookup texture bounds
            // ltex_pixel_coord = clamp(ltex_pixel_coord, ivec2(0), ltex_size - ivec2(1));
            
            // Walk from right to left to find first non-zero alpha pixel
            vec4 lt_texel = vec4(0.0);
            int max_n = int(ltex_size.x) / int(lt_tfw);
            for (int n = max_n - 1; n > 0; n--) {
                ivec2 check_coord = ivec2(
                    n * int(lt_tfw) + ltex_pixel_coord.x,
                    ltex_pixel_coord.y
                );
                
                // Clamp to texture bounds
                if (check_coord.x >= 0 && check_coord.x < ltex_size.x && 
                    check_coord.y >= 0 && check_coord.y < ltex_size.y) {
                    vec4 sample = texelFetch(ltex, check_coord, 0);
                    if (sample.a > 0.0) {
                        lt_texel = sample;
                        break;
                    }
                }
            }
            fragColor = vec4(
                mix(lt_texel.rgb, f_color.rgb, f_color.a),
                lt_texel.a * f_opacity
            );
        } else {
            fragColor = imtexel_raw;
        }
    } else {
        // color is not mapped in ltexim, fall back to original
        fragColor = imtexel_raw;
    }
}
