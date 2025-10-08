uniform sampler2D ltexim;
uniform ivec2 pixel_coords;

out vec4 fragColor;

void main() {
	fragColor = texelFetch(ltexim, pixel_coords, 0);
}
