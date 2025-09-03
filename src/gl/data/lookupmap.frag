in  vec4 f_color;
out vec4 fragColor;

void main() {
	if (f_color.a > 0.0) {
		fragColor = f_color;
	} else {
		discard;
	}
}
