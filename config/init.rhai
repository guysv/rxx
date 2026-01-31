/// Initialize script state (event-handler style).
/// Variables defined here persist for later event handlers.
fn init() {
    print("Rhai script loaded: init() ran.");
}

/// Draw event: called every frame. Use draw_line to add lines to the screen.
fn draw() {
    // Test line from top-left area diagonally
    draw_line(10.0, 10.0, 500.0, 200.0);
}
