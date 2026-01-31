/// Initialize script state (event-handler style).
/// session: width, height, offset_x, offset_y (same as draw).
/// Variables defined here persist for later event handlers.
fn init(session) {
    let global_session = session;
}

/// Draw event: called every frame. Use draw_line to add lines to the screen.
/// session: width, height, offset_x, offset_y
fn draw(session) {
    draw_line(10.0, 10.0, 500.0, 200.0);
}
