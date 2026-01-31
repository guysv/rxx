/// Initialize script state (event-handler style).
/// session: width, height, offset_x, offset_y (same as draw).
/// Variables defined here persist for later event handlers.
fn init(session) {
    let global_session = session;
    for view in session.views() {
        print(`view on init: ${view.id}`);
    }
}

/// Draw event: called every frame. Use draw_line to add lines to the screen.
/// Use global_session (set in init) for width, height, offset_x, offset_y.
fn draw() {
    draw_line(10.0, 10.0, 500.0, 200.0);
}

fn view_added(view_id) {
    print(`view added: ${view_id}`);
}

fn view_removed(view_id) {
    print(`view removed: ${view_id}`);
}