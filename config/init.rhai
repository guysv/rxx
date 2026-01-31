/// Initialize script state (event-handler style).
/// session: width, height, offset_x, offset_y (same as draw).
/// Variables defined here persist for later event handlers.
///
/// Use register_command(name, help) to register custom commands.
/// Then define a handler function named cmd_<name>(args) to handle the command.
fn init(session_in) {
    let session = session_in;
    for view in session.views() {
        print(`view on init: ${view.id}`);
    }

    // Register custom script commands
    register_command("greet", "Print a greeting message");
}

/// Handler for :greet command
fn cmd_greet(args) {
    if args.len() > 0 {
        print(`Hello, ${args[0]}!`);
    } else {
        print("Hello from script!");
    }
}

/// Draw event: called every frame. Use vec2(x,y), rgb8(r,g,b) or rgb8(r,g,b,a) (0..255).
/// draw_line(p1, p2) or draw_line(p1, p2, color); draw_text(pos, text) or draw_text(pos, text, color).
fn draw() {
    draw_line(vec2(10.0, 10.0), vec2(500.0, 200.0));
    draw_line(vec2(20.0, 20.0), vec2(400.0, 100.0), rgb8(255, 0, 0));  // red line
    draw_text(vec2(50.0, 50.0), "Hello from script");
    draw_text(vec2(50.0, 70.0), "Colored text", rgb8(0, 255, 0));  // green text
}

fn view_added(view_id) {
    print(`view added: ${view_id}`);
}

fn view_removed(view_id) {
    print(`view removed: ${view_id}`);
}