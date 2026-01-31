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

/// Draw event: called every frame. Use draw_line to add lines, draw_text to add text.
/// Use session (set in init) for width, height, offset_x, offset_y.
fn draw() {
    draw_line(10.0, 10.0, 500.0, 200.0);
    draw_text(50.0, 50.0, "Hello from script");
}

fn view_added(view_id) {
    print(`view added: ${view_id}`);
}

fn view_removed(view_id) {
    print(`view removed: ${view_id}`);
}