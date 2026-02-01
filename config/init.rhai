fn init(session_in) {
    let session = session_in;
    let lookup_views = #{};
    register_command("lt/on", "Add active view to lookup set");
    register_command("lt/off", "Remove active view from lookup set");
    print(`script initialized`);
}

fn draw() {
    let line_height = 18.0;
    let green = rgb8(0, 255, 0);
    for view in session.views() {
        let key = `${view.id}`;
        if key in lookup_views {
            let x = session.offset_x + view.offset.x;
            let y = session.offset_y + view.offset.y - 2.0 * line_height;
            draw_text(vec2(x, y), "LUT", green);
        }
    }
}

fn view_added(view_id) {
}

fn view_removed(view_id) {
    info(`view removed: ${view_id}`);
    lookup_views.remove(`${view_id}`);
}

fn cmd_lt_on(args) {
    let vid = session.active_view_id;
    lookup_views[`${vid}`] = true;
    let view = session.view(vid);
    print(`view: ${view.id}, offset: ${view.offset.x}, ${view.offset.y}`);
}

fn cmd_lt_off(args) {
    let vid = session.active_view_id;
    lookup_views.remove(`${vid}`);
    print(`view removed from lookup: ${vid}`);
}