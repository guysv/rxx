fn init(session_in) {
    let session = session_in;
    let lookup_views = #{};
    register_command("lt/on", "Add active view to lookup set");
    print(`script initialized`);
}

fn draw() {
}

fn view_added(view_id) {
}

fn view_removed(view_id) {
    lookup_views.remove(view_id);
}

fn cmd_lt_on(args) {
    let vid = session.active_view_id;
    lookup_views[`${vid}`] = true;
    let view = session.view(vid);
    print(`view: ${view.id}, offset: ${view.offset.x}, ${view.offset.y}`);
}