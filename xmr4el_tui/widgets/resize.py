"""Resizable left column (Dashboard explorer, Run monitor left column): width clamping and border dragging."""


def column_width(want, total, share, minimum, max_share, right_min):
    """Left column cells on a `total`-cell screen: `want` (None: `share` of it), at least `minimum`, at most
    `max_share` of the screen and never leaving the right side under `right_min` (the minimum wins on tiny screens)."""
    want = round(total * share) if want is None else want
    return max(minimum, min(want, int(total * max_share), total - right_min))


class DragRightBorder:
    """Dragging the widget's right border calls screen.set_left_width(cells from the widget's left edge)."""
    dragging = False

    def on_mouse_down(self, event):
        if event.x >= self.outer_size.width - 1:
            self.dragging = True
            self.capture_mouse()
            self.screen.clear_selection()  # a resize is not a text selection
            event.stop()

    def on_mouse_move(self, event):
        if self.dragging:
            self.screen.set_left_width(event.screen_x - self.region.x + 1)

    def on_mouse_up(self, _):
        if self.dragging:
            self.dragging = False
            self.release_mouse()
