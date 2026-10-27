"""Row rendering mixins for line-based widgets: full-row cursor, mouse text selection."""
from rich.segment import Segment
from textual.strip import Strip


class FullRow:
    """Paints the cursor style over the whole row (guides, padding, every column), not only the label or cells."""
    CURSOR = ""

    def render_line(self, y):
        strip = super().render_line(y)
        if not self.cursor_at(y):
            return strip
        return Strip(Segment.apply_style(list(strip), post_style=self.get_component_rich_style(self.CURSOR)),
                     strip.cell_length)


class Selectable:
    """Mouse drag selects text in a line-based widget, as Textual does for its text widgets (the app copies it on
    release). Each rendered line is remembered with its scroll x, so the selection is extracted from what was shown.
    ponytail: offsets are cells, so a wide character shifts the range; fine for the ASCII paths, ids and metrics."""
    ALLOW_SELECT = True  # Tree, OptionList and DataTable turn it off

    def __init__(self, *args, **kwargs):
        self.plain = {}  # virtual line -> (scroll x, text shown)
        super().__init__(*args, **kwargs)

    def render_line(self, y):
        strip = super().render_line(y)
        x, y = self.scroll_offset.x, self.scroll_offset.y + y
        self.plain[y] = (x, "".join(seg.text for seg in strip))
        span = self.text_selection.get_span(y) if self.text_selection else None
        if span is not None:
            start, end = max(0, span[0] - x), strip.cell_length if span[1] == -1 else min(strip.cell_length, span[1] - x)
            if start < end:
                style = self.screen.get_component_rich_style("screen--selection")
                strip = Strip.join([strip.crop(0, start), Strip(Segment.apply_style(
                    list(strip.crop(start, end)), post_style=style), end - start), strip.crop(end)])
        return strip.apply_offsets(x, y)

    def get_selection(self, selection):
        text = "\n".join(" " * self.plain[i][0] + self.plain[i][1] if i in self.plain else ""
                         for i in range(max(self.plain, default=-1) + 1))
        return "\n".join(line.rstrip() for line in selection.extract(text).split("\n")), "\n"
