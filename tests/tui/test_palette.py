"""Palettes (theme.py, tui.md §4 Style): every palette readable (WCAG AA), a switch reaches the Rich styles and the
Textual theme while the app runs, the choice is kept in outputs/tui/settings.json, offered on the Dashboard only."""
import asyncio
import json

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from xmr4el_tui import fmt, theme  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.widgets.explorer import Explorer  # noqa: E402

READ = ("text", "text-muted", "accent", "success", "warning", "error", "info", "delta", "spec-differs")


@pytest.fixture(autouse=True)
def default_palette():
    """A test that switches palettes leaves the default active for the next one."""
    yield
    theme.use(theme.DEFAULT)


@pytest.mark.parametrize("name", list(theme.PALETTES))
def test_palette_is_readable(name):
    p = theme.PALETTES[name]
    assert set(p) == set(theme.PALETTES[theme.DEFAULT]), "every palette defines every token"
    low = {k: round(theme.contrast(p[k], p["surface"]), 2) for k in READ if theme.contrast(p[k], p["surface"]) < theme.AA}
    assert not low, f"under 4.5:1 on the surface: {low}"
    assert theme.contrast(p["on-accent"], p["accent"]) >= theme.AA
    assert theme.contrast(p["text"], p["text-selection"]) >= theme.AA
    assert theme.contrast(p["text"], p["cursor-row"]) >= theme.AA
    assert min(theme.contrast(p["on-beam"], b) for b in p["beams"]) >= theme.AA
    assert p["error"] != p["accent"] and p["warning"] != p["accent"]


def test_use_switches_in_place_and_lazy_styles_follow():
    tokens = theme.TOKENS
    before = fmt.STATE_STYLE["FAILED"]
    assert theme.use("Classic") == "Classic" and theme.TOKENS is tokens
    assert theme.color("accent") == theme.PALETTES["Classic"]["accent"]
    assert fmt.STATE_STYLE["FAILED"] == theme.style("error") != before
    assert theme.use("no such palette") == theme.DEFAULT


def test_settings_round_trip(tmp_path):
    assert theme.load(tmp_path) == theme.DEFAULT, "no settings file"
    path = tmp_path / theme.SETTINGS
    path.parent.mkdir(parents=True)
    for bad in ("{not json", "[1, 2]", json.dumps({"palette": "Nope"})):
        path.write_text(bad)
        assert theme.load(tmp_path) == theme.DEFAULT
    path.write_text(json.dumps({"other": 1}))
    theme.save(tmp_path, "Notebook light")
    assert json.loads(path.read_text()) == {"other": 1, "palette": "Notebook light"}, "other keys kept"
    assert theme.load(tmp_path) == "Notebook light"


def test_switch_in_the_app(tmp_path):
    make_run(tmp_path, "alpha", tree_id="a" * 12, tree_here=True)
    make_run(tmp_path, "beta", stamp="2026-10-21_10-00-00", status="failed", tree_id=None, metrics={"dev": None})

    def label_colours(app):
        tree = app.screen.query_one(Explorer)
        return {str(s.style) for n in tree.nodes.values() for s in n.label.spans}

    async def go():
        app = XMR4ELApp(tmp_path)
        async with app.run_test(size=(120, 36)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.theme == "xmr4el-notebook"
            titles = [c.title for c in app.get_system_commands(app.screen)]
            assert [t for t in titles if t.startswith("Palette")] == [f"Palette: {n}" for n in theme.PALETTES]
            assert "Theme" not in titles
            app.set_palette("Classic")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.theme == "xmr4el-classic" and theme.color("accent") == theme.PALETTES["Classic"]["accent"]
            assert theme.PALETTES["Classic"]["error"] in " ".join(label_colours(app)), "explorer relabelled"
            assert theme.PALETTES["Notebook"]["error"] not in " ".join(label_colours(app))
            await pilot.press("n")  # New Experiment: no palette commands above the Dashboard
            await pilot.pause()
            assert not [c for c in app.get_system_commands(app.screen) if c.title.startswith("Palette")]
        assert theme.load(tmp_path) == "Classic", "kept for the next start"
        again = XMR4ELApp(tmp_path)
        assert again.palette == "Classic"

    asyncio.run(go())
