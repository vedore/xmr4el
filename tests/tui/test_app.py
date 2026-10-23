import asyncio

import pytest

pytest.importorskip("textual")

from xmr4el_tui.app import XMR4ELApp  # noqa: E402


def test_app_starts(tmp_path):
    async def run():
        app = XMR4ELApp(tmp_path)
        async with app.run_test(size=(80, 24)):
            assert app.root == tmp_path
    asyncio.run(run())
