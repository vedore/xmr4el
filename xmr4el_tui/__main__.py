"""xmr4el-tui: browse experiment runs under <root>/outputs/runs."""
import argparse
from pathlib import Path

from xmr4el_tui.app import XMR4ELApp


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xmr4el-tui", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="repository root holding outputs/runs (default: current directory)")
    args = parser.parse_args(argv)
    XMR4ELApp(args.root.resolve()).run()


if __name__ == "__main__":
    main()
