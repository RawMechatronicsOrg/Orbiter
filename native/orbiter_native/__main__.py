"""Entry point: `orbiter-native` (or `python -m orbiter_native`)."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import DEFAULT_SERVER


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="orbiter-native",
        description=(
            "Native CV workbench for the Orbiter binocular pair. Reads its "
            "configuration from the Orbiter server (which camera is which eye, "
            "per-eye orientation, board params) and pulls frames straight from "
            "camserver."
        ),
    )
    parser.add_argument(
        "--server", default=DEFAULT_SERVER,
        help=f"Orbiter server base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Imported here, not at module scope, so `--help` does not pay for Qt.
    from PySide6.QtWidgets import QApplication

    from .app import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Orbiter Native")
    window = MainWindow(args.server)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
