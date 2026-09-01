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
    _apply_dark_palette(app)
    window = MainWindow(args.server)
    window.show()
    return app.exec()


def _apply_dark_palette(app) -> None:
    """Dark theme for the whole window.

    Not decoration: the video panes are painted dark, and the panels around
    them inherit the platform's light theme, so the labels — styled for a dark
    background — came out light grey on white and were unreadable. Setting the
    palette here makes one theme apply to everything, rather than having each
    widget carry a colour that only works in one of them.
    """
    from PySide6.QtGui import QColor, QPalette

    p = QPalette()
    window, base, text = QColor(21, 26, 32), QColor(13, 16, 19), QColor(221, 229, 238)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, window)
    p.setColor(QPalette.ColorRole.Button, QColor(27, 34, 42))
    p.setColor(QPalette.ColorRole.ToolTipBase, window)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.ToolTipText,
                 QPalette.ColorRole.BrightText):
        p.setColor(role, text)
    p.setColor(QPalette.ColorRole.Highlight, QColor(79, 184, 255))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(13, 16, 19))
    disabled = QColor(120, 134, 150)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(p)
    app.setStyleSheet(
        "QToolTip { color: #dde5ee; background: #151a20; border: 1px solid #28313c; }"
        "QPushButton { background: #1b222a; border: 1px solid #28313c;"
        "              border-radius: 4px; padding: 4px 10px; }"
        "QPushButton:hover { border-color: #4fb8ff; }"
        "QPushButton:disabled { color: #5b6673; border-color: #222a33; }"
        # Scoped to the panels, not every QFrame: an unscoped rule boxes each
        # label row too, making read-only text look like an input field.
        "QFrame#panel { border: 1px solid #28313c; border-radius: 6px; }"
        "QSpinBox, QDoubleSpinBox { background: #0d1013; border: 1px solid #28313c;"
        "                           border-radius: 4px; padding: 2px 4px; }"
    )


if __name__ == "__main__":
    raise SystemExit(main())
