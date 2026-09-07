"""The guide's banner: the stage, the instruction and the counts, in type
sized for an operator standing at the bench, not sitting at the desk.

Colour carries the message before the words do — green: keep doing that;
amber: change something; red: a switch or a re-do stands in the way;
blue: done, ready. The words are `guide.Prompt`'s; this only shows them.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .guide import STAGES, Prompt

#: Background and text per tone. Dark grounds: the eye views beside the
#: banner are video, and a bright block next to them would pull the eye.
_TONES = {
    "go": ("#0f3d1f", "#8ff0a4", "#5fd07f"),
    "adjust": ("#4a3406", "#ffd166", "#e0b040"),
    "stop": ("#4a1010", "#ff8a80", "#e06060"),
    "done": ("#0e2c4a", "#8fd3ff", "#5fb0ff"),
}
_STEP_NAMES = {"left": "LEFT", "right": "RIGHT", "pair": "PAIR", "plane": "LASER",
               "readout": "READOUT", "check": "CHECK"}


class GuideBanner(QFrame):
    """Shows a `Prompt`; the buttons speak for the operator."""

    back_requested = Signal()
    next_requested = Signal()
    #: The guide switched on or off. Off, the banner shrinks to its switch.
    toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("guide")
        self._prompt: Prompt | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 10)
        root.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.title = QLabel("CALIBRATION GUIDE")
        self.title.setStyleSheet("font-size:15px; font-weight:600; letter-spacing:2px;")
        top.addWidget(self.title)
        top.addStretch(1)
        self.steps = QLabel()
        self.steps.setTextFormat(Qt.TextFormat.RichText)
        self.steps.setStyleSheet("font-size:13px; letter-spacing:1px;")
        top.addWidget(self.steps)
        self.back = QPushButton("◀ Back")
        self.back.setToolTip("Redo the previous stage. The guide then only moves forward, "
                             "as each stage is done.")
        self.back.clicked.connect(self.back_requested)
        self.forward = QPushButton("Next ▶")
        self.forward.setToolTip("Skip to the next stage — the readout is optional, and a "
                                "stage may be good enough for today.")
        self.forward.clicked.connect(self.next_requested)
        self.enabled = QCheckBox("guide")
        self.enabled.setChecked(True)
        self.enabled.setToolTip("Step-by-step calibration with the instruction in large type: "
                                "each lens on its own, the pair, the laser sheet, a scan check. "
                                "Off, the eyes are paired as usual and nothing is drawn on them.")
        self.enabled.toggled.connect(self._toggle)
        for b in (self.back, self.forward, self.enabled):
            top.addWidget(b)
        root.addLayout(top)

        self.action = QLabel("…")
        self.action.setWordWrap(True)
        self.action.setStyleSheet("font-size:34px; font-weight:700;")
        root.addWidget(self.action)

        self.detail = QLabel("")
        self.detail.setStyleSheet("font-size:15px; font-family:Consolas;")
        root.addWidget(self.detail)
        self._paint_tone("adjust")

    # ── content ───────────────────────────────────────────────────────────

    def set_prompt(self, prompt: Prompt) -> None:
        if prompt == self._prompt:
            return
        prev = self._prompt
        self._prompt = prompt
        self.title.setText(prompt.title)
        self.action.setText(prompt.action)
        self.detail.setText(prompt.detail)
        if prev is None or prev.stage != prompt.stage:
            self.steps.setText(_steps_html(prompt.stage))
        if prev is None or prev.tone != prompt.tone:
            self._paint_tone(prompt.tone)

    def _paint_tone(self, tone: str) -> None:
        bg, fg, dim = _TONES.get(tone, _TONES["adjust"])
        self.setStyleSheet(
            f"QFrame#guide {{ background: {bg}; border: 1px solid {dim}; border-radius: 8px; }}"
            f"QFrame#guide QLabel {{ color: {fg}; }}"
            f"QFrame#guide QCheckBox {{ color: {fg}; }}")

    def _toggle(self, on: bool) -> None:
        for w in (self.action, self.detail, self.steps, self.back, self.forward):
            w.setVisible(on)
        if not on:
            self.title.setText("CALIBRATION GUIDE — off")
            self._prompt = None
        self.toggled.emit(on)


def _steps_html(current: str) -> str:
    parts = []
    for i, s in enumerate(STAGES, 1):
        name = f"{i} {_STEP_NAMES[s]}"
        parts.append(f"<b><u>{name}</u></b>" if s == current else name)
    return " &nbsp;·&nbsp; ".join(parts)
