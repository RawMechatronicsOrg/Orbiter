# CadQuery sources

The parts of the v3 mechanics that were modelled in code rather than in Fusion. Both scripts need Python with
[CadQuery](https://github.com/CadQuery/cadquery) 2.x and write STEP files to an `export/` folder next to them;
the Fusion model was built from these exports.

| File | What it makes |
|---|---|
| [`gliding_arm.py`](gliding_arm.py) | the gliding tube's piston cap and wedge, the OrbitArm seal and its lock knob, the male and female 20 × 20 hinges, the swing tube, the phone clamp hinge block with its pivot bolt, knob and nut |
| [`gliding_arm_export_stl.py`](gliding_arm_export_stl.py) | the printable gliding-arm parts as STL in their print orientation |
| [`electronics_box.py`](electronics_box.py) | the electronics box, its lid and the U-shaped rail clamp, modelled in the machine's coordinates (Y up); it also writes print-oriented STL |

`gliding_arm_export_stl.py` imports `gliding_arm.py` as `gliding_arm`: rename it back to `export_stl.py` or run it
from this folder. The comments with dates record the changes asked for during the design.
