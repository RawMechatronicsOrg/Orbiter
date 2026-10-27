"""Write the printable parts of the gliding arm to export/print/ in their print orientation, on z=0.

  - piston_cap: lid down (its chamfer on the bed), skirt and insert up; the slot widens upward
    at 19 deg from vertical, so nothing overhangs;
  - piston_wedge: as built, wide face down, the nut pocket open to the bed (its roof bridges 8 mm);
  - seal: as built, flange down (its chamfer on the bed); the lock boss's teardrop points down;
  - lock_knob: outer face down, the head's pocket and its hex stub up.
Run: ../../../CondensateDrain/.venv/Scripts/python.exe export_stl.py
"""
from pathlib import Path

import cadquery as cq

import gliding_arm as g  # noqa: E402

OUT = Path(__file__).resolve().parent / 'export' / 'print'

PRINT = {
    'piston_cap': lambda: g.piston_cap().rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180),
    'piston_wedge': g.piston_wedge,
    'seal': g.seal,
    'lock_knob': lambda: g.lock_knob().rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180),
}

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in PRINT.items():
        shape = make()
        shape = shape.translate(cq.Vector(0, 0, -shape.BoundingBox().zmin))
        shape.exportStl(str(OUT / f'{name}.stl'), 0.01, 0.1)
        bb = shape.BoundingBox()
        print(f'{name}: vol {shape.Volume():.1f} mm3, {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f}')
