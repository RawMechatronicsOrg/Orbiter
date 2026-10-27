"""Electronics box for the Orbiter Factory F0.7 Compact board (100 x 75), on MainRail's outer side
face (-Z, away from the elevation axis and OrbitArm) between the azimuth and the elevation drive.

Frame: the Orbiter document's world (mm, Y up), so the parts drop into Fusion at identity. MainRail
is the 25 x 25 x 1.5 tube along X: y -2.5..22.5, z -117.5..-92.5; the floor (the three feet) is at
y = -32.1. OrbitArm swings in the plane x 187..212, so the box stays at x <= 182.5; the CrossRail
diagonal passes 6 mm under its outer right corner.

  - board (export/orbiter_factory_f07.step, kicad-cli): upright, facing -Z, turned 180 deg in its
    plane so the USB-C power input J19 points up and the sensor/CAN/DSLR row runs along the bottom
    (world x = kx + 20, y = ky - 63.5); its back 10 mm off the box's back wall on four M4 standoffs
    (heat-set inserts);
  - box: open toward -Z, back wall flat on the rail's -Z face. Cable exits are notches open at the
    rim, closed by the lid: top wall for the AZ motor (over the rail to the base) and the ADXL,
    right wall for the EL motor, EL sensor, CAN and PWM outputs, bottom wall for the AZ sensor
    (under the rail to the base) and the DSLR. USB-C cut-outs for J19 (top wall) and the ESP32
    module's own port (left wall); four corner posts take the lid's M3 screws
    (heat-set inserts), which is why the board sits 10 mm off the side walls;
  - lid: 2.5 mm plate, vents over the A4988 drivers;
  - rail_clamp: a П-shaped bracket around the rail: the web on its +Z face, the legs over its top
    and bottom faces reaching back to the box (0.5 short of it, so the screws clamp the rail). Two
    M5x35, counterbored into the web, run down the legs into M5 heat-set inserts in bosses on the
    box's back wall. Two brackets, at x 120 and 150.
Run: ../../../CondensateDrain/.venv/Scripts/python.exe electronics_box.py
  (writes export/*.step in the world frame and export/print/*.stl in print orientation)
"""
from pathlib import Path

import cadquery as cq

RAIL_Z0, RAIL_Z1 = -117.5, -92.5     # MainRail's outer (-Z) and inner (+Z) faces
RAIL_Y0, RAIL_Y1 = -2.5, 22.5

BOARD_T = 1.52
BOARD_DX, BOARD_DY = 20.0, -63.5     # world x = kx + 20, y = ky - 63.5 (KiCad outline x 50..150, y 47..122)
BOARD_BACK_Z = -130.5                # the board's back face; THT leads reach 1.8 behind it
HOLES = ((55.5, 52.5), (144.5, 52.5), (55.5, 116.5), (144.5, 116.5))   # KiCad, M4

WALL, BACK, LID_T = 2.5, 3.0, 2.5
GAP_X, GAP_Y = 10.0, 1.0             # board edge to the walls: the lid posts stand at the ends
HEADROOM = 24.0                      # board top face to the lid: XH plugs stand ~12, then the cables bend
CORNER_R = 3.0
X0, X1 = 50 + BOARD_DX - GAP_X - WALL, 150 + BOARD_DX + GAP_X + WALL      # 57.5, 182.5
Y0, Y1 = 47 + BOARD_DY - GAP_Y - WALL, 122 + BOARD_DY + GAP_Y + WALL      # -20, 62
Z_IN = RAIL_Z0 - BACK                # back wall's inner face
Z_RIM = BOARD_BACK_Z - BOARD_T - HEADROOM
Z_LID = Z_RIM - LID_T

M5_INSERT = (6.4, 7.2)               # heat-set insert hole, depth (as in gliding_arm)
M4_INSERT = (5.6, 7.0)
M3_INSERT = (4.0, 6.0)
STANDOFF_D = 9.0
POST_D, POST_IN = 8.0, 4.5           # lid posts in the inner corners, centre 4.5 off both walls
LID_HOLE = 3.4

NOTCH_DEPTH, NOTCH_R = 8.0, 3.0      # cable notches, from the rim toward the back
NOTCHES_TOP = ((100.0, 140.0),)      # x span: AZ motor, ADXL
NOTCHES_RIGHT = ((-10.0, 24.0),)     # y span: EL motor, EL sensor, CAN, PWM A..C
NOTCHES_BOTTOM = ((70.0, 90.0),)     # x span: AZ sensor, DSLR
USB_CUT = (13.0, 7.5, 2.5)           # plug overmould cut-out: width, height, corner radius
J19 = (69.0 + BOARD_DX, BOARD_BACK_Z - 3.19)        # power input on the top edge: x, z of its mouth's centre
ESP_USB = (72.7 + BOARD_DY, BOARD_BACK_Z - 13.34)   # ESP32 module's port on the left edge: y, z
VENTS = (124.0, 160.0, 4.5, 2.0, 33.5, 46.5)        # x first, x last, pitch, width, y span: over U3/U4

CLAMP_X, CLAMP_W, CLAMP_T = (120.0, 150.0), 16.0, 6.0   # bracket centres, width, web thickness
CLAMP_LEG, CLAMP_FIT, CLAMP_GAP = 10.0, 0.2, 0.5   # leg thickness, leg to rail, leg end to box
CLAMP_SCREW_Y = (RAIL_Y0 - CLAMP_FIT - CLAMP_LEG / 2, RAIL_Y1 + CLAMP_FIT + CLAMP_LEG / 2)   # down the legs' middles
CLAMP_BOSS_D, CLAMP_BOSS_H = 12.0, 5.0             # back wall 3 + 5 takes the 7.2 insert
CLAMP_HOLE, CLAMP_CBORE = 5.5, (9.0, 3.0)          # M5 clearance; head counterbore diameter, depth


def xy(z0):
    """Workplane at z = z0 with local (x, y) = world (x, y); extrude(dz) runs to z0 + dz."""
    return cq.Workplane(cq.Plane(origin=(0, 0, z0), xDir=(1, 0, 0), normal=(0, 0, 1)))


def block(x0, x1, y0, y1, z0, z1):
    return cq.Workplane().box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def k2w(kx, ky):
    return kx + BOARD_DX, ky + BOARD_DY


def posts():
    return [(x, y) for x in (X0 + WALL + POST_IN, X1 - WALL - POST_IN) for y in (Y0 + WALL + POST_IN, Y1 - WALL - POST_IN)]


def outline(z0, dz):
    return xy(z0).center((X0 + X1) / 2, (Y0 + Y1) / 2).rect(X1 - X0, Y1 - Y0).extrude(dz).edges('|Z').fillet(CORNER_R)


def notch(x0, x1, y0, y1, round_along):
    """Cable notch open at the rim, its bottom (toward the back) rounded."""
    cut = block(x0, x1, y0, y1, Z_RIM - 1, Z_RIM + NOTCH_DEPTH)
    return cut.edges(f'|{round_along}').edges('>Z').fillet(NOTCH_R)


def usb_cut(axis):
    """Rounded USB-C overmould cut-out through a wall whose normal is axis ('x' or 'y'), at the origin."""
    w, h, r = USB_CUT
    plane = {'y': cq.Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)), 'x': cq.Plane((0, 0, 0), (0, 1, 0), (1, 0, 0))}[axis]
    return cq.Workplane(plane).rect(w, h).extrude(WALL + 2, both=True).edges(f'|{axis.upper()}').fillet(r)


def box():
    body = outline(RAIL_Z0, Z_RIM - RAIL_Z0)
    cavity = block(X0 + WALL, X1 - WALL, Y0 + WALL, Y1 - WALL, Z_RIM - 1, Z_IN).edges('|Z').fillet(1.0)
    body = body.cut(cavity)
    for x, y in posts():
        body = body.union(xy(Z_IN + .5).center(x, y).circle(POST_D / 2).extrude(Z_RIM - Z_IN - .5))
        body = body.cut(xy(Z_RIM - .01).center(x, y).circle(M3_INSERT[0] / 2).extrude(M3_INSERT[1]))
    for kx, ky in HOLES:
        x, y = k2w(kx, ky)
        body = body.union(xy(Z_IN + .5).center(x, y).circle(STANDOFF_D / 2).extrude(BOARD_BACK_Z - Z_IN - .5))
        body = body.cut(xy(BOARD_BACK_Z - .01).center(x, y).circle(M4_INSERT[0] / 2).extrude(M4_INSERT[1]))
    for x in CLAMP_X:
        for y in CLAMP_SCREW_Y:
            body = body.union(xy(Z_IN + .5).center(x, y).circle(CLAMP_BOSS_D / 2).extrude(-CLAMP_BOSS_H - .5))
            body = body.cut(xy(RAIL_Z0 + .01).center(x, y).circle(M5_INSERT[0] / 2).extrude(-M5_INSERT[1]))
    for y0, y1 in NOTCHES_RIGHT:
        body = body.cut(notch(X1 - WALL - .01, X1 + 1, y0, y1, 'X'))
    for x0, x1 in NOTCHES_TOP:
        body = body.cut(notch(x0, x1, Y1 - WALL - .01, Y1 + 1, 'Y'))
    for x0, x1 in NOTCHES_BOTTOM:
        body = body.cut(notch(x0, x1, Y0 - 1, Y0 + WALL + .01, 'Y'))
    x, z = J19
    body = body.cut(usb_cut('y').translate((x, Y1 - WALL / 2, z)))
    y, z = ESP_USB
    body = body.cut(usb_cut('x').translate((X0 + WALL / 2, y, z)))
    return body


def lid():
    body = outline(Z_RIM, -LID_T)
    for x, y in posts():
        body = body.cut(xy(Z_RIM + 1).center(x, y).circle(LID_HOLE / 2).extrude(-LID_T - 2))
    x0, x1, pitch, w, y0, y1 = VENTS
    for i in range(round((x1 - x0) / pitch) + 1):
        body = body.cut(xy(Z_RIM + 1).center(x0 + i * pitch, (y0 + y1) / 2).slot2D(y1 - y0, w, 90).extrude(-LID_T - 2))
    return body


def rail_clamp(x=CLAMP_X[0]):
    y0, y1 = RAIL_Y0 - CLAMP_FIT - CLAMP_LEG, RAIL_Y1 + CLAMP_FIT + CLAMP_LEG
    z0, z1 = RAIL_Z0 + CLAMP_GAP, RAIL_Z1 + CLAMP_T
    body = block(x - CLAMP_W / 2, x + CLAMP_W / 2, y0, y1, z0, z1).edges('|X').edges('>Z').fillet(4.0)
    body = body.cut(block(x - CLAMP_W / 2 - 1, x + CLAMP_W / 2 + 1, y0 + CLAMP_LEG, y1 - CLAMP_LEG, z0 - 1, RAIL_Z1))
    for y in CLAMP_SCREW_Y:
        body = body.cut(xy(z1 + 1).center(x, y).circle(CLAMP_HOLE / 2).extrude(z0 - z1 - 2))
        body = body.cut(xy(z1 + 1).center(x, y).circle(CLAMP_CBORE[0] / 2).extrude(-CLAMP_CBORE[1] - 1))
    return body


PARTS = {'box': box, 'lid': lid, 'rail_clamp': rail_clamp}

PRINT = {   # print orientation: axis and angle of the rotation, deg
    'box': ((1, 0, 0), 180),    # back wall on the bed, notches open upward
    'lid': ((1, 0, 0), 0),      # outer face on the bed
    'rail_clamp': ((0, 1, 0), 90),   # on its side: the П profile on the bed, layers along the legs
}

if __name__ == '__main__':
    out = Path(__file__).resolve().parent / 'export'
    (out / 'print').mkdir(parents=True, exist_ok=True)
    for name, make in PARTS.items():
        shape = make().val()
        cq.exporters.export(shape, str(out / f'{name}.step'))
        bb = shape.BoundingBox()
        print(f'{name}: vol {shape.Volume():.0f} mm3, x {bb.xmin:.1f}..{bb.xmax:.1f} y {bb.ymin:.1f}..{bb.ymax:.1f} '
              f'z {bb.zmin:.1f}..{bb.zmax:.1f}, valid {shape.isValid()}')
    for name, (axis, angle) in PRINT.items():
        shape = PARTS[name]().val().rotate(cq.Vector(0, 0, 0), cq.Vector(*axis), angle)
        bb = shape.BoundingBox()
        shape = shape.translate(cq.Vector(-bb.center.x, -bb.center.y, -bb.zmin))
        shape.exportStl(str(out / 'print' / f'{name}.stl'), 0.01, 0.1)
