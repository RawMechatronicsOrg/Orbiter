"""Gliding Orbit Arm: a 20 x 20 x 1.5 tube sliding inside the 25 x 25 x 1.5 OrbitArm
(bore 22 x 22) as a piston in a cylinder closed at both ends.

Parts, each in its own frame (mm, tube axis = Z, so they drop into Fusion components):
  - tube: the gliding arm, 253 long like OrbitArm, frame at its inner end, tube along -Z;
  - piston_cap: plug for the tube's inner end, the same expanding design as the red
    VertivalBar top cap (insert split by a tapered slot, a wedge pulled up by an M5x16
    into a nut spreads the legs), but the insert fits the 20 tube's 17 x 17 bore and the
    lid is 22.05 x 22.05 and carries a 10 mm skirt (bore 20.1, wall 0.975 in the 1 mm gap) back over
    the tube: the piston slides 12 mm long in OrbitArm's bore.
    Frame like the cap's: lid z 0..2 (toward OrbitArm's axis end), insert z -15..0, skirt z -10..0;
  - piston_wedge: the wedge for it (nut pocket open from below, like the cap's);
  - seal: insert for OrbitArm's open end taking up the 1 mm gap between the tubes:
    sleeve 21.9 into the bore (z 0..15), bore 20.0 x 20.0 for the gliding tube, a 30 x 30 collar
    over OrbitArm's outside (z 0..15) joined to it by the 2 mm flange on the end face, and on the
    collar's +X face the lock's boss (M5x7 heat-set insert, an M5x12 socket head screw in lock_knob
    through OrbitArm's wall onto the sleeve). Frame on OrbitArm's end face, +Z into the tube;
  - lock_knob: the fluted knob over the lock screw's head, a hex stub in its socket.
Run: ../../../CondensateDrain/.venv/Scripts/python.exe gliding_arm.py  (writes export/*.step)
"""
import math
from pathlib import Path

import cadquery as cq

LENGTH = 253.0                 # = OrbitArm
OUTER, WALL = 20.0, 1.5
BORE = OUTER - 2 * WALL        # 17
ARM_OUTER, ARM_BORE = 25.0, 22.0   # OrbitArm 25 x 25 x 1.5

LID, LID_H, LID_CHAMFER = 22.05, 2.0, 0.5   # 21.8 printed loose in the arm's bore (user, 2026-10-27: +0.2..0.3)
SKIRT_LEN, SKIRT_BORE = 10.0, 20.1   # piston skirt over the tube: longer bearing in OrbitArm's bore
INSERT_H = 15.0
HOLE_D = 6.0                   # M5 clearance, as in the cap
SLOT_TOP, SLOT_Z0, SLOT_SLOPE = 3.0, -2.5, 0.34   # slot half-width at its top, and its taper per mm
WEDGE_GAP = 0.54               # wedge to slot, as in the cap
WEDGE_Z = (-15.0, -7.5)
NUT_POCKET = (4.62, 4.0, -10.0)   # half x, half y, top z (open at the bottom)

SEAL_FLANGE, SEAL_SLEEVE, SEAL_BORE = 25.0, 21.9, 20.0   # bore 20.2 -> 20.0 (user, 2026-10-27: tighter)
SEAL_FLANGE_H, SEAL_DEPTH = 2.0, 15.0
# the seal's collar over OrbitArm's outside, like the hinge fittings' sleeves, and its lock (user,
# 2026-10-27): on the collar's +X face (away from the swing arm's fold side) a boss with an M5x7 heat-set
# insert; an M5x12 socket head screw in the lock knob goes through it and a hole in OrbitArm's wall and
# presses on the seal's sleeve (no hole there), which presses on the gliding tube
SEAL_COLLAR_LEN = 15.0
LOCK_Z, LOCK_BOSS_R, LOCK_HOLE_D = 7.5, 6.0, 5.5     # from OrbitArm's end face; the boss is a teardrop
LOCK_SCREW_LEN, LOCK_KNOB_H, LOCK_KNOB_GAP = 12.0, 8.0, 2.95  # knob 2.95 off the boss: the tip 0.1 off the sleeve
SOCKET_HEAD_D, SOCKET_HEAD_H, SOCKET_AF, SOCKET_T = 8.5, 5.0, 4.0, 2.5   # ISO 4762 M5

SWING_LENGTH = 160.0          # the swing arm (was LENGTH / 2.5 = 101.2; user, 2026-10-27: 120, then 40 more)

# Hinge between tube ends, the BigBoyPrinter tube joint (JointMaleCenter / JointFemale...) at
# 20 x 20: a fork of two ears (male, pink) on one tube end, three knuckles (female, green) on the
# other, interleaved, an M5 hex bolt through them as the pin. Its head sits in a hex pocket in the
# first knuckle so it cannot turn; the HandNutM5 knob with an M5x7 heat-set insert on the other end
# clamps the stack. Every ear face carries a 45 deg cone boss (D16 -> D10, 3 high) sitting in a
# matching recess in the knuckle (0.2 bigger, 3.2 deep): clamped, the cones take the load.
# Pin axis = Y, the arms swing in the XZ plane. The stack is 38.6 wide, wider than the tube.
# Fitting frame: tube end face at z=0, insert into the tube toward +Z, shoulder z -3..0,
# pin at z = -PIN_FROM_END.
#
# Fold: the pin sits 11 past each shoulder but PIN_OFFSET off the tube axis, to the inside of the
# fold (both fittings, the same physical side). The inner tube faces then close at
# a = 2 atan((10 - PIN_OFFSET) / 11): about 30 deg between the arms at 7 mm, folding toward that
# side; the other way the fold is shorter. Ears taper from the shoulder to an R10 end on the pin.
#
# Each fitting's insert expands like the caps (pulled wedge): an M5x16 from the shoulder side,
# its head sunk in a D9 counterbore, pulls the wedge (wide end deep, nut in it) toward the shoulder
# and spreads the legs. The male's screw goes in sideways between its ears; the female's through a
# D9 hole along the axis in its middle knuckle, which is 12 thick so the hole stays a hole.
SHOULDER = 3.0
PIN_FROM_END = SHOULDER + 11.0     # the other side's R10 knuckles clear the shoulder by 1
PIN_OFFSET = -7.0                   # pin off the tube axis, toward the inside of the fold (-X, male frame)
# Sleeve ("cheeks") over the tube's outside, around the insert: walls on the two faces the pin crosses
# and on the outer side; none on the fold side, where a wall would stop the fold at ~43 deg.
SLEEVE_LEN, SLEEVE_WALL, SLEEVE_GAP = 15.0, 2.4, 0.1
KNUCKLE_R, PIN_D = 10.0, 5.3
FEMALE_LAYERS = ((-19.3, -12.3), (-6.0, 6.0), (12.3, 19.3))   # y ranges, 0.3 gaps between layers;
MALE_LAYERS = ((-12.0, -6.3), (6.3, 12.0))                    # outer knuckles 7, middle 12, ears 5.7
EDGE_R, ROOT_R, SHOULDER_R = 1.2, 1.0, 2.0          # rounds: knuckle/ear faces, knuckle roots, shoulder corners
SLEEVE_ROOT_H = 13.0                # where the sleeve's side walls leave a wider shoulder (female), the flare from the
                                    # shoulder's very edge meets them this high (user, 2026-10-27: as much body as fits)
CONE_D, CONE_TIP, CONE_H = 16.0, 10.0, 3.0
LAYER_GAP = 0.3
# recess flank 0.15 (radial = axial at 45 deg) off the boss flank: clamped, the cones seat before
# the 0.3 face gap closes; 0.2 deeper than the boss
RECESS_R = CONE_D / 2 - LAYER_GAP + 0.15
RECESS_H = CONE_H + 0.2
BOLT_AF, BOLT_HEAD, BOLT_LEN = 8.0, 3.5, 45.0      # hex bolt M5x45 (ISO 4017 head)
POCKET_AF, POCKET_H = 8.2, 2.5                      # its pocket in the first knuckle; the head stands 1 proud
HEAD_D, HEAD_SEAT = 9.0, 2.0                        # M5 socket head counterbore, from the shoulder face to z=2
FIT_SLOT_Z, FIT_SLOT_TOP = 5.0, 3.5                 # slot from z 5 (3 mm ledge under the head), half-width there,
FIT_SLOT_SLOPE = SLOT_SLOPE                         # widening toward the far end like the caps' slot
FIT_WEDGE_Z, FIT_WEDGE_X = 7.5, 16.6                # wedge z 7.5..15, WEDGE_GAP to the slot
NUT_AF, NUT_Z = 8.1, 11.0                           # nut pocket in the wedge, z 11..15, open at the far end
# HandNutM5 (BigBoyPrinter, copied from its Fusion doc to source/): star face y=0, hub y 5.5..8;
# its D7.1 bore is refilled and bored D6.4 x 7.2 from the hub end for an M5x7 heat-set insert.
KNOB_STEP = Path(__file__).resolve().parent / 'source' / 'HandNutM5.step'
INSERT_HOLE, INSERT_DEPTH = 6.4, 7.2


def box(x, y, z0, z1, cx=0.0, cy=0.0):
    return cq.Solid.makeBox(x, y, z1 - z0, cq.Vector(cx - x / 2, cy - y / 2, z0))


def slot_halfwidth(z):
    return SLOT_TOP + SLOT_SLOPE * (SLOT_Z0 - z)


def tapered(z0, z1, half, width):
    """Prism through along X, symmetric trapezoid in YZ: half-width half(z) from z0 to z1."""
    plane = cq.Plane(origin=(-width / 2, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0))
    pts = [(-half(z0), z0), (half(z0), z0), (half(z1), z1), (-half(z1), z1)]
    return cq.Workplane(plane).polyline(pts).close().extrude(width).val()


def tube():
    return box(OUTER, OUTER, -LENGTH, 0).cut(box(BORE, BORE, -LENGTH - 1, 1))


def piston_cap():
    lid = cq.Workplane().add(box(LID, LID, 0, LID_H)).faces('>Z').edges().chamfer(LID_CHAMFER).val()
    skirt = box(LID, LID, -SKIRT_LEN, 0).cut(box(SKIRT_BORE, SKIRT_BORE, -SKIRT_LEN - 1, .01))
    body = lid.fuse(skirt, box(BORE, BORE, -INSERT_H, 0))
    body = body.cut(cq.Solid.makeCylinder(HOLE_D / 2, LID_H - SLOT_Z0 + 1, cq.Vector(0, 0, SLOT_Z0 - .5)))
    return body.cut(tapered(-INSERT_H - .01, SLOT_Z0, slot_halfwidth, BORE + 2)).clean()


def piston_wedge():
    body = tapered(WEDGE_Z[0], WEDGE_Z[1], lambda z: slot_halfwidth(z) - WEDGE_GAP, BORE)
    body = body.cut(box(2 * NUT_POCKET[0], 2 * NUT_POCKET[1], WEDGE_Z[0] - .1, NUT_POCKET[2]))
    return body.cut(cq.Solid.makeCylinder(HOLE_D / 2, 5, cq.Vector(0, 0, NUT_POCKET[2] - .5))).clean()


def seal():
    """OrbitArm's open end: a sleeve into its bore (taking up the 1 mm gap round the gliding tube) and a
    collar over its outside like the hinge fittings' sleeves, joined by the flange on the end face. On
    the collar's +X face a teardrop boss (its point toward the flange, which prints on the bed) holds the
    lock's M5x7 heat-set insert; the screw's hole goes on through the collar, and the screw presses on
    the sleeve (OrbitArm's wall gets a LOCK_HOLE_D hole there, LOCK_Z from its end face).
    Frame on OrbitArm's end face, +Z into the tube."""
    outer = ARM_OUTER / 2 + SLEEVE_GAP + SLEEVE_WALL
    collar = (cq.Workplane().add(box(2 * outer, 2 * outer, -SEAL_FLANGE_H, SEAL_COLLAR_LEN))
              .edges('|Z').fillet(SHOULDER_R).faces('>Z').edges().fillet(EDGE_R / 2)
              .faces('<Z').edges().chamfer(.5).val())
    collar = collar.cut(box(ARM_OUTER + 2 * SLEEVE_GAP, ARM_OUTER + 2 * SLEEVE_GAP, 0, SEAL_COLLAR_LEN + 1))
    body = collar.fuse(box(SEAL_SLEEVE, SEAL_SLEEVE, -.01, SEAL_DEPTH))
    face = outer + INSERT_DEPTH + .3 - SLEEVE_WALL                  # the boss face: 0.3 under the insert
    r, k = LOCK_BOSS_R, LOCK_BOSS_R / math.sqrt(2)
    boss = (cq.Workplane('YZ', origin=(outer - .5, 0, 0)).moveTo(k, LOCK_Z - k)
            .threePointArc((0, LOCK_Z + r), (-k, LOCK_Z - k)).lineTo(0, LOCK_Z - r * math.sqrt(2)).close()
            .extrude(face - outer + .5).val())
    boss = cq.Workplane().add(boss).faces('>X').edges().chamfer(.5).val()
    body = body.fuse(boss)
    body = body.cut(box(SEAL_BORE, SEAL_BORE, -SEAL_FLANGE_H - 1, SEAL_DEPTH + 1))
    body = body.cut(cq.Solid.makeCylinder(INSERT_HOLE / 2, INSERT_DEPTH + .01, cq.Vector(face + .01, 0, LOCK_Z), cq.Vector(-1, 0, 0)))
    return body.cut(cq.Solid.makeCylinder(LOCK_HOLE_D / 2, face - INSERT_DEPTH - (ARM_OUTER / 2 + SLEEVE_GAP) + .2,
                                          cq.Vector(face - INSERT_DEPTH + .01, 0, LOCK_Z), cq.Vector(-1, 0, 0))).clean()


def lock_knob():
    """The seal lock's knob, the pivot's fluted one on an M5x12 socket head screw (ISO 4762), in the
    screw's frame: axis Z, the head's bearing face z=0, the knob z 0..LOCK_KNOB_H. The head sits in a
    pocket from the inner face and a hex stub in the pocket's floor goes into its socket, so the knob
    turns the screw."""
    body = cq.Workplane('XY').circle(THUMB_D / 2).extrude(LOCK_KNOB_H).edges().chamfer(.6).val()
    for i in range(THUMB_FLUTES):
        a = 2 * math.pi * (i + .5) / THUMB_FLUTES
        rr = THUMB_D / 2 + 1.2
        body = body.cut(cq.Solid.makeCylinder(2.0, LOCK_KNOB_H + 2, cq.Vector(rr * math.cos(a), rr * math.sin(a), -1)))
    body = body.cut(cq.Solid.makeCylinder(SOCKET_HEAD_D / 2 + .05, SOCKET_HEAD_H + .01, cq.Vector(0, 0, -.01)))
    stub = hexagon(SOCKET_AF - .15, SOCKET_HEAD_H - SOCKET_T + .3, SOCKET_HEAD_H + .01)
    return body.fuse(stub).clean()


def swing_tube():
    return box(OUTER, OUTER, -SWING_LENGTH, 0).cut(box(BORE, BORE, -SWING_LENGTH - 1, 1))


def fit_slot_halfwidth(z):
    return FIT_SLOT_TOP + FIT_SLOT_SLOPE * (z - FIT_SLOT_Z)


def hexagon(af, z0, z1, axis='z', center=(0, 0, 0), flats='x'):
    """Hex prism, across flats `af`. Along Z: flat faces at x = +-af/2 (as ISO 4032 M5 nut's
    local frame) or, flats='y', at y = +-af/2; along Y (from y=z0 to y=z1): flat faces at z = +-af/2."""
    if axis == 'z':
        wp = cq.Workplane('XY', origin=(center[0], center[1], z0))
        prism = wp.polygon(6, af / math.cos(math.radians(30))).extrude(z1 - z0)
        if flats == 'x':
            prism = prism.rotate((center[0], center[1], 0), (center[0], center[1], 1), 30)
        return prism.val()
    wp = cq.Workplane('XZ', origin=(center[0], z1, center[2]))   # along Y, from z0 to z1 in y
    return wp.polygon(6, af / math.cos(math.radians(30))).extrude(z1 - z0).val()


def cone_y(y_face, direction, r_base, r_tip, h, pin_x, pin_z):
    """Cone on the pin axis, base on the plane y = y_face, pointing along +-Y."""
    return cq.Solid.makeCone(r_base, r_tip, h, cq.Vector(pin_x, y_face, pin_z), cq.Vector(0, direction, 0))


def tangent_point(p, c, r, side):
    """Point where a line from p touches the circle (c, r); side=+1/-1 picks the one with larger/smaller x."""
    dx, dz = p[0] - c[0], p[1] - c[1]
    dist = math.hypot(dx, dz)
    base, half = math.atan2(dz, dx), math.acos(r / dist)
    pts = [(c[0] + r * math.cos(base + s * half), c[1] + r * math.sin(base + s * half)) for s in (1, -1)]
    return max(pts, key=lambda q: side * q[0])


def knuckle(y0, y1, pin_x, pin_z):
    """One ear/knuckle: the hull of the shoulder line (x +-10) and an R10 disc on the pin, its face
    edges rounded (not the root, which the shoulder swallows)."""
    root, half = -SHOULDER + .5, OUTER / 2 - SHOULDER_R / 2   # clear of the shoulder's rounded corners
    left, right = (-half, root), (half, root)
    t_left = tangent_point(left, (pin_x, pin_z), KNUCKLE_R, -1)
    t_right = tangent_point(right, (pin_x, pin_z), KNUCKLE_R, 1)
    plane = cq.Plane(origin=(0, y1, 0), xDir=(1, 0, 0), normal=(0, -1, 0))   # local (x, z), extrudes toward -Y
    plate = (cq.Workplane(plane).moveTo(*left).lineTo(*t_left)
             .threePointArc((pin_x, pin_z - KNUCKLE_R), t_right).lineTo(*right).close()
             .extrude(y1 - y0).val())
    edges = [e for e in plate.Edges()
             if e.Center().z < -SHOULDER and min(abs(e.Center().y - y0), abs(e.Center().y - y1)) < 1e-6]
    return plate.fillet(EDGE_R, edges)


def sleeve_root_round(x0, x1, y_wall, half_y):
    """Concave flare between the shoulder's top (z=0) and the sleeve's side walls (y = +-y_wall): an arc
    from the shoulder's outer edge (y = +-half_y) tangent to the wall SLEEVE_ROOT_H up, the profile
    swept along each wall's outline, round its SHOULDER_R corners, and cut to the shoulder's rounded
    footprint (OCC's own fillet fails where the edge runs into those faces)."""
    d, h, c = half_y - y_wall, SLEEVE_ROOT_H, SHOULDER_R
    r = (d * d + h * h) / (2 * d)                     # the arc's centre (r, h) in (out, z)
    u = (d - 2 * r, -h)
    n = math.hypot(*u)
    mid = (r + r * u[0] / n, h + r * u[1] / n)
    k = c / math.sqrt(2)
    path = (cq.Workplane('XY').moveTo(x0, y_wall - c).threePointArc((x0 + c - k, y_wall - c + k), (x0 + c, y_wall))
            .lineTo(x1 - c, y_wall).threePointArc((x1 - c + k, y_wall - c + k), (x1, y_wall - c)).wire().val())
    plane = cq.Plane(origin=(x0, y_wall - c, 0), xDir=(-1, 0, 0), normal=(0, 1, 0))   # local (out, z)
    profile = cq.Workplane(plane).moveTo(0, 0).lineTo(d, 0).threePointArc(mid, (0, h)).close()
    strip = profile.sweep(cq.Workplane().add(path)).val()
    footprint = (cq.Workplane().add(box(x1 - x0, 2 * half_y, -1, h + 1, cx=(x0 + x1) / 2))
                 .edges('|Z').fillet(SHOULDER_R).val())
    strip = strip.intersect(footprint)
    return strip.fuse(strip.mirror('XZ'))


def hinge_fitting(layers, male, pin_x, tube=True):
    """tube=False: knuckles on a plain 3 mm pad (z -3..0) to fuse onto another part, no insert or sleeve."""
    pin_z = -PIN_FROM_END
    span = max(abs(y) for layer in layers for y in layer)
    insert = cq.Workplane().add(box(BORE, BORE, 0, INSERT_H)).faces('>Z').edges().chamfer(.5).val()
    # the fold side is the pin's side; the sleeve's outer wall and the shoulder grow on the other
    outer = OUTER / 2 + SLEEVE_GAP + SLEEVE_WALL
    x0, x1 = (-OUTER / 2, outer) if pin_x < 0 else (-outer, OUTER / 2)
    if not tube:
        x0, x1 = -OUTER / 2, OUTER / 2
    half_y = max(span, outer)
    shoulder = (cq.Workplane().add(box(x1 - x0, 2 * half_y, -SHOULDER, 0, cx=(x0 + x1) / 2))
                .edges('|Z').fillet(SHOULDER_R).faces('<Z').edges().fillet(EDGE_R / 2).val())
    sleeve = (cq.Workplane().add(box(x1 - x0, 2 * outer, -.01, SLEEVE_LEN, cx=(x0 + x1) / 2))
              .edges('|Z').fillet(SHOULDER_R).faces('>Z').edges().fillet(EDGE_R / 2).val())
    clear = OUTER / 2 + SLEEVE_GAP
    sleeve = sleeve.cut(box(OUTER + 2 * SLEEVE_WALL + 2, 2 * clear, 0, SLEEVE_LEN + 1,
                            cx=(clear - (OUTER / 2 + SLEEVE_WALL + 1)) * (1 if pin_x < 0 else -1)))
    body = insert.fuse(shoulder, sleeve) if tube else shoulder
    if tube and half_y > outer:   # the shoulder sticks out past the sleeve's side walls: round that inner corner
        body = body.fuse(sleeve_root_round(x0, x1, outer, half_y))
    for y0, y1 in layers:
        body = body.fuse(knuckle(y0, y1, pin_x, pin_z))
    # rounds in the corners where the knuckles leave the shoulder face (not on the outer faces,
    # which run flush with the shoulder's sides)
    roots = [e for e in body.Edges()
             if abs(e.Center().z + SHOULDER) < 1e-6 and e.geomType() == 'LINE' and abs(e.tangentAt(0).x) > .99
             and abs(abs(e.Center().y) - span) > 1e-6
             and any(min(abs(e.Center().y - y0), abs(e.Center().y - y1)) < 1e-6 for y0, y1 in layers)]
    body = body.fillet(ROOT_R, roots)
    for y0, y1 in layers:
        if male:   # cone bosses on both faces of every ear
            body = body.fuse(cone_y(y1, 1, CONE_D / 2, CONE_TIP / 2, CONE_H, pin_x, pin_z),
                             cone_y(y0, -1, CONE_D / 2, CONE_TIP / 2, CONE_H, pin_x, pin_z))
    if not male:   # recesses on the knuckle faces that meet an ear
        for y0, y1 in layers:
            if y0 > -span:
                body = body.cut(cone_y(y0 - .01, 1, RECESS_R + .01, RECESS_R - RECESS_H, RECESS_H + .01, pin_x, pin_z))
            if y1 < span:
                body = body.cut(cone_y(y1 + .01, -1, RECESS_R + .01, RECESS_R - RECESS_H, RECESS_H + .01, pin_x, pin_z))
        # bolt head pocket on the outer face of the first knuckle
        body = body.cut(hexagon(POCKET_AF, -span - 1, -span + POCKET_H, axis='y', center=(pin_x, 0, pin_z)))
        # the wedge screw goes in through the middle knuckle, along the axis
        if tube:
            body = body.cut(cq.Solid.makeCylinder(HEAD_D / 2, PIN_FROM_END + KNUCKLE_R + 1, cq.Vector(0, 0, pin_z - KNUCKLE_R - 1)))
    reach = span + CONE_H + 1        # through the cone bosses too
    body = body.cut(cq.Solid.makeCylinder(PIN_D / 2, 2 * reach, cq.Vector(pin_x, -reach, pin_z), cq.Vector(0, 1, 0)))
    if not tube:
        return body.clean()
    # expanding insert, pulled wedge: head counterbore, ledge with the screw hole, slot widening to the far end
    body = body.cut(cq.Solid.makeCylinder(HEAD_D / 2, HEAD_SEAT + SHOULDER + .05, cq.Vector(0, 0, -SHOULDER - .05)))
    body = body.cut(cq.Solid.makeCylinder(HOLE_D / 2 - .25, FIT_SLOT_Z - HEAD_SEAT + .1, cq.Vector(0, 0, HEAD_SEAT - .05)))
    return body.cut(tapered(FIT_SLOT_Z, INSERT_H + .01, fit_slot_halfwidth, BORE + 2)).clean()


def hinge_male():
    return hinge_fitting(MALE_LAYERS, True, PIN_OFFSET)


def hinge_female():
    return hinge_fitting(FEMALE_LAYERS, False, -PIN_OFFSET)   # mounted turned about Y: its -X is the male's +X


def hinge_wedge():
    """Pulled wedge, z FIT_WEDGE_Z..INSERT_H in the fitting frame: narrow end toward the shoulder,
    screw hole, nut pocket (flats across Y) open at the far end, as in the caps' wedges."""
    body = tapered(FIT_WEDGE_Z, INSERT_H, lambda z: fit_slot_halfwidth(z) - WEDGE_GAP, FIT_WEDGE_X)
    body = body.cut(hexagon(NUT_AF, NUT_Z, INSERT_H + .1, flats='y'))
    return body.cut(cq.Solid.makeCylinder(HOLE_D / 2, NUT_Z - FIT_WEDGE_Z + 1, cq.Vector(0, 0, FIT_WEDGE_Z - .5))).clean()


def hex_bolt(length=BOLT_LEN):
    """ISO 4017-like M5 hex bolt (thread not modelled): head y -3.5..0, shank y 0..length (pin axis = Y)."""
    head = hexagon(BOLT_AF, -BOLT_HEAD, 0, axis='y')
    return head.fuse(cq.Solid.makeCylinder(2.5, length, cq.Vector(0, 0, 0), cq.Vector(0, 1, 0))).clean()


def knob():
    """HandNutM5 with its bore redone for an M5x7 heat-set insert from the hub end (y=8)."""
    body = cq.importers.importStep(str(KNOB_STEP)).val()
    body = body.fuse(cq.Solid.makeCylinder(3.6, 6.1, cq.Vector(0, 1.9, 0), cq.Vector(0, 1, 0)))
    body = body.cut(cq.Solid.makeCylinder(INSERT_HOLE / 2, INSERT_DEPTH + .1, cq.Vector(0, 8 - INSERT_DEPTH, 0), cq.Vector(0, 1, 0)))
    return body.cut(cq.Solid.makeCylinder(2.75, 10, cq.Vector(0, -1, 0), cq.Vector(0, 1, 0))).clean()


HANDSCANNER_MODELS = Path(__file__).resolve().parents[3] / 'HandScanner' / 'models'


def phone_clamp_arm_base():
    """The phone clamp's base (HandScanner/models/phone_clamp.py, solid remodel) without its profile
    channel, with the tilt pivot's ear hanging under the sole by its -X edge (base_pivot_ear): the
    knuckle block the clamp hangs on (clamp_hinge_block) is a separate part on that pivot. Clamp frame:
    sole z=0, clamp axis Y."""
    import sys
    sys.path.insert(0, str(HANDSCANNER_MODELS))
    import phone_clamp
    base = phone_clamp.lower({k: v['default'] for k, v in phone_clamp.PARAMS.items()}, channel=False)
    return base.fuse(base_pivot_ear(), clamp_profile_channel(phone_clamp)).clean()


def clamp_hinge_block():
    """The cross the phone clamp hangs on: the female knuckles on the male fork at the swing arm's far
    end (yaw, pin along X) hang under a block (the user's, 2026-10-26: over the stack, its pad's
    outline, up to BLOCK_TOP) whose side faces run on as a fork of two cheeks (PIVOT_LAYERS 0 and 2)
    to the tilt pivot along Y, where the base's ear sits between them (the hinges' way: cone recesses
    on the cheeks' inner faces, the pivot nut's hex pocket in the left cheek, the knob on the right one,
    toward the clamp's knurled head).
    One side outline (x, z) for block and cheeks: the top flat on to the R10 disc on the pivot (flush,
    PIVOT_Z = BLOCK_TOP - R10), round it and up a concave neck onto the stack's bottom at its inner end;
    the fork's slot cut from that end; the top's edges, outside and round the slot, chamfered
    TOP_CHAMFER (a ridge along each cheek). Its edges are rounded EDGE_R, the slot's inner corners
    ROOT_R, except the two bottom edges the knuckles' end faces continue flush. Under it the other hinges'
    female knuckles (hinge_fitting: roots, recesses, the bolt head's pocket, bore) without their pad;
    then the pivot's recesses, pocket and bore.
    Clamp frame. The knuckles are the fitting's frame turned -90 deg about Z (CLAMP_FROM_FITTING): pin
    axis across the clamp (X), so its knob sits off the base's side. The far male is mounted turned
    180 deg about the swing arm's axis, so this hinge's pin offset is the first hinge's mirrored: the
    pin sits at fitting x = PIN_OFFSET, i.e. clamp y = CLAMP_MOUNT_Y - PIN_OFFSET (toward the knurled
    head), z = -PIN_FROM_END - PIVOT_DROP. It sits CLAMP_MOUNT_X along the pin, beside the base, so the
    swing arm folded toward the profile passes beside the profile channel; mirrored on -X so the knob
    stays on its outer end (the bolt head's pocket faces the base)."""
    span = max(abs(y) for layer in FEMALE_LAYERS for y in layer)
    inner, outer = CLAMP_MOUNT_X + span, CLAMP_MOUNT_X - span          # the stack's ends along X
    bottom = -PIVOT_DROP - SHOULDER                                    # the knuckles' root plane
    y0, y1 = CLAMP_MOUNT_Y - OUTER / 2, CLAMP_MOUNT_Y + OUTER / 2
    # neck: the concave round tangent to the bottom at the stack's inner end and to the pivot's disc
    a, b = PIVOT_X - inner, PIVOT_Z - bottom
    r = (a * a + b * b - KNUCKLE_R ** 2) / (2 * KNUCKLE_R - 2 * b)
    n = (inner, bottom - r)
    d = math.hypot(PIVOT_X - n[0], PIVOT_Z - n[1])
    t = (PIVOT_X + KNUCKLE_R * (n[0] - PIVOT_X) / d, PIVOT_Z + KNUCKLE_R * (n[1] - PIVOT_Z) / d)
    m = (math.atan2(t[1] - n[1], t[0] - n[0]) + math.pi / 2) / 2
    plane = cq.Plane(origin=(0, y1, 0), xDir=(1, 0, 0), normal=(0, -1, 0))   # local (x, z), extrudes toward -Y
    body = (cq.Workplane(plane).moveTo(outer, BLOCK_TOP).lineTo(PIVOT_X, BLOCK_TOP)
            .threePointArc((PIVOT_X + KNUCKLE_R, PIVOT_Z), t)
            .threePointArc((n[0] + r * math.cos(m), n[1] + r * math.sin(m)), (inner, bottom))
            .lineTo(outer, bottom).close().extrude(y1 - y0).val())
    slot_y0, slot_y1 = PIVOT_LAYERS[0][1], PIVOT_LAYERS[2][0]
    far = PIVOT_X + KNUCKLE_R + 1
    body = body.cut(box(far - inner, slot_y1 - slot_y0, PIVOT_Z - KNUCKLE_R - 1, BLOCK_TOP + 1,
                        cx=(far + inner) / 2, cy=(slot_y0 + slot_y1) / 2))
    corners = [e for e in body.Edges() if e.geomType() == 'LINE' and abs(e.tangentAt(0).z) > .99
               and abs(e.Center().x - inner) < 1e-6]
    body = body.fillet(ROOT_R, corners)
    # 45 deg chamfers on the top's edges, half a cheek wide (user, 2026-10-27): round the outside and
    # round the slot, so each cheek's top is a ridge; on the discs they run out
    c, end = TOP_CHAMFER, PIVOT_X + KNUCKLE_R + 5
    keep = (cq.Workplane('XY', origin=((outer + end) / 2, CLAMP_MOUNT_Y, BLOCK_TOP - c))
            .rect(end - outer, OUTER).extrude(c + 1, taper=45).val())
    body = body.intersect(keep.fuse(box(end - outer + 2, OUTER + 2, bottom - KNUCKLE_R - 2, BLOCK_TOP - c,
                                        cx=(outer + end) / 2, cy=CLAMP_MOUNT_Y)))
    flare = cq.Sketch().rect(end - inner, slot_y1 - slot_y0).vertices('<X').fillet(ROOT_R)
    body = body.cut(cq.Workplane('XY', origin=((inner + end) / 2, (slot_y0 + slot_y1) / 2, BLOCK_TOP - c))
                    .placeSketch(flare).extrude(c + 1, taper=-45).val())

    def rounded(e):   # not the bottom edges along Y at the stack's ends: the knuckles' faces go on there
        return not (abs(e.Center().z - bottom) < 1e-6 and abs(e.tangentAt(.5).y) > .99
                    and min(abs(e.Center().x - outer), abs(e.Center().x - inner)) < 1e-6)
    body = body.fillet(EDGE_R, [e for e in creases(body) if rounded(e)])
    # the knuckles as on the other hinges (with their roots, recesses, the bolt head's pocket and the
    # bore), their pad dropped: the block is the pad now
    knuckles = to_clamp(hinge_fitting(FEMALE_LAYERS, False, PIN_OFFSET, tube=False))
    body = body.fuse(knuckles.cut(box(inner - outer + 2, OUTER + 2, bottom, 0, cx=CLAMP_MOUNT_X, cy=CLAMP_MOUNT_Y))).clean()
    (l0, l1), _, (r0, r1) = PIVOT_LAYERS
    body = body.cut(cone_y(l1 + .01, -1, PIVOT_RECESS_R + .01, PIVOT_RECESS_R - PIVOT_RECESS_H, PIVOT_RECESS_H + .01,
                           PIVOT_X, PIVOT_Z))
    body = body.cut(cone_y(r0 - .01, 1, PIVOT_RECESS_R + .01, PIVOT_RECESS_R - PIVOT_RECESS_H, PIVOT_RECESS_H + .01,
                           PIVOT_X, PIVOT_Z))
    body = body.cut(hexagon(POCKET_AF, l0 - 1, l0 + PIVOT_POCKET_H, axis='y', center=(PIVOT_X, 0, PIVOT_Z)))
    body = body.cut(cq.Solid.makeCylinder(PIN_D / 2, r1 - l0 + 2, cq.Vector(PIVOT_X, l0 - 1, PIVOT_Z), cq.Vector(0, 1, 0)))
    return body.cut(yaw_bolt_passage()).clean()


def yaw_bolt_passage():
    """Room for the yaw hinge's bolt to come out of its pocket along the pin (user, 2026-10-27), under the
    right cheek's disc and the base's ear: the pocket's hexagon 0.2 looser a side, from the stack's inner
    face on past the discs. Clamp frame."""
    span = max(abs(y) for layer in FEMALE_LAYERS for y in layer)
    return to_clamp(hexagon(POCKET_AF + .4, -span - 30, -span + .01, axis='y', center=(PIN_OFFSET, 0, -PIN_FROM_END)))


def to_clamp(shape):
    """The knuckle block's fitting frame -> clamp frame (see clamp_hinge_block)."""
    shape = shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), CLAMP_FROM_FITTING)
    shape = shape.translate(cq.Vector(abs(CLAMP_MOUNT_X), CLAMP_MOUNT_Y, -PIVOT_DROP))
    return shape.mirror('YZ') if CLAMP_MOUNT_X < 0 else shape


def creases(shape):
    """Edges where two faces meet at an angle (not the seams of tangent faces)."""
    out = []
    for e in shape.Edges():
        faces = e.ancestors(shape, 'Face').Faces()
        p = e.positionAt(.5)
        if len(faces) == 2 and faces[0].normalAt(p).dot(faces[1].normalAt(p)) < .999:
            out.append(e)
    return out


def tangent_points(p, c, r):
    """Both points where a line from p touches the circle (c, r), in the (x, z) plane."""
    dx, dz = p[0] - c[0], p[1] - c[1]
    dist = math.hypot(dx, dz)
    base, half = math.atan2(dz, dx), math.acos(r / dist)
    return [(c[0] + r * math.cos(base + s * half), c[1] + r * math.sin(base + s * half)) for s in (1, -1)]


def base_pivot_ear():
    """The tilt pivot's ear on the base, the middle of the three: a slab PIVOT_LAYERS[1] hanging from the
    sole (x -16 .. the disc's far side, that face vertical) to an R10 disc on the pivot, its face edges
    rounded like the knuckles', cone bosses on both faces like the hinges' male ears (the pivot's smaller
    ones). Its root (user, 2026-10-27: a fat one) is an EAR_ROOT_R concave round swept along both faces
    and round the vertical end; toward -X it stops where the sole's flat bottom ends (EAR_ROOT_X0), the
    base's edge rounding up from there (OCC's own fillet fails on that edge, Fusion's too)."""
    y0, y1 = PIVOT_LAYERS[1]
    c = (PIVOT_X, PIVOT_Z)
    xe = PIVOT_X + KNUCKLE_R
    left, right = (-16.0, .5), (xe, .5)       # .5 into the sole
    t_left = min(tangent_points(left, c, KNUCKLE_R), key=lambda q: q[0])
    plane = cq.Plane(origin=(0, y1, 0), xDir=(1, 0, 0), normal=(0, -1, 0))
    ear = (cq.Workplane(plane).moveTo(*left).lineTo(*t_left)
           .threePointArc((PIVOT_X, PIVOT_Z - KNUCKLE_R), (xe, PIVOT_Z)).lineTo(*right).close().extrude(y1 - y0).val())
    ear = ear.fillet(EDGE_R, [e for e in ear.Edges() if e.Center().z < 0
                              and min(abs(e.Center().y - y0), abs(e.Center().y - y1)) < 1e-6])
    r, rc, k = EAR_ROOT_R, EDGE_R, EDGE_R * (1 - 1 / math.sqrt(2))
    x0 = EAR_ROOT_X0 - 1
    path = (cq.Workplane('XY').moveTo(x0, y0).lineTo(xe - rc, y0).threePointArc((xe - k, y0 + k), (xe, y0 + rc))
            .lineTo(xe, y1 - rc).threePointArc((xe - k, y1 - k), (xe - rc, y1)).lineTo(x0, y1).wire().val())
    m = r - r / math.sqrt(2)
    profile = (cq.Workplane(cq.Plane(origin=(x0, y0, 0), xDir=(0, -1, 0), normal=(1, 0, 0)))   # local (out, -z)
               .moveTo(0, 0).lineTo(r, 0).threePointArc((m, m), (0, r)).close())
    root = profile.sweep(cq.Workplane().add(path)).val()
    root = root.cut(box(20, y1 - y0 + 2 * r + 2, -r - 1, 1, cx=EAR_ROOT_X0 - 10, cy=(y0 + y1) / 2))
    ear = ear.fuse(root).clean()
    ear = ear.fuse(cone_y(y1, 1, PIVOT_CONE_D / 2, PIVOT_CONE_TIP / 2, PIVOT_CONE_H, PIVOT_X, PIVOT_Z),
                   cone_y(y0, -1, PIVOT_CONE_D / 2, PIVOT_CONE_TIP / 2, PIVOT_CONE_H, PIVOT_X, PIVOT_Z))
    ear = ear.cut(cq.Solid.makeCylinder(PIN_D / 2, y1 - y0 + 2 * PIVOT_CONE_H + 2,
                                        cq.Vector(PIVOT_X, y0 - PIVOT_CONE_H - 1, PIVOT_Z), cq.Vector(0, 1, 0)))
    return ear.cut(yaw_bolt_passage())


def pivot_bolt():
    return hex_bolt(PIVOT_BOLT_LEN)


def pivot_nut():
    """ISO 4035 M5 thin nut (thread not modelled): axis Y, y 0..PIVOT_NUT_H, flats at z = +-af/2."""
    return hexagon(PIVOT_NUT_AF, 0, PIVOT_NUT_H, axis='y').cut(
        cq.Solid.makeCylinder(2.5, PIVOT_NUT_H + 2, cq.Vector(0, -1, 0), cq.Vector(0, 1, 0)))


def thumb_knob():
    """Small fluted knob for the pivot, frame like HandNutM5's: axis Y, outer face y=0, the end toward
    the ear at y=8. The pivot bolt's head sits flush in a hex pocket in its outer face, so the knob turns
    the bolt into the nut in the left cheek."""
    body = cq.Workplane('XZ', origin=(0, 8, 0)).circle(THUMB_D / 2).extrude(8).edges().chamfer(.6).val()
    for i in range(THUMB_FLUTES):
        a = 2 * math.pi * (i + .5) / THUMB_FLUTES
        r = THUMB_D / 2 + 1.2
        body = body.cut(cq.Solid.makeCylinder(2.0, 10, cq.Vector(r * math.cos(a), -1, r * math.sin(a)), cq.Vector(0, 1, 0)))
    body = body.cut(hexagon(POCKET_AF, -1, BOLT_HEAD, axis='y'))
    return body.cut(cq.Solid.makeCylinder(PIN_D / 2, 10, cq.Vector(0, -1, 0), cq.Vector(0, 1, 0))).clean()


def clamp_profile_channel(phone_clamp):
    """The 16 x 13 profile snap channel back under the base, on the stretch the hinge leaves free: from
    y -29.5 to 1 mm short of the knuckle block (its pad, or the R10 knuckle discs if they reach further):
    same walls and 3.5 mm spacer as phone_clamp_lower.channel, teeth CLIP_HOOK_SCALE times bigger. The
    profile runs along the clamp and ends against the block."""
    import phone_clamp_lower
    import profile_clip
    p = {k: v['default'] for k, v in phone_clamp.PARAMS.items()}
    p['hook'] *= CLIP_HOOK_SCALE
    block_y0 = CLAMP_MOUNT_Y - max(OUTER / 2, PIN_OFFSET + KNUCKLE_R)   # the block's end toward -Y
    y_range = (phone_clamp_lower.SOLE_Y[0], block_y0 - 1.0)
    gap, s = p['head_gap'], p['slot_width'] / 2
    edge = (phone_clamp_lower.EDGE_X, phone_clamp_lower.EDGE_Z)
    walls = profile_clip.walls(p, y_range, -gap, edge, phone_clamp_lower.EDGE_Z)
    spacer = cq.Solid.makeBox(2 * s + .2, y_range[1] - y_range[0], gap + phone_clamp_lower.OVERLAP,
                              cq.Vector(-s - .1, y_range[0], -gap))
    return walls.fuse(spacer)


CLIP_HOOK_SCALE = 2.0   # the user wanted the snap teeth about twice as big (0.6 -> 1.2): ~3 % root strain, PETG

PROFILE_W, PROFILE_H, PROFILE_T, PROFILE_LEN = 16.0, 13.0, 1.5, 120.0


def u_profile():
    """Aluminium U channel 16 x 13 x 1.5 the HandScanner parts snap onto, drawn open side down: web on
    top (z -1.5..0), flange ends at z -13. Runs from y=0 toward -Y. In Orbiter it is mounted turned
    over about its length (the user's choice), open side toward the clamp's sole: the flange ends rest
    on the channel's ceiling and the teeth catch the web; the 16 x 13 outline the channel holds is the
    same either way."""
    body = box(PROFILE_W, PROFILE_LEN, -PROFILE_H, 0, cy=-PROFILE_LEN / 2)
    return body.cut(box(PROFILE_W - 2 * PROFILE_T, PROFILE_LEN + 2, -PROFILE_H - 1, -PROFILE_T, cy=-PROFILE_LEN / 2))


CLAMP_FROM_FITTING = -90.0   # deg about Z: fitting x -> clamp -y, fitting y (pin axis) -> clamp x
# block centre along the clamp (the profile channel ends 1 mm short of it, the profile and laser stay put)
CLAMP_MOUNT_Y = 10.5
# block centre along the pin: the swing arm's fitting (+-12.5) clears the lower jaw's fingers (+-31, by
# the knurled head) by 0.5, so it passes beside the base and its profile channel both ways: swing
# [-145, +65] instead of [-104, +16] (the block's own pad stops the male at 68, the phone at -148; at
# -30 the fingers stopped it at -118). On -X (toward the phone's bottom), so the clamp with its knurled
# head sits above the hinge when the pin is vertical.
CLAMP_MOUNT_X = -44.0

# Phone tilt (user, 2026-10-25, sketches: an eye under the base's -X edge, the phone nods up/down;
# then a three-tooth cross): the knuckle block is a cross hanging on a pivot along the clamp's long axis
# (Y) under the base's -X edge: its two cheeks take the base's ear between them like the hinges'
# knuckles (cone bosses on the ear, recesses in the cheeks, 0.3 gaps), clamped by a knob screw from the
# right cheek, toward the clamp's knurled head (the D38 star does not fit there): an M5x25 whose head sits
# in the small fluted knob, into a thin M5 nut in a pocket in the left cheek. It comes out toward +Y (user,
# 2026-10-27): the profile channel starts 1 mm past the left cheek. The block sits PIVOT_DROP lower than when
# it was fused to the sole, so its knob clears the phone when the phone nods toward it: 16 deg that
# way, 115 the other (then the swing arm); yaw on the hinge -65 / +150.
PIVOT_DROP = 11.0
BLOCK_TOP = -3.5                         # the user's block over the knuckles
PIVOT_X, PIVOT_Z = -13.0, BLOCK_TOP - KNUCKLE_R   # the cheeks' R10 discs flush with the block's top
# three equal layers across the user's block (y 0.5..20.5), 0.3 gaps: left cheek, the base's ear, right cheek
PIVOT_LAYER_W = (OUTER - 2 * LAYER_GAP) / 3
PIVOT_LAYERS = tuple((y, y + PIVOT_LAYER_W) for y in
                     (CLAMP_MOUNT_Y - OUTER / 2 + i * (PIVOT_LAYER_W + LAYER_GAP) for i in range(3)))
# the hinges' cones scaled down to 2 high (D14 -> D10, 45 deg) so the left cheek holds the nut's
# pocket and a recess with 1.3 between them
PIVOT_CONE_D, PIVOT_CONE_TIP, PIVOT_CONE_H = 14.0, 10.0, 2.0
PIVOT_RECESS_R, PIVOT_RECESS_H = PIVOT_CONE_D / 2 - LAYER_GAP + 0.15, PIVOT_CONE_H + 0.2
EAR_ROOT_R = 6.0                         # the base ear's root round (user, 2026-10-27: R3, then twice that)
EAR_ROOT_X0 = -15.0                      # the sole's flat bottom ends at -15.15 by the ear (the base's edge rounds up)
TOP_CHAMFER = PIVOT_LAYER_W / 2 + .1      # the block's top edges: the two on a cheek meet in a ridge (0.1 under the top)
PIVOT_POCKET_H = 3.0                     # the left cheek's nut pocket, y 0.5..3.5 (the profile channel ends at -0.5)
PIVOT_NUT_AF, PIVOT_NUT_H = 8.0, 2.7     # ISO 4035 M5 thin nut, seated on the pocket's floor (y 0.8..3.5)
PIVOT_BOLT_LEN = 25.0                    # head flush in the knob (y 25..28.5), end at y 0: through the nut
THUMB_D, THUMB_FLUTES = 18.0, 8


PARTS = {'tube': tube, 'piston_cap': piston_cap, 'piston_wedge': piston_wedge, 'seal': seal,
         'swing_tube': swing_tube, 'hinge_male': hinge_male, 'hinge_female': hinge_female,
         'hinge_wedge': hinge_wedge, 'hex_bolt': hex_bolt, 'knob': knob, 'phone_clamp_arm_base': phone_clamp_arm_base,
         'u_profile': u_profile, 'clamp_hinge_block': clamp_hinge_block, 'pivot_bolt': pivot_bolt,
         'thumb_knob': thumb_knob, 'pivot_nut': pivot_nut, 'lock_knob': lock_knob}

if __name__ == '__main__':
    out = Path(__file__).resolve().parent / 'export'
    out.mkdir(exist_ok=True)
    for name, make in PARTS.items():
        shape = make()
        bb = shape.BoundingBox()
        shape.exportStep(str(out / f'{name}.step'))
        print(f'{name}: valid={shape.isValid()} vol={shape.Volume():.1f} '
              f'x {bb.xmin:.2f}..{bb.xmax:.2f} y {bb.ymin:.2f}..{bb.ymax:.2f} z {bb.zmin:.2f}..{bb.zmax:.2f}')
