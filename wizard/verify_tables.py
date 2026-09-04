#!/usr/bin/env python3
"""
Verify a Peregrine card image against docs/TABLE_FORMAT.md.

This deliberately re-implements the lookup from the *spec*, not from the
solver's internals, so that it tests the contract the robot firmware will
rely on rather than agreeing with whatever the solver happened to write.

    py -3.12 wizard/verify_tables.py <dir-or-drive>
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import struct
import sys

DTYPE = {
    "u8":  ("B", 1),
    "u16": ("<H", 2),
    "f16": ("<e", 2),
    "f32": ("<f", 4),
}


BOM = codecs.BOM_UTF8


def read_json(path: str):
    """Read JSON, tolerating the UTF-8 BOM that Windows editors add."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(BOM):
        raw = raw[len(BOM):]
    return json.loads(raw.decode("utf-8"))


class Card:
    """Read-only view of a card image, addressed the way the robot addresses it."""

    def __init__(self, root: str):
        self.root = root
        self.m = read_json(os.path.join(root, "MANIFEST.JSON"))
        g = self.m["grid"]
        self.n = g["n"]
        self.lo = g["min"]
        self.wrap = g["wrap"]
        self.step = [
            (2 * math.pi / self.n[k]) if self.wrap[k]
            else (g["max"][k] - g["min"][k]) / (self.n[k] - 1)
            for k in range(6)
        ]
        e = self.m["encoding"]
        self.dtype = e["dtype"]
        self.elem_bytes = e["elem_bytes"]
        self.scale = e["scale"]
        self.chunk_elements = e["chunk_elements"]
        self.chunk_shift = e["chunk_shift"]
        self.unreachable = e["unreachable"]
        # The escape band. Absent, or zero codes, means a card solved before
        # this existed or with `escape` off -- every unreachable cell is then
        # simply unreachable, which is what the old decode already did.
        self.escape_base = e.get("escape_base", 0) or 0
        self.escape_codes = e.get("escape_codes", 0) or 0
        self.escape_scale = e.get("escape_scale", 0.0) or 0.0
        self._fh: dict[tuple[int, int], object] = {}

    def flat_index(self, ix, iy, ih, ivx, ivy, iw) -> int:
        n = self.n
        return ((((ix * n[1] + iy) * n[2] + ih) * n[3] + ivx) * n[4] + ivy) * n[5] + iw

    def raw(self, target: int, subs):
        """The stored code for one cell, undecoded."""
        idx = self.flat_index(*subs)
        chunk = idx >> self.chunk_shift
        offset = (idx & (self.chunk_elements - 1)) * self.elem_bytes
        key = (target, chunk)
        if key not in self._fh:
            pat = self.m["targets"][target]["file_pattern"]
            path = os.path.join(self.root, (pat % chunk).replace("/", os.sep))
            self._fh[key] = open(path, "rb")
        fh = self._fh[key]
        fh.seek(offset)
        buf = fh.read(self.elem_bytes)
        fmt, _ = DTYPE[self.dtype]
        return struct.unpack(fmt, buf)[0]

    def cell(self, target: int, subs) -> float:
        """
        Seconds to go, or inf. Exactly the robot's two-shift lookup.

        An escape cell is unreachable and returns inf from here, deliberately:
        this is the number every routing decision compares, and an escape time
        must never win one. Read `escape` for the other half.
        """
        val = self.raw(target, subs)
        if self.dtype in ("u8", "u16"):
            if val == self.unreachable:
                return math.inf
            if self.escape_codes and val >= self.escape_base:
                return math.inf
            return val * self.scale
        if math.isnan(val) or val < 0:
            return math.inf
        return float(val)

    def escape(self, target: int, subs):
        """
        Seconds to reach a state that has a route, for a cell that has none;
        `None` if this cell is reachable or has no way out either.

        Only for a robot that is already stuck. Section 6 of TABLE_FORMAT.md.
        """
        val = self.raw(target, subs)
        if self.dtype in ("u8", "u16"):
            if not self.escape_codes or val == self.unreachable:
                return None
            if val < self.escape_base:
                return None
            r = val - self.escape_base
            return self.escape_scale * r * r
        if math.isnan(val) or val >= 0:
            return None
        return -float(val)

    def nearest_subs(self, state):
        out = []
        for k, v in enumerate(state):
            c = (v - self.lo[k]) / self.step[k]
            if self.wrap[k]:
                out.append(int(round(c)) % self.n[k])
            else:
                out.append(max(0, min(self.n[k] - 1, int(round(c)))))
        return out

    def close(self):
        for fh in self._fh.values():
            fh.close()


def check_escape(card: "Card", m: dict, fails: list, warns: list) -> None:
    """
    Validate the escape band: the codes that mean "unreachable, but here is
    the way out".

    Checked from the manifest and the bytes, not from anything the solver
    says it did, because the point of this file is to test the contract.
    """
    e = m["encoding"]
    base = e.get("escape_base", 0) or 0
    codes = e.get("escape_codes", 0) or 0
    esc_scale = e.get("escape_scale", 0.0) or 0.0
    on = bool((m.get("solver") or {}).get("escape"))
    isfloat = card.dtype in ("f16", "f32")
    print()
    # A float dtype carries the escape in the sign and has no band, so
    # `escape_codes` is legitimately 0 there and says nothing about whether
    # the pass ran. Keyed off dtype rather than off the band for that reason.
    if not on or (not codes and not isfloat):
        print("  escape band: none (table has no escape values)")
        return

    if isfloat:
        print(f"  escape: sign-encoded ({card.dtype} carries -seconds, "
              "no band)")
    else:
        top = esc_scale * (codes - 1) ** 2
        print(f"  escape band: {codes} codes from {base}, "
              f"{esc_scale:.3e} s/code^2, spanning 0..{top:.2f} s")

    if card.dtype in ("u8", "u16"):
        top = esc_scale * (codes - 1) ** 2
        # The band must sit strictly above every code a real time can use, or
        # a route and an escape are the same bits and the robot cannot tell
        # which it is holding.
        if base <= 0 or base >= card.unreachable:
            fails.append(f"escape_base {base} is not inside the code range "
                         f"below the unreachable sentinel {card.unreachable}")
        if base + codes != card.unreachable:
            fails.append(
                f"escape band does not stop at the sentinel: "
                f"escape_base {base} + escape_codes {codes} = {base + codes}, "
                f"expected {card.unreachable}")
        # The band has to reach as far as a value can, or escape times get
        # silently clipped at the top of it.
        cap = ((m.get("solver") or {}).get("value_cap_s"))
        if isinstance(cap, (int, float)) and top < cap - 1e-6:
            warns.append(f"escape band tops out at {top:.2f} s but value_cap "
                         f"is {cap} s, so long escapes saturate")
    if not isfloat and esc_scale <= 0:
        fails.append("escape band is declared but escape_scale is not positive")

    # An escape cell must still read as unreachable. That is the property the
    # whole design rests on: the robot compares `cell` and must never be able
    # to prefer a state inside an obstacle.
    fr = m["targets"][0].get("escape_frac")
    if fr is not None:
        print(f"    target 0: {fr * 100:.2f}% of cells carry an escape "
              f"({m['targets'][0].get('escape_of_unreached_frac', 0) * 100:.1f}%"
              " of those with no route)")

    # Walk a line of states and confirm the two decodes are exclusive
    # everywhere: a cell has a finite time, or an escape, never both.
    st = list(m["targets"][0]["state"])
    both = 0
    seen = 0
    for d in range(0, 200, 7):
        s = list(st)
        s[0] = min(s[0] + d, m["grid"]["max"][0])
        subs = card.nearest_subs(s)
        v = card.cell(0, subs)
        x = card.escape(0, subs)
        if x is not None:
            seen += 1
            if math.isfinite(v):
                both += 1
            if x < 0:
                fails.append("an escape time decoded negative")
    if both:
        fails.append(f"{both} cells decode as both reachable and escapable; "
                     "the escape band overlaps the value range")
    print(f"    probed {len(range(0, 200, 7))} states, {seen} with an escape, "
          f"{both} inconsistent")


def check_model(root: str, fails: list, warns: list) -> None:
    """
    Validate MODEL.JSON: the drivetrain the robot evaluates alongside the
    tables.

    Shapes are checked against the declared state and control vectors rather
    than against hard-coded 3/6, so the file can gain terms later without this
    needing to change.
    """
    path = os.path.join(root, "MODEL.JSON")
    if not os.path.exists(path):
        fails.append("MODEL.JSON is missing")
        return
    try:
        m = read_json(path)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fails.append(f"MODEL.JSON is not valid JSON: {e}")
        return

    ns = len(m["state"]["vector"])
    nu = len(m["control"]["vector"])
    no = len(m["output"]["vector"])
    print()
    print(f"  model: {no} outputs, state {ns}, control {nu}")
    print(f"    {m['equation']}")

    for name, rows, cols in (("A_s", no, ns), ("A_u", no, nu),
                             ("A_ss", no, ns), ("A_uu", no, nu),
                             ("A_sgn", no, ns), ("A_absv", no, ns)):
        M = m.get(name)
        if M is None:
            fails.append(f"MODEL.JSON has no {name}")
            continue
        if len(M) != rows or any(len(r) != cols for r in M):
            fails.append(f"MODEL.JSON {name} should be {rows}x{cols}, "
                         f"got {len(M)}x{len(M[0]) if M else 0}")
    if len(m.get("k", [])) != no:
        fails.append(f"MODEL.JSON k should have {no} entries")

    nz = m.get("nonzero_blocks", [])
    print(f"    non-zero blocks: {', '.join(nz) if nz else 'NONE'}")
    if not nz:
        fails.append("MODEL.JSON is entirely zero; the robot would never move")
    if "A_u" not in nz:
        fails.append("MODEL.JSON A_u is zero: control has no effect")
    if m.get("constant_zeroed"):
        print("    constant term zeroed, matching the solve")

    # Coulomb is smoothed, and the robot must use the same band or its
    # dynamics will not match the tables that were solved from them.
    if "A_sgn" in nz:
        cs = m.get("csign") or {}
        eps = cs.get("coulomb_eps")
        if not eps or not any(e > 0 for e in eps):
            fails.append("MODEL.JSON has a Coulomb block but no usable "
                         "coulomb_eps, so csign() is undefined")
        else:
            print("    coulomb_eps = %s" % [round(e, 3) for e in eps])

    # The knee is mandatory: the gains describe force delivered, not force
    # requested, so a card without one drives a robot that thinks it is
    # stronger than it is at full stick.
    sat = (m.get("control") or {}).get("saturation") or {}
    knee = sat.get("knee")
    if not isinstance(knee, (int, float)) or knee <= 0:
        fails.append("MODEL.JSON has no positive traction knee; the gains "
                     "were fitted against the saturated command")
    else:
        print("    traction knee = %.3f (saturate the command first)" % knee)
        if "tanh" not in str(sat.get("formula", "")):
            fails.append("MODEL.JSON declares a traction knee but no formula")

    # Control reaches position only through velocity: it enters as an
    # acceleration and nothing else. A_u carrying a position row, or any state
    # block carrying an x/y/h column, would break that.
    for name in ("A_s", "A_ss", "A_sgn", "A_absv"):
        blk = m.get(name) or []
        if any(any(c != 0 for c in r[:3]) for r in blk if len(r) >= 3):
            fails.append(f"MODEL.JSON {name} has a non-zero x/y/h column; "
                         "the dynamics must not depend on where the robot is")
    if any(any(c != 0 for c in r) for r in (m.get("A_uu") or [])):
        fails.append("MODEL.JSON A_uu is non-zero; control-squared terms are "
                     "not part of this model")

    # The tables index field-frame velocity but this model is body-frame; that
    # mismatch is the easiest thing to get wrong on the robot, so make sure the
    # file says so.
    if "body" not in json.dumps(m.get("state", {})).lower():
        warns.append("MODEL.JSON does not flag that its velocities are "
                     "body-frame while the tables are field-frame")


def main(root: str) -> int:
    card = Card(root)
    m = card.m
    fails, warns = [], []

    print(f"  manifest schema      {m['schema_version']}")
    print(f"  grid                 {m['grid']['n']}  = {m['grid']['total_cells']:,} cells")
    print(f"  encoding             {card.dtype} ({card.elem_bytes} B), scale {card.scale}")
    print(f"  chunk                2^{card.chunk_shift} = {card.chunk_elements:,} elements")
    print(f"  targets              {len(m['targets'])}")
    print()

    total_cells = m["grid"]["total_cells"]
    if math.prod(card.n) != total_cells:
        fails.append(f"total_cells {total_cells} != prod(n) {math.prod(card.n)}")

    # The stored x/y span is the field inset by the footprint and the wall
    # clearance, so it has to sit inside the field it was cut from -- and it
    # has to be strictly inside, because a span equal to the field is the old
    # bug back again: a boundary enforced on the tracking point alone, letting
    # the chassis hang through the wall. See section 2 of TABLE_FORMAT.md.
    fb = m["grid"].get("field_bounds")
    if fb is None:
        warns.append("manifest has no grid.field_bounds; written by a solver "
                     "from before the wall was treated as an obstacle")
    else:
        inset = [card.lo[0] - fb[0], card.lo[1] - fb[1],
                 fb[2] - m["grid"]["max"][0], fb[3] - m["grid"]["max"][1]]
        print(f"  field                {fb[2]-fb[0]:.0f} x {fb[3]-fb[1]:.0f} cm, "
              f"table inset {min(inset):.1f}..{max(inset):.1f} cm")
        if min(inset) < -1e-6:
            fails.append(
                f"the table spans {card.lo[0]:.1f}..{m['grid']['max'][0]:.1f} x "
                f"{card.lo[1]:.1f}..{m['grid']['max'][1]:.1f}, which reaches "
                f"outside the field {fb}: states are stored that the robot "
                "cannot occupy")
        wc = (m.get("solver") or {}).get("wall_clearance_cm")
        if wc is not None and min(inset) + 1e-6 < wc:
            fails.append(
                f"the table is inset {min(inset):.1f} cm but the manifest "
                f"claims {wc} cm of wall clearance; the footprint would reach "
                "the wall")

    for t in m["targets"]:
        ti = t["index"]
        name = t["name"]
        # Bytes and chunk count must follow from the grid, not be taken on trust.
        expect_bytes = total_cells * card.elem_bytes
        if t["bytes"] != expect_bytes:
            fails.append(f"[{name}] bytes {t['bytes']} != expected {expect_bytes}")
        expect_chunks = -(-total_cells // card.chunk_elements)
        if t["n_chunks"] != expect_chunks:
            fails.append(f"[{name}] n_chunks {t['n_chunks']} != expected {expect_chunks}")

        h = hashlib.sha256()
        seen = 0
        for c in range(t["n_chunks"]):
            p = os.path.join(root, (t["file_pattern"] % c).replace("/", os.sep))
            if not os.path.exists(p):
                fails.append(f"[{name}] missing chunk {p}")
                break
            with open(p, "rb") as fh:
                data = fh.read()
            h.update(data)
            seen += len(data)
            if len(data) > (1 << 32) - 1:
                fails.append(f"[{name}] chunk {c} exceeds FAT32 file limit")
        else:
            if seen != t["bytes"]:
                fails.append(f"[{name}] chunk bytes {seen} != declared {t['bytes']}")
            digest = h.hexdigest()
            if digest != t["sha256"]:
                fails.append(f"[{name}] sha256 mismatch")

        # The target's own cell must be zero: it is the seed of the recursion.
        at = card.cell(ti, card.nearest_subs(t["state"]))
        reach = t.get("reached_frac", float("nan"))
        flag = ""
        if not (at < 0.25):
            fails.append(f"[{name}] value at the target is {at}, expected ~0")
            flag = "  <-- BAD"
        print(f"  [{ti}] {name:<14} V(target) = {at:7.3f} s   "
              f"reached {reach*100:5.1f}%   chunks {t['n_chunks']}{flag}")

        if reach == reach and reach < 0.30:
            warns.append(f"[{name}] only {reach*100:.1f}% of cells reachable; "
                         f"raise iterations or value_cap")

    # Times should grow as you move away from the target.
    t0 = m["targets"][0]
    st = list(t0["state"])
    prev, mono = -1.0, True
    print()
    print("  time-to-go along +x from target 0:")
    for d in (0, 20, 40, 80, 120):
        s = list(st)
        s[0] = min(s[0] + d, m["grid"]["max"][0])
        v = card.cell(0, card.nearest_subs(s))
        print(f"    +{d:3d} cm -> {v:7.3f} s")
        if v < prev - 1e-6:
            mono = False
        prev = v
    if not mono:
        warns.append("time-to-go is not monotone moving away from the target; "
                     "expected for detours around obstacles, suspicious otherwise")

    # Velocity axes must be FIELD frame. Approaching the target should beat
    # standing still, which should beat receding from it. If the axes were
    # body-frame this ordering would depend on heading and generally break --
    # which is exactly the kind of silent regression worth catching here.
    probe = list(st)
    probe[0] = min(probe[0] + 100.0, m["grid"]["max"][0])
    speed = min(100.0, m["grid"]["max"][3])
    sign = -1.0 if probe[0] > st[0] else 1.0      # point velocity at the target
    vs = {}
    for label, vx in (("toward", sign * speed), ("rest", 0.0),
                      ("away", -sign * speed)):
        s = list(probe)
        s[3], s[4] = vx, 0.0
        vs[label] = card.cell(0, card.nearest_subs(s))
    print()
    print("  field-frame velocity check, 100 cm from target 0:")
    for k in ("toward", "rest", "away"):
        print(f"    v {k:<7} -> {vs[k]:7.3f} s")
    if all(math.isfinite(v) for v in vs.values()):
        if not (vs["toward"] <= vs["rest"] <= vs["away"]):
            fails.append(
                "velocity axes do not behave as field frame: expected "
                f"toward <= rest <= away, got {vs['toward']:.3f}, "
                f"{vs['rest']:.3f}, {vs['away']:.3f}")
    else:
        warns.append("velocity-frame check skipped; the probe state is "
                     "unreachable (obstacle, or too coarse a grid)")

    check_escape(card, m, fails, warns)

    card.close()

    check_model(root, fails, warns)

    print()
    for w in warns:
        print(f"  WARN  {w}")
    for f in fails:
        print(f"  FAIL  {f}")
    print()
    print(f"  {'PASS' if not fails else 'FAIL'}  "
          f"({len(fails)} failures, {len(warns)} warnings)")
    return 0 if not fails else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify_tables.py <dir-or-drive>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
