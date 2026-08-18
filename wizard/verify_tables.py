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
        self._fh: dict[tuple[int, int], object] = {}

    def flat_index(self, ix, iy, ih, ivx, ivy, iw) -> int:
        n = self.n
        return ((((ix * n[1] + iy) * n[2] + ih) * n[3] + ivx) * n[4] + ivy) * n[5] + iw

    def cell(self, target: int, subs) -> float:
        """Seconds to go, or inf. Exactly the robot's two-shift lookup."""
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
        raw = fh.read(self.elem_bytes)
        fmt, _ = DTYPE[self.dtype]
        val = struct.unpack(fmt, raw)[0]
        if self.dtype in ("u8", "u16"):
            return math.inf if val == self.unreachable else val * self.scale
        return math.inf if math.isnan(val) else float(val)

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
                             ("A_ss", no, ns), ("A_uu", no, nu)):
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
