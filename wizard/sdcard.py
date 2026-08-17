#!/usr/bin/env python3
"""
SD card detection and safe writing.

Everything in Peregrine that can destroy data lives in this one file, kept
small on purpose so it can be read end to end.

The wipe is guarded by five independent checks, and *all* of them must pass:

  1. The drive must not be on a disk Windows marks as system or boot.
  2. The drive letter must not host Windows, the user profile, the Peregrine
     workspace, or this source tree.
  3. The disk must look genuinely external -- removable, or on a USB/SD/MMC
     bus. (Some built-in card readers report "Fixed", which is why bus type
     is accepted as an alternative rather than requiring removability alone.)
  4. The root must not contain the signature directories of a system volume.
  5. The caller must supply the drive letter again, typed by the user.

It deletes files. It never formats, never touches partition tables, and never
operates on a path that is not the root of a verified removable volume.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

# Directory names that mean "this is a system volume, do not touch".
SYSTEM_MARKERS = {
    "windows", "program files", "program files (x86)", "programdata",
    "system volume information.sys", "$winreagent", "perflogs", "users",
    "bootmgr", "pagefile.sys", "hiberfil.sys", "swapfile.sys",
}
# Buses that indicate genuinely external media.
EXTERNAL_BUSES = {"USB", "SD", "MMC"}

# Windows PowerShell 5.1 has no ConvertTo-Json -AsArray, so the result is
# wrapped in @() to force an array even when a single volume matches.
_QUERY = r"""
$r = Get-Volume | Where-Object { $_.DriveLetter } | ForEach-Object {
  $v = $_
  $p = Get-Partition -DriveLetter $v.DriveLetter -ErrorAction SilentlyContinue |
       Select-Object -First 1
  $d = if ($p) { Get-Disk -Number $p.DiskNumber -ErrorAction SilentlyContinue } else { $null }
  [PSCustomObject]@{
    Letter     = [string]$v.DriveLetter
    Label      = [string]$v.FileSystemLabel
    FileSystem = [string]$v.FileSystem
    DriveType  = [string]$v.DriveType
    Size       = [int64]$v.Size
    Free       = [int64]$v.SizeRemaining
    DiskNumber = if ($d) { [int]$d.Number } else { -1 }
    BusType    = if ($d) { [string]$d.BusType } else { "" }
    IsSystem   = if ($d) { [bool]$d.IsSystem } else { $true }
    IsBoot     = if ($d) { [bool]$d.IsBoot } else { $true }
    FriendlyName = if ($d) { [string]$d.FriendlyName } else { "" }
  }
}
ConvertTo-Json -Depth 3 -InputObject @($r)
"""


def list_volumes() -> list[dict]:
    """Every lettered volume, with the facts the safety checks need."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _QUERY],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"could not enumerate drives: {e}") from e
    if out.returncode != 0:
        raise RuntimeError(f"could not enumerate drives: {out.stderr.strip()}")
    txt = out.stdout.strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"unexpected drive listing: {e}") from e
    return data if isinstance(data, list) else [data]


def _protected_roots(workspace: str | None) -> set[str]:
    """Drive letters we must never write to, whatever else is true."""
    roots = set()
    for env in ("SystemDrive", "SystemRoot", "windir", "ProgramFiles",
                "ProgramData", "USERPROFILE", "LOCALAPPDATA", "APPDATA"):
        v = os.environ.get(env)
        if v and len(v) > 1 and v[1] == ":":
            roots.add(v[0].upper())
    for p in (workspace, os.path.abspath(__file__), os.getcwd()):
        if p and len(p) > 1 and p[1] == ":":
            roots.add(p[0].upper())
    roots.add("C")
    return roots


def assess(vol: dict, workspace: str | None = None) -> dict:
    """
    Decide whether a volume may be wiped, and say why not when it may not.

    Returns the volume dict with `safe` and `reasons` added. A volume is only
    safe when every reason list is empty -- the checks are AND-ed, never
    OR-ed, so a single failure is disqualifying.
    """
    reasons: list[str] = []
    letter = (vol.get("Letter") or "").upper()

    if not letter:
        reasons.append("no drive letter")

    if vol.get("IsSystem"):
        reasons.append("disk is marked as the SYSTEM disk")
    if vol.get("IsBoot"):
        reasons.append("disk is marked as the BOOT disk")

    if letter in _protected_roots(workspace):
        reasons.append(f"{letter}: holds Windows, your profile, or the workspace")

    drive_type = (vol.get("DriveType") or "").lower()
    bus = (vol.get("BusType") or "").upper()
    if drive_type != "removable" and bus not in EXTERNAL_BUSES:
        reasons.append(
            f"not external (DriveType={vol.get('DriveType')}, BusType={bus or '?'})")

    fs = (vol.get("FileSystem") or "").upper()
    if fs and fs != "FAT32":
        reasons.append(f"filesystem is {fs}; the Control Hub only accepts FAT32")

    # Look at what is actually on the volume, as an independent check on the
    # metadata above.
    root = f"{letter}:\\"
    if letter and os.path.isdir(root):
        try:
            entries = {e.lower() for e in os.listdir(root)}
            hits = entries & SYSTEM_MARKERS
            if hits:
                reasons.append(
                    f"root contains system directories ({', '.join(sorted(hits))})")
        except OSError as e:
            reasons.append(f"cannot read {root}: {e}")
    elif letter:
        reasons.append(f"{root} is not readable")

    out = dict(vol)
    out["safe"] = not reasons
    out["reasons"] = reasons
    return out


def describe(vol: dict) -> str:
    gb = vol.get("Size", 0) / 2**30
    free = vol.get("Free", 0) / 2**30
    return (f"{vol.get('Letter','?')}:  "
            f"{vol.get('Label') or '(no label)':<16} "
            f"{vol.get('FileSystem') or '?':<6} "
            f"{gb:7.1f} GB ({free:.1f} free)  "
            f"{vol.get('DriveType','?')}/{vol.get('BusType') or '?'}  "
            f"{vol.get('FriendlyName','')}")


def inventory(letter: str) -> tuple[int, int, list[str]]:
    """Count what a wipe would remove, so the user sees it before agreeing."""
    root = f"{letter.upper()}:\\"
    files = 0
    total = 0
    top: list[str] = []
    for name in os.listdir(root):
        top.append(name)
        p = os.path.join(root, name)
        if os.path.isfile(p):
            files += 1
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
        else:
            for dirpath, _, fs in os.walk(p):
                for f in fs:
                    files += 1
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
    return files, total, sorted(top)


def wipe(letter: str, confirm_letter: str, workspace: str | None = None) -> int:
    """
    Delete every file and directory at the root of a verified removable volume.

    Deletes only. Does not format, does not touch the partition table. The
    volume is re-assessed here rather than trusting an earlier check, so a
    card swapped between confirmation and execution cannot slip through.
    """
    letter = letter.strip().rstrip(":").upper()
    if letter != confirm_letter.strip().rstrip(":").upper():
        raise ValueError("confirmation letter does not match")
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError(f"not a drive letter: {letter!r}")

    vols = [v for v in list_volumes() if (v.get("Letter") or "").upper() == letter]
    if not vols:
        raise RuntimeError(f"{letter}: is no longer present")
    a = assess(vols[0], workspace)
    if not a["safe"]:
        raise RuntimeError("refusing to wipe " + letter + ": " +
                           "; ".join(a["reasons"]))

    root = f"{letter}:\\"
    removed = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        # Never follow a link off the volume.
        if os.path.islink(p):
            os.unlink(p)
            removed += 1
            continue
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.chmod(p, 0o666)
            except OSError:
                pass
            try:
                os.remove(p)
            except OSError:
                continue
        removed += 1
    return removed


# Build artefacts that belong in the workspace, not on the card. The run log
# in particular grows with iteration count and would be pure dead weight.
CARD_EXCLUDE = {"solve.log", "config.json"}


def card_payload(src_dir: str) -> list[tuple[str, str]]:
    """The (source, relative) pairs that actually get written to a card."""
    items: list[tuple[str, str]] = []
    for dirpath, _, files in os.walk(src_dir):
        for f in files:
            if f in CARD_EXCLUDE:
                continue
            s = os.path.join(dirpath, f)
            items.append((s, os.path.relpath(s, src_dir)))
    return items


def copy_image(src_dir: str, letter: str, progress=None) -> tuple[int, int]:
    """Copy a solved card image onto the card, preserving the layout."""
    letter = letter.strip().rstrip(":").upper()
    dst_root = f"{letter}:\\"
    if not os.path.isdir(dst_root):
        raise RuntimeError(f"{dst_root} not available")

    items = [(s, os.path.join(dst_root, rel)) for s, rel in card_payload(src_dir)]
    total = sum(os.path.getsize(s) for s, _ in items)

    done = 0
    for i, (s, d) in enumerate(items):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        done += os.path.getsize(s)
        if progress:
            progress(i + 1, len(items), done, total)
    return len(items), total


if __name__ == "__main__":
    # Read-only listing, safe to run any time.
    ws = sys.argv[1] if len(sys.argv) > 1 else None
    for v in list_volumes():
        a = assess(v, ws)
        mark = "OK " if a["safe"] else "NO "
        print(mark, describe(v))
        for r in a["reasons"]:
            print("      -", r)
