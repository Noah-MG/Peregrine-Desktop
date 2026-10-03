#!/usr/bin/env python3
"""
SD card detection and safe writing.

Everything in Peregrine that can destroy data lives in this one file, kept
small on purpose so it can be read end to end.

The wipe is guarded by five independent checks, and *all* of them must pass:

  1. The drive must not be on a disk the OS marks as system or boot.
  2. The volume must not host the OS, the user profile, the Peregrine
     workspace, or this source tree.
  3. The disk must look genuinely external -- removable, or on a USB/SD/MMC
     bus. (Some built-in card readers report "Fixed", which is why bus type
     is accepted as an alternative rather than requiring removability alone.)
  4. The root must not contain the signature directories of a system volume.
  5. The caller must supply the volume's name again, typed by the user.

It deletes files. It never formats, never touches partition tables, and never
operates on a path that is not the root of a verified removable volume.

A volume is named the way its platform names it: a drive letter on Windows
(`E`), and the name it is mounted under in /Volumes on macOS (`NO NAME`).
Every volume dict carries that as `Id`, and the path of its root as `Root`,
so callers never have to build either themselves.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# What the user types to pick a volume, in this platform's words.
ID_NOUN = "drive letter" if IS_WINDOWS else "volume name"

# Directory names that mean "this is a system volume, do not touch".
# A FAT32 card that has been in a Mac carries .Spotlight-V100, .fseventsd and
# .Trashes; those are deliberately not markers.
SYSTEM_MARKERS = {
    "windows", "program files", "program files (x86)", "programdata",
    "system volume information.sys", "$winreagent", "perflogs", "users",
    "bootmgr", "pagefile.sys", "hiberfil.sys", "swapfile.sys",
} if IS_WINDOWS else {
    "system", "library", "applications", "users", "private", "volumes",
    "cores", "usr", "bin", "sbin", "etc", "var",
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
    """Every mounted volume, with the facts the safety checks need."""
    if IS_WINDOWS:
        vols = _list_windows()
        for v in vols:
            letter = (v.get("Letter") or "").upper()
            v["Id"] = letter
            v["Root"] = f"{letter}:\\" if letter else ""
        return vols
    if IS_MAC:
        return _list_mac()
    raise RuntimeError(f"SD card detection is not supported on {sys.platform}")


def _list_windows() -> list[dict]:
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


# --------------------------------------------------------------------------
# macOS
#
# There is no single query like Get-Volume, so each mount under /Volumes is
# asked about separately with `diskutil info -plist`, and the answer is
# reshaped into the same fields the Windows query produces. The boot disk is
# whatever "/" lives on, including the physical disk under its APFS
# container -- the user's data volume is a sibling of "/" in that container.
# --------------------------------------------------------------------------

VOLUMES = "/Volumes"

# diskutil's BusProtocol, in the Windows BusType vocabulary EXTERNAL_BUSES uses.
_MAC_BUS = {"USB": "USB", "SECURE DIGITAL": "SD", "SD": "SD", "MMC": "MMC"}


def _diskutil_info(target: str) -> dict | None:
    try:
        out = subprocess.run(["diskutil", "info", "-plist", target],
                             capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"could not run diskutil: {e}") from e
    if out.returncode != 0:
        return None
    try:
        return plistlib.loads(out.stdout)
    except Exception:
        return None


def _whole_disks(info: dict) -> set[str]:
    """Every whole disk a volume sits on: its own, and under APFS, the
    physical disk beneath the container."""
    disks = set()
    for k in ("ParentWholeDisk", "APFSContainerReference", "DeviceIdentifier"):
        m = re.match(r"(disk\d+)", info.get(k) or "")
        if m:
            disks.add(m.group(1))
    for store in info.get("APFSPhysicalStores") or []:
        m = re.match(r"(disk\d+)", store.get("APFSPhysicalStore") or "")
        if m:
            disks.add(m.group(1))
    return disks


def _mac_filesystem(info: dict) -> str:
    """The FAT variant in the Windows spelling, so one check covers both."""
    names = " ".join(str(info.get(k) or "") for k in
                     ("FilesystemName", "FilesystemUserVisibleName",
                      "FilesystemType")).upper()
    for fat in ("FAT32", "FAT16", "FAT12"):
        if fat in names:
            return fat
    if "EXFAT" in names:
        return "exFAT"
    return (info.get("FilesystemName") or info.get("FilesystemType") or "").strip()


def mac_volume(root: str, info: dict | None, boot_disks: set[str]) -> dict:
    """One volume in the shape the Windows query gives, from diskutil info.

    Pure: everything it knows comes from its arguments, which is what lets
    the self-test feed it the plists of disks this machine does not have.
    An unanswered query is treated as a system disk, as on Windows -- not
    knowing is not a reason to allow a wipe.
    """
    name = os.path.basename(root.rstrip("/"))
    if info is None:
        return {"Id": name, "Root": root, "Label": name, "FileSystem": "",
                "DriveType": "Unknown", "Size": 0, "Free": 0, "Disk": "",
                "BusType": "", "IsSystem": True, "IsBoot": True,
                "FriendlyName": "(diskutil could not describe it)"}
    disks = _whole_disks(info)
    bus = (info.get("BusProtocol") or "").strip()
    # A mounted .dmg reports itself as removable media. It is a file on some
    # other disk, as a mounted VHD is on Windows, and is refused the same way.
    if bus.upper() == "DISK IMAGE":
        drive_type = "Virtual"
    elif info.get("RemovableMedia") or info.get("Removable"):
        drive_type = "Removable"
    elif info.get("Internal", True):
        drive_type = "Fixed"
    else:
        drive_type = "External"
    return {
        "Id": name,
        "Root": root,
        "Label": info.get("VolumeName") or name,
        "FileSystem": _mac_filesystem(info),
        "DriveType": drive_type,
        "Size": int(info.get("TotalSize") or info.get("Size") or 0),
        "Free": int(info.get("FreeSpace") or info.get("APFSContainerFree") or 0),
        "Disk": info.get("ParentWholeDisk") or "",
        "BusType": _MAC_BUS.get(bus.upper(), bus.upper()),
        "IsSystem": bool(disks & boot_disks),
        "IsBoot": info.get("MountPoint") == "/",
        "FriendlyName": (info.get("MediaName")
                         or info.get("IORegistryEntryName") or ""),
    }


def _list_mac() -> list[dict]:
    boot = _diskutil_info("/")
    if boot is None:
        raise RuntimeError("could not identify the boot disk with diskutil")
    boot_disks = _whole_disks(boot)
    vols = []
    try:
        names = sorted(os.listdir(VOLUMES))
    except OSError as e:
        raise RuntimeError(f"could not list {VOLUMES}: {e}") from e
    for name in names:
        root = os.path.join(VOLUMES, name)
        # "Macintosh HD" is a symlink to "/", not a mount of its own.
        if os.path.islink(root) or not os.path.ismount(root):
            continue
        vols.append(mac_volume(root, _diskutil_info(root), boot_disks))
    return vols


# --------------------------------------------------------------------------
# Naming a volume
# --------------------------------------------------------------------------

def norm_id(s: str) -> str:
    """What the user typed, reduced to something comparable with an `Id`."""
    s = (s or "").strip()
    if IS_WINDOWS:
        return s.rstrip(":\\").upper()
    s = s.rstrip("/")
    if s.startswith(VOLUMES + "/"):
        s = s[len(VOLUMES) + 1:]
    # The Mac's own filesystems compare names case-insensitively, so a
    # confirmation should too.
    return s.casefold()


def show_id(vol_id: str) -> str:
    """A volume's name the way this platform writes it: `E:` or `NO NAME`."""
    return f"{vol_id}:" if IS_WINDOWS else vol_id


def find(vols: list[dict], typed: str) -> dict | None:
    """The volume in `vols` that `typed` names, if any."""
    want = norm_id(typed)
    if not want:
        return None
    for v in vols:
        if norm_id(v.get("Id") or "") == want:
            return v
    return None


def volume_root(vol_id: str) -> str:
    if IS_WINDOWS:
        return f"{norm_id(vol_id)}:\\"
    return os.path.join(VOLUMES, vol_id)


def on_volume(path: str, vol: dict) -> bool:
    """Is `path` stored on `vol`? Used to refuse wiping the copy's source."""
    path = os.path.abspath(path)
    if IS_WINDOWS:
        return os.path.splitdrive(path)[0].upper() == f"{vol['Id']}:"
    try:
        return os.stat(path).st_dev == os.stat(vol["Root"]).st_dev
    except OSError:
        root = vol["Root"].rstrip("/") + "/"
        return (path.rstrip("/") + "/").startswith(root)


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


def _mac_protected(vol: dict, workspace: str | None) -> str | None:
    """Why a Mac volume must not be written to, or None.

    The same question as `_protected_roots`, answered by device rather than
    by drive letter: a volume is protected if it is the filesystem that "/",
    the home folder, the workspace, this source tree or the current
    directory live on.
    """
    try:
        dev = os.stat(vol["Root"]).st_dev
    except OSError:
        return None             # `assess` reports the unreadable root itself
    for what, p in (("the system volume", "/"),
                    ("your home folder", os.path.expanduser("~")),
                    ("the workspace", workspace),
                    ("this source tree", os.path.abspath(__file__)),
                    ("the current directory", os.getcwd())):
        if not p:
            continue
        try:
            if os.stat(p).st_dev == dev:
                return f"{vol['Root']} holds {what}"
        except OSError:
            continue
    return None


def assess(vol: dict, workspace: str | None = None) -> dict:
    """
    Decide whether a volume may be wiped, and say why not when it may not.

    Returns the volume dict with `safe` and `reasons` added. A volume is only
    safe when every reason list is empty -- the checks are AND-ed, never
    OR-ed, so a single failure is disqualifying.
    """
    reasons: list[str] = []
    vol_id = vol.get("Id") or ""
    root = vol.get("Root") or ""

    if not vol_id:
        reasons.append(f"no {ID_NOUN}")

    if vol.get("IsSystem"):
        reasons.append("disk is marked as the SYSTEM disk")
    if vol.get("IsBoot"):
        reasons.append("disk is marked as the BOOT disk")

    if IS_WINDOWS:
        if vol_id.upper() in _protected_roots(workspace):
            reasons.append(f"{vol_id}: holds Windows, your profile, or the "
                           f"workspace")
    elif root:
        why = _mac_protected(vol, workspace)
        if why:
            reasons.append(why)

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
    if root and os.path.isdir(root):
        try:
            entries = {e.lower() for e in os.listdir(root)}
            hits = entries & SYSTEM_MARKERS
            if hits:
                reasons.append(
                    f"root contains system directories ({', '.join(sorted(hits))})")
        except OSError as e:
            reasons.append(f"cannot read {root}: {e}")
    elif vol_id:
        reasons.append(f"{root or vol_id} is not readable")

    out = dict(vol)
    out["safe"] = not reasons
    out["reasons"] = reasons
    return out


def describe(vol: dict) -> str:
    gb = vol.get("Size", 0) / 2**30
    free = vol.get("Free", 0) / 2**30
    name = show_id(vol.get("Id") or "?")
    # On a Mac the name *is* the label, so it is not printed twice.
    head = (f"{name}  {vol.get('Label') or '(no label)':<16} " if IS_WINDOWS
            else f"{name:<18} ")
    return (head +
            f"{vol.get('FileSystem') or '?':<6} "
            f"{gb:7.1f} GB ({free:.1f} free)  "
            f"{vol.get('DriveType','?')}/{vol.get('BusType') or '?'}  "
            f"{vol.get('FriendlyName','')}")


def inventory(vol_id: str) -> tuple[int, int, list[str]]:
    """Count what a wipe would remove, so the user sees it before agreeing."""
    root = volume_root(vol_id)
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


def wipe(vol_id: str, confirm_id: str, workspace: str | None = None) -> int:
    """
    Delete every file and directory at the root of a verified removable volume.

    Deletes only. Does not format, does not touch the partition table. The
    volume is re-assessed here rather than trusting an earlier check, so a
    card swapped between confirmation and execution cannot slip through.
    """
    if norm_id(vol_id) != norm_id(confirm_id):
        raise ValueError(f"confirmation {ID_NOUN} does not match")
    if IS_WINDOWS:
        letter = norm_id(vol_id)
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"not a drive letter: {letter!r}")
    elif not norm_id(vol_id) or "/" in norm_id(vol_id):
        raise ValueError(f"not a volume name: {vol_id!r}")

    vol = find(list_volumes(), vol_id)
    if vol is None:
        raise RuntimeError(f"{show_id(vol_id)} is no longer present")
    a = assess(vol, workspace)
    if not a["safe"]:
        raise RuntimeError("refusing to wipe " + show_id(vol["Id"]) + ": " +
                           "; ".join(a["reasons"]))

    root = vol["Root"]
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


def copy_image(src_dir: str, vol_id: str, progress=None) -> tuple[int, int]:
    """Copy a solved card image onto the card, preserving the layout."""
    dst_root = volume_root(vol_id)
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
    if IS_MAC:
        _drop_appledouble(dst_root, [d for _, d in items])
    # Windows mounts removable drives without a write cache; a Mac does not,
    # so without this the read-back verify can pass from cache while the card
    # itself is still half written.
    if hasattr(os, "sync"):
        os.sync()
    return len(items), total


def _drop_appledouble(root: str, written: list[str]) -> None:
    """Delete the `._` files macOS left beside what was just written.

    macOS tags every file a process creates with extended attributes, and
    FAT32 has nowhere to keep them, so each one is written to a `._<name>`
    sidecar instead. On the card that is one long filename -- several
    directory entries -- beside every 8.3 chunk name, and the format allows
    nothing on the card but the tables. Only sidecars of files and
    directories this copy wrote are touched.
    """
    names = set()
    for d in written:
        while True:
            names.add(d)
            parent = os.path.dirname(d)
            if parent == d or os.path.samefile(parent, root):
                break
            d = parent
    for p in names:
        side = os.path.join(os.path.dirname(p), "._" + os.path.basename(p))
        try:
            os.remove(side)
        except FileNotFoundError:
            pass


def eject(vol_id: str) -> tuple[bool, str]:
    """Unmount the card so it can be pulled. macOS only; Windows needs none."""
    if not IS_MAC:
        return True, ""
    try:
        r = subprocess.run(["diskutil", "eject", volume_root(vol_id)],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


# --------------------------------------------------------------------------
# Self-test
#
# Everything here that can be checked without a card in the slot: that the
# macOS diskutil answers are read the way the Windows ones are, that every
# guard refuses what it should, and that names are matched the way the user
# will type them. Touches nothing outside a temporary directory.
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile

    npass = nfail = 0

    def check(name, cond, detail=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  pass  {name}")
        else:
            nfail += 1
            print(f"  FAIL  {name}  {detail}")

    # What diskutil says about "/" on an Apple-silicon Mac, trimmed to the
    # keys that are read, and about the cards that might sit beside it.
    boot = {"MountPoint": "/", "ParentWholeDisk": "disk3",
            "APFSContainerReference": "disk3", "DeviceIdentifier": "disk3s1s1",
            "APFSPhysicalStores": [{"APFSPhysicalStore": "disk0s2"}],
            "Internal": True, "RemovableMedia": False,
            "BusProtocol": "Apple Fabric", "FilesystemName": "APFS"}
    boot_disks = _whole_disks(boot)
    sd_slot = {"MountPoint": "/Volumes/NO NAME", "VolumeName": "NO NAME",
               "ParentWholeDisk": "disk4", "DeviceIdentifier": "disk4s1",
               "Internal": True, "RemovableMedia": True,
               "BusProtocol": "Secure Digital", "FilesystemType": "msdos",
               "FilesystemName": "MS-DOS FAT32", "TotalSize": 31 * 2**30,
               "FreeSpace": 30 * 2**30, "MediaName": "SDXC Reader"}
    usb_reader = dict(sd_slot, ParentWholeDisk="disk5", DeviceIdentifier="disk5s1",
                      Internal=False, RemovableMedia=False, BusProtocol="USB")
    exfat = dict(sd_slot, FilesystemType="exfat", FilesystemName="ExFAT")
    ssd = dict(sd_slot, ParentWholeDisk="disk6", DeviceIdentifier="disk6s1",
               Internal=False, RemovableMedia=False, BusProtocol="PCI-Express")
    dmg = dict(sd_slot, ParentWholeDisk="disk7", DeviceIdentifier="disk7s1",
               BusProtocol="Disk Image")
    sibling = dict(sd_slot, ParentWholeDisk="disk3", DeviceIdentifier="disk3s7",
                   APFSContainerReference="disk3")

    print("\n[1] diskutil answers read like Get-Volume ones")
    check("the boot disk includes the physical disk under its container",
          boot_disks == {"disk0", "disk3"}, str(boot_disks))
    v = mac_volume("/Volumes/NO NAME", sd_slot, boot_disks)
    check("the volume is named by its mount", v["Id"] == "NO NAME", v["Id"])
    check("FAT32 is spelled as on Windows", v["FileSystem"] == "FAT32",
          v["FileSystem"])
    check("the built-in SD slot is removable on the SD bus",
          (v["DriveType"], v["BusType"]) == ("Removable", "SD"),
          f"{v['DriveType']}/{v['BusType']}")
    check("and is not the system disk", not v["IsSystem"] and not v["IsBoot"])
    u = mac_volume("/Volumes/NO NAME", usb_reader, boot_disks)
    check("a USB reader is on the USB bus",
          (u["DriveType"], u["BusType"]) == ("External", "USB"),
          f"{u['DriveType']}/{u['BusType']}")
    check("exFAT is reported as exFAT",
          mac_volume("/Volumes/X", exfat, boot_disks)["FileSystem"] == "exFAT")
    check("a volume in the boot container is the system disk",
          mac_volume("/Volumes/X", sibling, boot_disks)["IsSystem"])
    nobody = mac_volume("/Volumes/share", None, boot_disks)
    check("a volume diskutil cannot describe is treated as system",
          nobody["IsSystem"] and nobody["IsBoot"])

    print("\n[2] every guard refuses on its own")
    with tempfile.TemporaryDirectory() as tmp:
        card = os.path.join(tmp, "NO NAME")
        os.makedirs(os.path.join(card, "TABLES"))
        os.makedirs(os.path.join(card, ".Spotlight-V100"))

        def reasons(info, root=card):
            vol = mac_volume(root, info, boot_disks)
            # The temp directory lives on the system volume, so the device
            # guard always fires here; it is checked on its own below and
            # set aside for the rest.
            return [r for r in assess(vol)["reasons"] if " holds " not in r]

        if IS_WINDOWS:
            print("  (skipped: the macOS guards do not run on Windows)")
        else:
            check("a good card passes every other guard",
                  reasons(sd_slot) == [], str(reasons(sd_slot)))
            check("a USB reader passes too", reasons(usb_reader) == [],
                  str(reasons(usb_reader)))
            check("exFAT is refused",
                  any("EXFAT" in r.upper() for r in reasons(exfat)), str(reasons(exfat)))
            check("an external SSD is not external enough",
                  any("not external" in r for r in reasons(ssd)), str(reasons(ssd)))
            check("a mounted disk image is not external",
                  any("not external" in r for r in reasons(dmg)), str(reasons(dmg)))
            check("the boot container is refused",
                  any("SYSTEM" in r for r in reasons(sibling)),
                  str(reasons(sibling)))
            os.makedirs(os.path.join(card, "Library"))
            check("a root that looks like a Mac system volume is refused",
                  any("system directories" in r for r in reasons(sd_slot)),
                  str(reasons(sd_slot)))
            os.rmdir(os.path.join(card, "Library"))
            check("a missing root is refused",
                  any("not readable" in r for r in
                      reasons(sd_slot, os.path.join(tmp, "gone"))))
            whole = assess(mac_volume(card, sd_slot, boot_disks))
            check("a volume this machine's files live on is refused",
                  not whole["safe"] and any(" holds " in r
                                            for r in whole["reasons"]),
                  str(whole["reasons"]))

    print("\n[3] the copy leaves no macOS sidecars behind it")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "card")
        os.makedirs(os.path.join(root, "TABLES"))
        written = [os.path.join(root, "MANIFEST.JSON"),
                   os.path.join(root, "TABLES", "T00C0000.BIN")]
        for p in written + [os.path.join(root, "KEEP.TXT")]:
            open(p, "wb").close()
            side = os.path.join(os.path.dirname(p), "._" + os.path.basename(p))
            open(side, "wb").close()
        open(os.path.join(root, "._TABLES"), "wb").close()
        _drop_appledouble(root, written)
        left = sorted(f for _, _, fs in os.walk(root) for f in fs
                      if f.startswith("._"))
        check("every sidecar of a written file or directory is gone",
              left == ["._KEEP.TXT"], str(left))
        check("the files themselves are untouched",
              all(os.path.exists(p) for p in written))

    print("\n[4] names match the way they are typed")
    vols = [{"Id": "NO NAME"}, {"Id": "PEREGRINE"}] if not IS_WINDOWS \
        else [{"Id": "E"}, {"Id": "F"}]
    if IS_WINDOWS:
        check("a letter matches with or without the colon",
              find(vols, "e:") is vols[0] and find(vols, "F") is vols[1])
    else:
        check("a name matches in any case", find(vols, "no name") is vols[0])
        check("a pasted /Volumes path matches",
              find(vols, "/Volumes/PEREGRINE/") is vols[1])
        check("a name is not matched by a prefix", find(vols, "NO") is None)
    check("blank matches nothing", find(vols, "  ") is None)
    try:
        wipe(vols[0]["Id"], vols[1]["Id"])
        check("a mismatched confirmation is refused before anything else", False)
    except ValueError:
        check("a mismatched confirmation is refused before anything else", True)

    print(f"\n{npass} passed, {nfail} failed\n")
    return 1 if nfail else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    # Read-only listing, safe to run any time.
    ws = sys.argv[1] if len(sys.argv) > 1 else None
    for v in list_volumes():
        a = assess(v, ws)
        mark = "OK " if a["safe"] else "NO "
        print(mark, describe(v))
        for r in a["reasons"]:
            print("      -", r)
