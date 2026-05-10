"""MCP server: validate SVG schematic symbols for the HW02 editor.

The schematic editor expects every symbol SVG to follow a small, strict
convention so that any two symbols can connect cleanly on a shared grid:

  * A ``viewBox`` whose width and height are multiples of ``GRID`` (10 px).
  * One or more elements carrying ``class="pin"`` (typically ``<circle>``)
    with ``data-pin="<name>"`` and numeric ``cx`` / ``cy`` attributes.
  * Pin names unique within the file.
  * Pin coordinates aligned to the ``GRID``.
  * No two pins sharing the same coordinate.

This file is the *single* MCP tool implementation. Keep the validation
deterministic: the ``symbol-author`` subagent invokes ``inspect_symbol``
after writing a candidate SVG and only commits it when ``ok`` is ``True``.

Transport is stdio — the server is launched by the Claude Code MCP client
via ``uv run symbol-pin-inspector`` (see ``../.mcp.json``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from mcp.server.fastmcp import FastMCP

GRID: int = 10
"""Grid spacing in SVG user units. Every pin coordinate must be a multiple of this."""

mcp = FastMCP("symbol-pin-inspector")


@dataclass(frozen=True, slots=True)
class Pin:
    """A connection point parsed from a symbol SVG."""

    name: str
    cx: float
    cy: float


def _local_name(tag: str) -> str:
    """Return the element local name, stripping any XML namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def _parse_pins(root: ET.Element) -> tuple[list[Pin], list[str]]:
    """Extract pins from the SVG tree.

    A pin is any element whose ``class`` attribute contains the token ``pin``
    and which carries ``data-pin``, ``cx``, ``cy``. Malformed candidates are
    skipped and reported via the returned error list rather than raising.
    """
    pins: list[Pin] = []
    errs: list[str] = []
    for el in root.iter():
        classes = (el.get("class") or "").split()
        if "pin" not in classes:
            continue
        name = el.get("data-pin")
        cx_s = el.get("cx")
        cy_s = el.get("cy")
        if name is None:
            errs.append(f'<{_local_name(el.tag)}> with class="pin" is missing data-pin')
            continue
        if cx_s is None or cy_s is None:
            errs.append(f"pin {name!r} is missing cx or cy")
            continue
        try:
            cx = float(cx_s)
            cy = float(cy_s)
        except ValueError:
            errs.append(f"pin {name!r} has non-numeric cx/cy ({cx_s!r}, {cy_s!r})")
            continue
        pins.append(Pin(name=name, cx=cx, cy=cy))
    return pins, errs


def _check_viewbox(root: ET.Element) -> list[str]:
    """Validate the ``<svg>`` viewBox: 4 numbers, width and height divisible by GRID."""
    errs: list[str] = []
    vb = root.get("viewBox")
    if not vb:
        errs.append("missing viewBox attribute on <svg>")
        return errs
    try:
        parts = [float(x) for x in re.split(r"[\s,]+", vb.strip()) if x]
    except ValueError:
        errs.append(f"viewBox not parseable: {vb!r}")
        return errs
    if len(parts) != 4:
        errs.append(f"viewBox must have 4 numbers, got {len(parts)}: {vb!r}")
        return errs
    _, _, w, h = parts
    if w <= 0 or h <= 0:
        errs.append(f"viewBox width/height must be positive, got {w}x{h}")
    if w % GRID or h % GRID:
        errs.append(f"viewBox dimensions must be multiples of {GRID}, got {w}x{h}")
    return errs


@mcp.tool()
def inspect_symbol(svg_path: str) -> dict:
    """Validate a single SVG schematic symbol file.

    Args:
        svg_path: Absolute or workspace-relative path to the candidate SVG.

    Returns:
        A dict with keys:
          - ``pins``: list of ``{"name": str, "cx": float, "cy": float}``.
          - ``errors``: list of human-readable error strings (empty on success).
          - ``grid``: the grid size used for the check.
          - ``ok``: ``True`` iff ``errors`` is empty.

    The function never raises — file/parse failures are reported in ``errors``
    so the caller (typically the ``symbol-author`` subagent) can react.
    """
    path = Path(svg_path).expanduser()
    if not path.is_file():
        return {"pins": [], "errors": [f"file not found: {svg_path}"], "grid": GRID, "ok": False}

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {"pins": [], "errors": [f"XML parse error: {exc}"], "grid": GRID, "ok": False}

    errors: list[str] = []
    errors.extend(_check_viewbox(root))

    pins, pin_errs = _parse_pins(root)
    errors.extend(pin_errs)

    if not pins:
        errors.append('no elements with class="pin" found')

    # Name uniqueness.
    name_counts: dict[str, int] = {}
    for p in pins:
        name_counts[p.name] = name_counts.get(p.name, 0) + 1
    for name, count in name_counts.items():
        if count > 1:
            errors.append(f"duplicate pin name {name!r} ({count} occurrences)")

    # Grid alignment.
    for p in pins:
        if math.isnan(p.cx) or math.isnan(p.cy):
            errors.append(f"pin {p.name!r} has NaN coordinates")
            continue
        if p.cx % GRID or p.cy % GRID:
            errors.append(
                f"pin {p.name!r} at ({p.cx}, {p.cy}) is not on the {GRID}px grid"
            )

    # Coordinate uniqueness.
    by_coord: dict[tuple[float, float], list[str]] = {}
    for p in pins:
        by_coord.setdefault((p.cx, p.cy), []).append(p.name)
    for coord, names in by_coord.items():
        if len(names) > 1:
            errors.append(f"pins {names} overlap at {coord}")

    return {
        "pins": [{"name": p.name, "cx": p.cx, "cy": p.cy} for p in pins],
        "errors": errors,
        "grid": GRID,
        "ok": not errors,
    }


def main() -> None:
    """Console-script entry point. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
