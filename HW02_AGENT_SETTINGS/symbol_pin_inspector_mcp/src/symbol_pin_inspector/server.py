"""MCP server: validate and bundle SVG schematic symbols for the HW02 editor.

Two tools live here:

  * ``inspect_symbol`` — validates a single candidate SVG against the symbol
    contract used by the editor. The ``symbol-author`` subagent calls it
    after every write and only commits when ``ok`` is ``True``.

  * ``bundle_library`` — walks the ``symbols/`` tree, validates every file,
    and writes ``bundle.js`` (a ``window.SYMBOLS = {...}`` blob) so the
    editor can run from ``file://`` with no HTTP fetches.

The contract every symbol must satisfy:

  * A ``viewBox`` whose width and height are multiples of ``GRID`` (10 px).
  * One or more elements carrying ``class="pin"`` (typically ``<circle>``)
    with ``data-pin="<name>"`` and numeric ``cx`` / ``cy`` attributes.
  * Pin names unique within the file.
  * Pin coordinates aligned to the ``GRID``.
  * No two pins sharing the same coordinate.

Transport is stdio — the server is launched by the Claude Code MCP client
via ``uv run symbol-pin-inspector`` (see ``../.mcp.json``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from mcp.server.fastmcp import FastMCP

GRID: int = 10
"""Grid spacing in SVG user units. Every pin coordinate must be a multiple of this."""

SVG_NS: str = "http://www.w3.org/2000/svg"
"""SVG namespace URI, used to address ``<title>`` etc. in parsed documents."""

CATEGORIES: tuple[tuple[str, str], ...] = (
    ("protection", "Protection"),
    ("switching", "Switching"),
    ("loads", "Loads"),
    ("connection", "Connection"),
)
"""Library category folders in display order. Each entry is ``(folder, label)``.

Adding a new category means adding a row here and creating the matching
folder under ``symbols/``. The editor renders categories in this order.
"""

_LABEL_ABBREVS: frozenset[str] = frozenset({"mcb", "no", "nc", "pe", "ac", "dc", "vfd", "led"})
"""Lowercase tokens that should render as uppercase in auto-generated labels."""

_PROJECT_ROOT_DEFAULT: Path = Path(__file__).resolve().parents[3]
"""Fallback project root: ``symbol_pin_inspector_mcp/`` parent directory.

Used when ``bundle_library`` is invoked without an explicit ``symbols_dir``.
"""

mcp = FastMCP("symbol-pin-inspector")


@dataclass(frozen=True, slots=True)
class Pin:
    """A connection point parsed from a symbol SVG."""

    name: str
    cx: float
    cy: float


# ---------------------------------------------------------------------------
# Parsing helpers (shared between inspect_symbol and bundle_library)
# ---------------------------------------------------------------------------

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


def _validate_root(root: ET.Element) -> tuple[list[Pin], list[str]]:
    """Run the full symbol contract against an already-parsed SVG root.

    Returns the parsed pins (possibly empty) and a list of error strings.
    Used by both ``inspect_symbol`` (single file) and ``bundle_library``
    (per file during a tree walk) so both stay in lockstep.
    """
    errors: list[str] = []
    errors.extend(_check_viewbox(root))

    pins, pin_errs = _parse_pins(root)
    errors.extend(pin_errs)

    if not pins:
        errors.append('no elements with class="pin" found')

    name_counts: dict[str, int] = {}
    for p in pins:
        name_counts[p.name] = name_counts.get(p.name, 0) + 1
    for name, count in name_counts.items():
        if count > 1:
            errors.append(f"duplicate pin name {name!r} ({count} occurrences)")

    for p in pins:
        if math.isnan(p.cx) or math.isnan(p.cy):
            errors.append(f"pin {p.name!r} has NaN coordinates")
            continue
        if p.cx % GRID or p.cy % GRID:
            errors.append(
                f"pin {p.name!r} at ({p.cx}, {p.cy}) is not on the {GRID}px grid"
            )

    by_coord: dict[tuple[float, float], list[str]] = {}
    for p in pins:
        by_coord.setdefault((p.cx, p.cy), []).append(p.name)
    for coord, names in by_coord.items():
        if len(names) > 1:
            errors.append(f"pins {names} overlap at {coord}")

    return pins, errors


# ---------------------------------------------------------------------------
# Tool 1: inspect_symbol
# ---------------------------------------------------------------------------

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

    pins, errors = _validate_root(root)
    return {
        "pins": [{"name": p.name, "cx": p.cx, "cy": p.cy} for p in pins],
        "errors": errors,
        "grid": GRID,
        "ok": not errors,
    }


# ---------------------------------------------------------------------------
# Tool 2: bundle_library
# ---------------------------------------------------------------------------

def _humanize(stem: str) -> str:
    """Turn a filename stem into a human-readable label.

    Splits on dashes/underscores. Uppercases short known abbreviations
    (``mcb``, ``no``, ``pe``, …) and mixed alphanumeric tokens (``1p``,
    ``3ph``). Capitalizes ordinary words. Preserves digit-only tokens.

    Examples:
      ``mcb-1p`` → ``"MCB 1P"``
      ``contact-no`` → ``"Contact NO"``
      ``motor-3ph`` → ``"Motor 3PH"``
      ``pushbutton`` → ``"Pushbutton"``
    """
    out: list[str] = []
    for tok in re.split(r"[-_]+", stem):
        if not tok:
            continue
        low = tok.lower()
        if low in _LABEL_ABBREVS:
            out.append(low.upper())
        elif re.fullmatch(r"\d+[a-z]+|[a-z]+\d+", low):
            out.append(low.upper())
        elif re.fullmatch(r"\d+", low):
            out.append(low)
        else:
            out.append(low.capitalize())
    return " ".join(out) or stem


def _read_label(path: Path, root: ET.Element) -> str:
    """Return the symbol's display label.

    Prefers the SVG's first ``<title>`` element when present; otherwise
    derives a label from the filename stem via :func:`_humanize`.
    Authors who want a non-default label should add ``<title>...</title>``
    as the first child of their ``<svg>``.
    """
    title = root.findtext(f"{{{SVG_NS}}}title")
    if title and title.strip():
        return title.strip()
    return _humanize(path.stem)


@mcp.tool()
def bundle_library(
    symbols_dir: str | None = None,
    output_path: str | None = None,
    soft_validate: bool = True,
) -> dict:
    """Discover every symbol on disk and write the editor's ``bundle.js``.

    Walks each known category folder under ``symbols_dir`` (in the order
    declared by :data:`CATEGORIES`), parses every ``*.svg``, and inlines
    its text plus a derived display label into a single JavaScript file
    that sets ``window.SYMBOLS = {manifest, files}``.

    The editor opens via ``file://`` and cannot ``fetch()`` cross-origin,
    so all symbol content must be available synchronously at load time —
    that is what this bundle provides.

    Args:
        symbols_dir: Path to the project's ``symbols/`` directory. When
            omitted, falls back to ``<project root>/symbols`` derived from
            the server's install location.
        output_path: Where to write the bundle. Defaults to
            ``<symbols_dir>/bundle.js``.
        soft_validate: When ``True`` (default), invalid SVGs are skipped
            (not bundled) and reported under ``skipped``. When ``False``,
            any validation failure aborts the bundle with ``ok=False``.

    Returns:
        A dict with keys:
          - ``ok``: ``True`` iff a bundle was written successfully.
          - ``output``: absolute path to the written ``bundle.js`` (or
            ``None`` on failure).
          - ``size_bytes``: bundle size on disk (0 on failure).
          - ``categories``: list of ``{name, folder, symbol_count}`` summaries.
          - ``included``: project-relative paths bundled, in walk order.
          - ``skipped``: list of ``{path, errors}`` entries for files that
            failed validation (non-empty only when ``soft_validate`` is
            ``True``).
          - ``errors``: top-level errors (e.g. missing ``symbols/`` dir).
    """
    sym_dir = (
        Path(symbols_dir).expanduser().resolve()
        if symbols_dir
        else (_PROJECT_ROOT_DEFAULT / "symbols").resolve()
    )
    if not sym_dir.is_dir():
        return {
            "ok": False,
            "output": None,
            "size_bytes": 0,
            "categories": [],
            "included": [],
            "skipped": [],
            "errors": [f"symbols dir not found: {sym_dir}"],
        }

    out = (
        Path(output_path).expanduser().resolve()
        if output_path
        else sym_dir / "bundle.js"
    )
    project_root = sym_dir.parent

    categories_out: list[dict] = []
    files_out: dict[str, str] = {}
    skipped: list[dict] = []
    included: list[str] = []
    fatal_errors: list[str] = []

    for folder, display in CATEGORIES:
        cat_dir = sym_dir / folder
        cat_entry: dict = {"name": display, "folder": folder, "symbols": []}
        if cat_dir.is_dir():
            for svg_path in sorted(cat_dir.glob("*.svg")):
                rel = svg_path.relative_to(project_root).as_posix()
                try:
                    root = ET.parse(svg_path).getroot()
                except ET.ParseError as exc:
                    skipped.append({"path": rel, "errors": [f"XML parse error: {exc}"]})
                    continue
                _, errs = _validate_root(root)
                if errs:
                    if soft_validate:
                        skipped.append({"path": rel, "errors": errs})
                        continue
                    fatal_errors.append(f"{rel}: {'; '.join(errs)}")
                    continue
                label = _read_label(svg_path, root)
                cat_entry["symbols"].append({"file": svg_path.name, "label": label})
                files_out[rel] = svg_path.read_text(encoding="utf-8")
                included.append(rel)
        categories_out.append(cat_entry)

    if fatal_errors:
        return {
            "ok": False,
            "output": None,
            "size_bytes": 0,
            "categories": [
                {"name": c["name"], "folder": c["folder"], "symbol_count": len(c["symbols"])}
                for c in categories_out
            ],
            "included": included,
            "skipped": skipped,
            "errors": fatal_errors,
        }

    bundle = {"manifest": {"categories": categories_out}, "files": files_out}
    payload = json.dumps(bundle, indent=2, ensure_ascii=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "// Auto-generated by mcp__symbol-pin-inspector__bundle_library — do not hand-edit.\n"
        "// Regenerate after adding, removing, or renaming a symbol.\n"
        f"window.SYMBOLS = {payload};\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "output": str(out),
        "size_bytes": out.stat().st_size,
        "categories": [
            {"name": c["name"], "folder": c["folder"], "symbol_count": len(c["symbols"])}
            for c in categories_out
        ],
        "included": included,
        "skipped": skipped,
        "errors": [],
    }


def main() -> None:
    """Console-script entry point. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
