"""Smoke tests for the symbol-pin-inspector MCP server.

Covers both tools:

  * ``inspect_symbol`` — happy path plus three failure modes.
  * ``bundle_library`` — full discovery on a synthesized ``symbols/`` tree:
    valid SVGs are bundled, invalid ones are reported under ``skipped``,
    and labels follow the title/filename heuristic.

Run from the project root::

    uv --cache-dir .uv-cache run python tests/smoke.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from symbol_pin_inspector.server import bundle_library, inspect_symbol


# ---------------------------------------------------------------------------
# inspect_symbol fixtures
# ---------------------------------------------------------------------------

GOOD = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">
  <rect x="10" y="5" width="20" height="10" fill="none" stroke="black"/>
  <circle class="pin" data-pin="A" cx="0"  cy="10" r="2"/>
  <circle class="pin" data-pin="B" cx="40" cy="10" r="2"/>
</svg>
"""

BAD_OFFGRID = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">
  <circle class="pin" data-pin="A" cx="3"  cy="10" r="2"/>
  <circle class="pin" data-pin="B" cx="40" cy="10" r="2"/>
</svg>
"""

BAD_DUPLICATE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">
  <circle class="pin" data-pin="A" cx="0"  cy="10" r="2"/>
  <circle class="pin" data-pin="A" cx="40" cy="10" r="2"/>
</svg>
"""

BAD_VIEWBOX = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 33 20">
  <circle class="pin" data-pin="A" cx="0"  cy="10" r="2"/>
  <circle class="pin" data-pin="B" cx="30" cy="10" r="2"/>
</svg>
"""

# ---------------------------------------------------------------------------
# bundle_library fixtures
# ---------------------------------------------------------------------------

# A valid SVG with no <title> — label should fall back to humanized filename.
BUNDLE_GOOD_NO_TITLE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">
  <circle class="pin" data-pin="1" cx="0"  cy="10" r="2"/>
  <circle class="pin" data-pin="2" cx="40" cy="10" r="2"/>
</svg>
"""

# A valid SVG with an explicit <title> — label should come from <title>.
BUNDLE_GOOD_WITH_TITLE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">
  <title>Custom Coil Label</title>
  <circle class="pin" data-pin="A1" cx="0"  cy="10" r="2"/>
  <circle class="pin" data-pin="A2" cx="40" cy="10" r="2"/>
</svg>
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_inspect(svg: str) -> dict:
    """Write ``svg`` to a temp file and return ``inspect_symbol``'s output."""
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as fh:
        fh.write(svg)
        path = fh.name
    try:
        return inspect_symbol(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _build_workspace(root: Path) -> None:
    """Lay out a minimal ``symbols/`` tree for ``bundle_library`` testing.

    ``symbols/protection/mcb-1p.svg`` — valid, no title (auto-label "MCB 1P")
    ``symbols/switching/coil.svg``    — valid, with <title> override
    ``symbols/loads/broken.svg``      — invalid (off-grid pin) → skipped
    """
    (root / "symbols" / "protection").mkdir(parents=True)
    (root / "symbols" / "switching").mkdir(parents=True)
    (root / "symbols" / "loads").mkdir(parents=True)
    (root / "symbols" / "connection").mkdir(parents=True)

    (root / "symbols" / "protection" / "mcb-1p.svg").write_text(BUNDLE_GOOD_NO_TITLE)
    (root / "symbols" / "switching" / "coil.svg").write_text(BUNDLE_GOOD_WITH_TITLE)
    (root / "symbols" / "loads" / "broken.svg").write_text(BAD_OFFGRID)


def _expect(condition: bool, label: str, detail: str = "") -> int:
    """Print PASS/FAIL line and return 0 (pass) or 1 (fail)."""
    verdict = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{verdict}] {label}{suffix}")
    return 0 if condition else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    failures = 0

    # --- inspect_symbol ----------------------------------------------------
    inspect_cases: list[tuple[str, str, bool]] = [
        ("inspect.good", GOOD, True),
        ("inspect.bad_offgrid", BAD_OFFGRID, False),
        ("inspect.bad_duplicate", BAD_DUPLICATE, False),
        ("inspect.bad_viewbox", BAD_VIEWBOX, False),
    ]
    for name, svg, expect_ok in inspect_cases:
        got = _run_inspect(svg)
        failures += _expect(
            bool(got["ok"]) is expect_ok,
            name,
            f"ok={got['ok']} errors={got['errors']}",
        )

    # --- bundle_library ----------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        _build_workspace(ws)
        result = bundle_library(symbols_dir=str(ws / "symbols"))

        failures += _expect(result["ok"] is True, "bundle.ok", str(result.get("errors")))
        failures += _expect(
            len(result["included"]) == 2,
            "bundle.included_count",
            f"included={result['included']}",
        )
        failures += _expect(
            len(result["skipped"]) == 1
            and result["skipped"][0]["path"].endswith("broken.svg"),
            "bundle.skipped_invalid",
            f"skipped={result['skipped']}",
        )

        out_path = Path(result["output"])
        failures += _expect(out_path.is_file(), "bundle.file_written", str(out_path))

        text = out_path.read_text(encoding="utf-8")
        # Strip header comment lines and the leading "window.SYMBOLS = " to
        # parse the JSON payload.
        json_start = text.index("{")
        json_end = text.rindex("}")
        payload = json.loads(text[json_start : json_end + 1])

        labels = {
            s["file"]: s["label"]
            for cat in payload["manifest"]["categories"]
            for s in cat["symbols"]
        }
        failures += _expect(
            labels.get("mcb-1p.svg") == "MCB 1P",
            "bundle.label_humanized",
            f"got={labels.get('mcb-1p.svg')!r}",
        )
        failures += _expect(
            labels.get("coil.svg") == "Custom Coil Label",
            "bundle.label_from_title",
            f"got={labels.get('coil.svg')!r}",
        )
        failures += _expect(
            "symbols/protection/mcb-1p.svg" in payload["files"]
            and "symbols/switching/coil.svg" in payload["files"],
            "bundle.files_inlined",
        )

    print()
    print("All smoke tests passed." if failures == 0 else f"{failures} failure(s).")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
