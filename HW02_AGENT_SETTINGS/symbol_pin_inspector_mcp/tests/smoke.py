"""Smoke test for ``inspect_symbol``.

Constructs four in-memory SVGs covering the success path and the three
main failure modes (off-grid pin, duplicate pin name, non-grid viewBox),
runs the validator on each, and asserts the ``ok`` flag matches.

Run from the project root::

    uv --cache-dir .uv-cache run python tests/smoke.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from symbol_pin_inspector.server import inspect_symbol


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


def _run(svg: str) -> dict:
    """Write ``svg`` to a temp file and return ``inspect_symbol``'s output."""
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as fh:
        fh.write(svg)
        path = fh.name
    try:
        return inspect_symbol(path)
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    """Run each case and report. Exit status is the number of failures."""
    cases: list[tuple[str, str, bool]] = [
        ("good", GOOD, True),
        ("bad_offgrid", BAD_OFFGRID, False),
        ("bad_duplicate", BAD_DUPLICATE, False),
        ("bad_viewbox", BAD_VIEWBOX, False),
    ]
    failures = 0
    for label, svg, expect_ok in cases:
        result = _run(svg)
        got_ok = bool(result["ok"])
        verdict = "PASS" if got_ok is expect_ok else "FAIL"
        if got_ok is not expect_ok:
            failures += 1
        print(f"[{verdict}] {label}: ok={got_ok} errors={result['errors']}")
    print()
    print("All smoke tests passed." if failures == 0 else f"{failures} failure(s).")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
