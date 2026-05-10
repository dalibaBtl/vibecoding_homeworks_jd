"""Symbol Pin Inspector — MCP server validating SVG schematic symbols.

Exposes a single MCP tool, ``inspect_symbol``, used by the ``symbol-author``
subagent to self-verify generated SVG files before saving them.
"""

__all__ = ["server"]
