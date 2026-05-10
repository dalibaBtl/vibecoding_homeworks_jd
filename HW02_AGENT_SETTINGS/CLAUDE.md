# CLAUDE.md — HW02_AGENT_SETTINGS

Training material: configure a Claude Code agent setup **from scratch** using only the primitives — **MCP servers**, **skills**, **subagents** — and **no plugins or marketplace**. The demo app is a small web-based schematic editor for industrial cabinet symbols (IEC 60617). The app is the *vehicle*; the agent configuration is the *deliverable*.

> Parent `../CLAUDE.md` defines repo-wide conventions (uv, per-homework projects). It still applies.

## What lives where

```
HW02_AGENT_SETTINGS/
├── CLAUDE.md                          # this file — project rules + invariants
├── .mcp.json                          # registers both MCP servers (project-scoped)
├── .claude/
│   ├── settings.json                  # enables MCP servers, allow/deny permissions
│   ├── agents/
│   │   └── symbol-author.md           # subagent: generates IEC SVG symbols
│   └── skills/
│       └── btl101-styling/SKILL.md    # skill: applies BTL101 visual identity
├── symbol_pin_inspector_mcp/          # custom MCP server (Python, uv)
│   ├── pyproject.toml
│   ├── src/symbol_pin_inspector/
│   │   ├── __init__.py
│   │   └── server.py                  # FastMCP server: inspect_symbol + bundle_library
│   └── tests/smoke.py                 # smoke tests for both tools
├── refs/                              # PDF references for the subagent to read
│   └── toaz.info-iec-60617-symbols-...-2.pdf
├── symbols/                           # generated SVGs (subagent writes here)
│   ├── protection/                    # circuit breakers, fuses
│   ├── switching/                     # coils, contacts, switches
│   ├── loads/                         # motors, lamps, heaters
│   └── connection/                    # terminals, earth/ground
├── index.html                         # the editor (single page, no build)
├── app.css                            # styled by the btl101-styling skill
└── app.js                             # drag/drop, wires, save/load
```

## The three agent primitives — quick reference

| Primitive | File | Loaded | Cost in context |
|---|---|---|---|
| **MCP server** | `.mcp.json` + server impl | At session start (subprocess) | Tool list permanently in context |
| **Subagent** | `.claude/agents/<name>.md` | At session start (frontmatter only) | Description in context; body loads when invoked via `Agent` tool |
| **Skill** | `.claude/skills/<name>/SKILL.md` | At session start (frontmatter only) | Description in context; body loads when the model decides to activate it |

CLAUDE.md (this file) is fully loaded **every turn**. Skills are activated **on demand**. That's why BTL101 styling lives in a skill, not here — it only matters when styling.

## MCP servers registered

Both are stdio servers, project-scoped via `.mcp.json`. Settings opt them in with `enableAllProjectMcpServers: true`.

### `symbol-pin-inspector` (custom, Python)

- Built with the `mcp` Python SDK (`FastMCP`).
- Launched via `uv --directory symbol_pin_inspector_mcp run symbol-pin-inspector`.
- Two tools:
  - `mcp__symbol-pin-inspector__inspect_symbol(svg_path)` → `{pins, errors, grid, ok}` — validates a single SVG. Used by the `symbol-author` subagent after every write.
  - `mcp__symbol-pin-inspector__bundle_library(symbols_dir?, output_path?, soft_validate?)` → `{ok, output, size_bytes, categories, included, skipped, errors}` — walks `symbols/<category>/*.svg`, soft-validates each, derives a label from `<title>` or filename, and writes `symbols/bundle.js` (the editor's `window.SYMBOLS` blob). Replaces the old hand-maintained `manifest.json` + `tools/bundle.py` pair — the discovery walk is the source of truth.

### `btl101-fs` (off-the-shelf, npx)

- Official `@modelcontextprotocol/server-filesystem`, run via `npx -y …`.
- Pinned to `/home/jandaliba/dev/BTL101/` (read-only — write tools are denied in `.claude/settings.json`).
- Exposes `read_text_file`, `directory_tree`, `search_files`, etc. — used by the `btl101-styling` skill, **not** by the symbol-author subagent.

## Symbol contract (one line)

Symbols live in `symbols/<category>/<name>.svg`. The `inspect_symbol` MCP tool is the authoritative check for the geometry contract (10 px grid, pin circles, unique names). The full invariants and authoring conventions live in `.claude/agents/symbol-author.md` and load only when the subagent runs — do not hand-edit symbols, re-run the subagent.

## Editor convention

- Single `index.html` + `app.css` + `app.js` + generated `symbols/bundle.js`. **No build step, no server.**
- Editor opens via `file://` directly. Symbol data is inlined into `bundle.js` as `window.SYMBOLS = {manifest, files}` — the editor never calls `fetch()`, which would be blocked over `file://`.
- After adding/removing/renaming a symbol, regenerate the bundle by calling `mcp__symbol-pin-inspector__bundle_library` (no args needed — it walks `symbols/`).
- Canvas snaps drops to the **10 px grid**; wire endpoints clamp the same way.
- Schematics serialize to JSON: `{symbols: [{id, src, x, y}], wires: [{id, from: [symId,pin], to: [symId,pin]}]}`. Round-trip (export → import → export) is diff-stable: IDs are preserved on import and re-emitted on export.

## How to verify the setup (after first restart of Claude Code in this folder)

1. `claude mcp list` → should show **two** servers, both healthy.
2. In a Claude Code session: type `/agents` (or `/help` if your build differs) and confirm `symbol-author` appears.
3. Skill discoverability: ask Claude "what skills are available here?" — `btl101-styling` should be listed.
4. Smoke test the validator standalone:
   ```bash
   cd symbol_pin_inspector_mcp
   uv --cache-dir .uv-cache run python tests/smoke.py
   ```
   Expect 4 PASS lines.

## Hard rules in this folder

- **Never** modify anything under `/home/jandaliba/dev/BTL101/`. It's reference-only via the `btl101-fs` MCP. The MCP's write tools are denied in `.claude/settings.json`; respect that even if a tool slips through.
- **Never** create symbols outside `symbols/<category>/`. The subagent is constrained; manual edits should respect the same boundary.
- **Never** commit `node_modules/`, `.venv/`, `.uv-cache/`, or `uv.lock` outside `symbol_pin_inspector_mcp/`.
- Symbol SVGs are generated artefacts produced by the `symbol-author` subagent. Don't hand-edit them — re-run the subagent with refined instructions instead.

## Build status (handoff)

Done:
- Custom MCP server (`symbol-pin-inspector`) — tested, all 4 smoke cases pass.
- Off-the-shelf MCP (`btl101-fs`) — verified handshake + tools/list.
- `.mcp.json`, `.claude/settings.json`, `symbol-author.md`, `btl101-styling/SKILL.md` written.

Not yet done (requires a fresh Claude Code session to pick up the new config):
- Generate the six IEC symbols (MCB, contactor coil, NO contact, 3-phase motor, terminal, PE) via the `symbol-author` subagent.
- Build the editor (`index.html`, `app.css`, `app.js`).
- Activate the `btl101-styling` skill to apply BTL101 chrome.

Next session's first move: confirm both MCP servers loaded (`claude mcp list`), then ask the `symbol-author` subagent to create the first symbol (e.g. the motor) so the full chain (subagent → MCP validator → SVG file on disk) is exercised end-to-end.
