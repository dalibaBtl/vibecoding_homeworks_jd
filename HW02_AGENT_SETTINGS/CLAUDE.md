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
│   │   └── server.py                  # FastMCP server, one tool: inspect_symbol
│   └── tests/smoke.py                 # four-case validator smoke test
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
- One tool: `mcp__symbol-pin-inspector__inspect_symbol(svg_path)` → `{pins, errors, grid, ok}`.

### `btl101-fs` (off-the-shelf, npx)

- Official `@modelcontextprotocol/server-filesystem`, run via `npx -y …`.
- Pinned to `/home/jandaliba/dev/BTL101/` (read-only — write tools are denied in `.claude/settings.json`).
- Exposes `read_text_file`, `directory_tree`, `search_files`, etc. — used by the `btl101-styling` skill, **not** by the symbol-author subagent.

## Symbol convention (invariants)

Every SVG under `symbols/` must:

1. Have a `viewBox` whose width and height are multiples of **10** (the grid).
2. Use `<circle class="pin" data-pin="<name>" cx="<n>" cy="<n>" r="2"/>` for every connection point.
3. Place every pin's `cx` / `cy` on a multiple of 10.
4. Give each pin a name unique within the file.
5. Use industry-standard pin names where one exists (U1/V1/W1, A1/A2, 13/14, 1/3/5–2/4/6, PE).
6. Stroke black, 1.5 px, fill none — unless the IEC drawing requires a fill (e.g. earth bar).

The `inspect_symbol` MCP tool is the authoritative check for points 1–4. The subagent calls it after every write and retries up to 3 times.

## Editor convention

- Single `index.html` + `app.css` + `app.js`. **No build step.**
- Symbols are loaded by relative URL from `symbols/`.
- Canvas snaps drops to the **10 px grid**; wire endpoints clamp the same way.
- Schematics serialize to JSON: `{symbols: [{id, src, x, y}], wires: [{from: [symId,pin], to: [symId,pin]}]}`.

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
