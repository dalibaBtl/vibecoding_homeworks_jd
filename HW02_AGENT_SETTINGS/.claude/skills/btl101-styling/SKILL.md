---
name: btl101-styling
description: Apply BTL101 visual identity (palette, typography, spacing, chrome) to the HW02 schematic editor's HTML and CSS. Activate when the user asks to "style like BTL101", "apply BTL101 design", "make this look like Bitlago", or anything else that wants the editor's chrome to match the BTL101 web site. Reads BTL101 source read-only via the `btl101-fs` MCP and rewrites the editor's stylesheet using CSS custom properties.
---

# BTL101 styling skill

You are styling the HW02 schematic editor (`index.html`, `app.css`, `app.js` in the project root) to match the BTL101 visual identity. All BTL101 source is **read-only** — every read goes through the `btl101-fs` MCP, and the corresponding write tools are denied in `.claude/settings.json`. You never modify a file under `/home/jandaliba/dev/BTL101/`.

## Step 1 — Survey BTL101 source

Use the `btl101-fs` MCP to discover what's there. **Do not use the built-in `Read` / `Glob`** for BTL101 files — the whole point of this skill is to demonstrate MCP access to an out-of-tree project.

1. `mcp__btl101-fs__list_allowed_directories` — confirm the BTL101 root.
2. `mcp__btl101-fs__directory_tree` with `excludePatterns: [".git", "node_modules", "*.pdf"]` — find the web entry point (`_WEB/index.html`) and any CSS files.
3. `mcp__btl101-fs__search_files` with `pattern: "**/*.css"` — locate stylesheets.
4. `mcp__btl101-fs__read_text_file` on `_WEB/index.html` and each `.css` you found.
5. If the page references custom fonts or CSS variables, follow those references too.

## Step 2 — Extract design tokens

Capture the following from what BTL101 actually declares — do not invent values:

| Token | Where to find it |
|---|---|
| `--btl-bg` | Body / page background color |
| `--btl-panel` | Card / sidebar / panel background |
| `--btl-fg` | Primary text color |
| `--btl-muted` | Secondary text / borders |
| `--btl-accent` | Headings, links, brand-tinted UI |
| `--btl-font` | `font-family` for body text |
| `--btl-font-heading` | `font-family` for h1/h2 (if different) |
| `--btl-radius` | Border radius on cards / buttons |
| `--btl-space` | Base spacing unit (likely 8 / 12 / 16 px) |

Note in your reasoning which token came from which file/line of BTL101. If a token has no obvious source, leave it derived from the closest neighbor and call it out in the report.

## Step 3 — Apply to the editor

Update only files **inside HW02_AGENT_SETTINGS**:

1. `app.css` — declare all tokens on `:root` and consume them from the existing rules. Replace every hard-coded color/font/spacing in the file with a `var(--btl-…)` reference. Add hover/active states using `--btl-accent`.
2. `index.html` — wrap the existing layout in a header / main / sidebar structure if needed for the BTL101 chrome to apply cleanly. **Do not** rename DOM ids or classes that `app.js` depends on.
3. Canvas grid — render the grid as a light dotted pattern in `--btl-muted` so symbols stand out.

## Step 4 — Verify

- The editor still functions: drag a symbol from the palette, click two pins, save/load JSON. Open `index.html` in a browser if practical; otherwise re-read `app.js` and confirm the DOM hooks it depends on still exist.
- No BTL101 file was modified — the `mcp__btl101-fs__*` write tools are denied in settings, so any accidental write attempt would have failed. State this verification in the report.
- Report a one-paragraph diff summary: tokens extracted (with their BTL101 source), HW02 files touched, anything you couldn't extract cleanly.

## Out of scope

- Pixel-perfect parity with BTL101.
- Copying logos / images from BTL101 into HW02. Reference them by absolute path only if explicitly requested; otherwise omit.
- Marketing copy, hero sections, or animations from BTL101's homepage — this editor's chrome is functional, not promotional.
- Modifying `app.js` logic. Styling only.
