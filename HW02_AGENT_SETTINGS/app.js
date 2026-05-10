/*
 * Schematic editor — vanilla ES2020, no build step.
 *
 * Responsibilities:
 *   - Fetch and render the library from symbols/manifest.json.
 *   - Cache parsed symbol SVGs (viewBox dimensions + pin coordinates +
 *     inner markup), keyed by src.
 *   - Place symbols on the canvas via drag-and-drop, snapped to a 10 px grid.
 *   - Move placed symbols by dragging their body (not their pins).
 *   - Create orthogonal (H-V-H) wires by clicking pin A then pin B.
 *   - Select-and-Delete for symbols and wires; Esc cancels in-progress wire.
 *   - Export the schematic as JSON matching the schema in CLAUDE.md.
 *
 * The editor uses a single SVG element as the canvas. Each placed symbol
 * becomes a <g class="placed" transform="translate(x,y)"> whose children are
 * the original symbol's drawing primitives (so .pin circles remain clickable).
 * Wires live in a separate <g> beneath the symbol layer.
 *
 * Coordinate space: canvas viewBox is 1600x1000 user units, rendered at 1:1
 * pixels. screen-to-canvas conversion goes through getScreenCTM().inverse()
 * so it stays correct even if the user zooms the browser.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const GRID = 10;
const DRAG_MIME = "application/x-symbol-src";

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------

/** @type {Map<string, {w:number,h:number,pins:Array<{name:string,cx:number,cy:number}>,template:DocumentFragment}>} */
const symbolCache = new Map();

const state = {
  /** @type {Map<string, {src:string,x:number,y:number,w:number,h:number,pinsByName:Map<string,{cx:number,cy:number}>}>} */
  symbols: new Map(),
  /** @type {Array<{id:string, from:[string,string], to:[string,string]}>} */
  wires: [],

  selection: null,            // {kind:'symbol',id} | {kind:'wire',id} | null
  wireStart: null,            // {symId, pin, x, y} or null
  drag: null,                 // {symId, dx, dy} during move, else null

  nextSymId: 1,
  nextWireId: 1,
};

// --------------------------------------------------------------------------
// DOM refs
// --------------------------------------------------------------------------

const canvas = document.getElementById("canvas");
const layerSymbols = document.getElementById("layer-symbols");
const layerWires = document.getElementById("layer-wires");
const layerOverlay = document.getElementById("layer-overlay");
const libraryEl = document.getElementById("library");
const statusEl = document.getElementById("status");

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

const snap = (n) => Math.round(n / GRID) * GRID;

/** Convert client (mouse) coordinates to canvas user-space coordinates. */
function clientToCanvas(clientX, clientY) {
  const pt = canvas.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const ctm = canvas.getScreenCTM();
  if (!ctm) return { x: clientX, y: clientY };
  const local = pt.matrixTransform(ctm.inverse());
  return { x: local.x, y: local.y };
}

function setStatus(text) {
  statusEl.textContent = text;
}

/** Build an orthogonal H-V-H path between two points (snapped midpoint). */
function orthogonalPath(x1, y1, x2, y2) {
  if (x1 === x2 || y1 === y2) {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
  const midX = snap((x1 + x2) / 2);
  return `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`;
}

/** Absolute pin position in canvas coordinates. */
function pinAbs(symId, pinName) {
  const sym = state.symbols.get(symId);
  if (!sym) return null;
  const p = sym.pinsByName.get(pinName);
  if (!p) return null;
  return { x: sym.x + p.cx, y: sym.y + p.cy };
}

// --------------------------------------------------------------------------
// Symbol loader
// --------------------------------------------------------------------------

/**
 * Parse a symbol from the inlined bundle (window.SYMBOLS.files) and return
 * its dimensions, pin list, and a DocumentFragment template of its drawing
 * primitives. Cached by src.
 *
 * Synchronous on purpose: there is no I/O — the bundle is a single <script>
 * loaded before app.js, so all symbol text is in memory at startup.
 */
function loadSymbol(src) {
  if (symbolCache.has(src)) return symbolCache.get(src);

  const text = window.SYMBOLS && window.SYMBOLS.files && window.SYMBOLS.files[src];
  if (!text) throw new Error(`Symbol not in bundle: ${src} (regenerate symbols/bundle.js)`);

  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  const svg = doc.documentElement;
  if (svg.tagName.toLowerCase() !== "svg") {
    throw new Error(`Not an SVG: ${src}`);
  }

  const vb = (svg.getAttribute("viewBox") || "0 0 0 0").split(/\s+/).map(Number);
  const w = vb[2] || +svg.getAttribute("width") || 0;
  const h = vb[3] || +svg.getAttribute("height") || 0;

  const pins = [...svg.querySelectorAll("circle.pin")].map((c) => ({
    name: c.getAttribute("data-pin"),
    cx: +c.getAttribute("cx"),
    cy: +c.getAttribute("cy"),
  }));

  // DocumentFragment template of the symbol's children, ready to clone into
  // each canvas placement. importNode adopts cross-document SVG nodes safely.
  const template = document.createDocumentFragment();
  for (const child of [...svg.children]) {
    template.appendChild(document.importNode(child, true));
  }

  const def = { w, h, pins, template, text };
  symbolCache.set(src, def);
  return def;
}

/** Build a data: URI for the SVG text so library thumbs render with no HTTP/file lookup. */
function symbolDataUri(text) {
  return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
}

// --------------------------------------------------------------------------
// Library
// --------------------------------------------------------------------------

function buildLibrary() {
  const bundle = window.SYMBOLS;
  if (!bundle || !bundle.manifest) {
    setStatus("symbols/bundle.js not loaded — run python3 tools/bundle.py");
    console.error("window.SYMBOLS is missing. Did symbols/bundle.js fail to load?");
    return;
  }

  /** @type {{categories:Array<{name:string,folder:string,symbols:Array<{file:string,label:string}>}>}} */
  const manifest = bundle.manifest;

  for (const cat of manifest.categories) {
    const section = document.createElement("div");
    section.className = "library-section";
    const h = document.createElement("h2");
    h.textContent = cat.name;
    section.appendChild(h);
    libraryEl.appendChild(section);

    for (const s of cat.symbols) {
      const src = `symbols/${cat.folder}/${s.file}`;
      try {
        // Eagerly populate the cache so canvas placement is instant after drop.
        loadSymbol(src);
        section.appendChild(buildLibraryItem(src, s.label));
      } catch (err) {
        console.error(`Failed to load ${src}:`, err);
        section.appendChild(buildBrokenLibraryItem(src, s.label, err));
      }
    }
  }
}

function buildLibraryItem(src, label) {
  const item = document.createElement("div");
  item.className = "library-item";
  item.draggable = true;
  item.dataset.src = src;

  // Render the thumbnail with an <img> whose src is a data: URI built from
  // the cached SVG text — the browser handles SVG rendering natively and
  // there is no file:// or HTTP lookup at all.
  const def = symbolCache.get(src);
  const thumb = document.createElement("div");
  thumb.className = "library-thumb";
  const img = document.createElement("img");
  img.src = symbolDataUri(def.text);
  img.alt = label;
  img.draggable = false;
  img.className = "library-thumb-img";
  thumb.appendChild(img);

  const lab = document.createElement("div");
  lab.className = "library-label";
  lab.textContent = label;

  item.appendChild(thumb);
  item.appendChild(lab);

  item.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData(DRAG_MIME, src);
    e.dataTransfer.effectAllowed = "copy";
  });

  return item;
}

function buildBrokenLibraryItem(src, label, err) {
  const item = document.createElement("div");
  item.className = "library-item library-item-broken";
  item.title = `Failed to load ${src}: ${err.message}`;
  const thumb = document.createElement("div");
  thumb.className = "library-thumb library-thumb-broken";
  thumb.textContent = "?";
  const lab = document.createElement("div");
  lab.className = "library-label";
  lab.textContent = label;
  item.appendChild(thumb);
  item.appendChild(lab);
  return item;
}

// --------------------------------------------------------------------------
// Symbol placement / rendering
// --------------------------------------------------------------------------

/**
 * Place a symbol at an exact top-left coordinate with a caller-supplied id.
 * Used by both drag-drop placement (via :func:`placeSymbol`) and JSON import.
 *
 * Coordinates are snapped to the grid. If the requested id is already in
 * use, the function returns false and does nothing — callers must dedupe.
 */
function instantiateSymbol(id, src, x, y) {
  if (state.symbols.has(id)) return false;
  const def = loadSymbol(src);
  const sx = snap(x);
  const sy = snap(y);

  const pinsByName = new Map(def.pins.map((p) => [p.name, { cx: p.cx, cy: p.cy }]));
  state.symbols.set(id, { src, x: sx, y: sy, w: def.w, h: def.h, pinsByName });

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("class", "placed");
  g.setAttribute("data-sym-id", id);
  g.setAttribute("transform", `translate(${sx}, ${sy})`);

  // Invisible hit-rect first (so it's beneath the drawing primitives and pins
  // — pin clicks reach the pin element directly).
  const hit = document.createElementNS(SVG_NS, "rect");
  hit.setAttribute("class", "placed-hit");
  hit.setAttribute("x", "0");
  hit.setAttribute("y", "0");
  hit.setAttribute("width", String(def.w));
  hit.setAttribute("height", String(def.h));
  g.appendChild(hit);

  // Then the symbol's drawing primitives (pins included), cloned from the
  // cached template so namespaces are preserved.
  g.appendChild(def.template.cloneNode(true));

  // Tag every pin with its parent symbol id for delegated handlers.
  for (const pin of g.querySelectorAll(".pin")) {
    pin.setAttribute("data-sym-id", id);
  }

  layerSymbols.appendChild(g);
  return true;
}

/**
 * Place a symbol centered on a drop coordinate, generating a fresh id.
 * Thin wrapper around :func:`instantiateSymbol`.
 */
function placeSymbol(src, x, y) {
  const def = loadSymbol(src);
  const id = `s${state.nextSymId++}`;
  const sx = x - def.w / 2;
  const sy = y - def.h / 2;
  instantiateSymbol(id, src, sx, sy);
  setStatus(`Placed ${src.split("/").pop()} at (${snap(sx)}, ${snap(sy)})`);
}

function moveSymbol(id, x, y) {
  const sym = state.symbols.get(id);
  if (!sym) return;
  sym.x = x;
  sym.y = y;
  const node = layerSymbols.querySelector(`[data-sym-id="${id}"]`);
  if (node) node.setAttribute("transform", `translate(${x}, ${y})`);
  // Reroute any wires touching this symbol.
  for (const w of state.wires) {
    if (w.from[0] === id || w.to[0] === id) updateWirePath(w);
  }
}

function deleteSymbol(id) {
  // Remove wires first.
  const remainingWires = [];
  for (const w of state.wires) {
    if (w.from[0] === id || w.to[0] === id) {
      const el = layerWires.querySelector(`[data-wire-id="${w.id}"]`);
      if (el) el.remove();
    } else {
      remainingWires.push(w);
    }
  }
  state.wires = remainingWires;

  state.symbols.delete(id);
  const node = layerSymbols.querySelector(`[data-sym-id="${id}"]`);
  if (node) node.remove();
}

// --------------------------------------------------------------------------
// Wires
// --------------------------------------------------------------------------

function startWire(symId, pinName) {
  const p = pinAbs(symId, pinName);
  if (!p) return;
  state.wireStart = { symId, pin: pinName, x: p.x, y: p.y };
  // arm the pin visually
  const node = layerSymbols.querySelector(
    `[data-sym-id="${symId}"] .pin[data-pin="${pinName}"]`,
  );
  if (node) node.classList.add("armed");
  setStatus(`Wire start: ${symId}/${pinName} — click another pin or Esc to cancel`);
  ensurePreview();
}

function finishWire(symId, pinName) {
  if (!state.wireStart) return;
  const start = state.wireStart;
  if (start.symId === symId && start.pin === pinName) {
    cancelWire();
    return;
  }
  const id = `w${state.nextWireId++}`;
  const wire = { id, from: [start.symId, start.pin], to: [symId, pinName] };
  state.wires.push(wire);

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("class", "wire");
  path.setAttribute("data-wire-id", id);
  layerWires.appendChild(path);
  updateWirePath(wire);

  cancelWire();
  setStatus(`Wired ${wire.from.join("/")} → ${wire.to.join("/")}`);
}

function cancelWire() {
  if (state.wireStart) {
    const node = layerSymbols.querySelector(
      `[data-sym-id="${state.wireStart.symId}"] .pin[data-pin="${state.wireStart.pin}"]`,
    );
    if (node) node.classList.remove("armed");
  }
  state.wireStart = null;
  removePreview();
}

function updateWirePath(wire) {
  const a = pinAbs(wire.from[0], wire.from[1]);
  const b = pinAbs(wire.to[0], wire.to[1]);
  if (!a || !b) return;
  const el = layerWires.querySelector(`[data-wire-id="${wire.id}"]`);
  if (el) el.setAttribute("d", orthogonalPath(a.x, a.y, b.x, b.y));
}

function deleteWire(id) {
  state.wires = state.wires.filter((w) => w.id !== id);
  const el = layerWires.querySelector(`[data-wire-id="${id}"]`);
  if (el) el.remove();
}

// preview line that follows the cursor while a wire is in progress

let previewEl = null;
function ensurePreview() {
  if (previewEl) return;
  previewEl = document.createElementNS(SVG_NS, "path");
  previewEl.setAttribute("class", "wire-preview");
  layerOverlay.appendChild(previewEl);
}
function removePreview() {
  if (previewEl) {
    previewEl.remove();
    previewEl = null;
  }
}
function updatePreviewTo(x, y) {
  if (!previewEl || !state.wireStart) return;
  const sx = snap(x);
  const sy = snap(y);
  previewEl.setAttribute(
    "d",
    orthogonalPath(state.wireStart.x, state.wireStart.y, sx, sy),
  );
}

// --------------------------------------------------------------------------
// Selection
// --------------------------------------------------------------------------

function clearSelection() {
  if (!state.selection) return;
  if (state.selection.kind === "symbol") {
    const node = layerSymbols.querySelector(
      `[data-sym-id="${state.selection.id}"]`,
    );
    if (node) node.classList.remove("selected");
  } else if (state.selection.kind === "wire") {
    const node = layerWires.querySelector(
      `[data-wire-id="${state.selection.id}"]`,
    );
    if (node) node.classList.remove("selected");
  }
  state.selection = null;
}

function selectSymbol(id) {
  clearSelection();
  state.selection = { kind: "symbol", id };
  const node = layerSymbols.querySelector(`[data-sym-id="${id}"]`);
  if (node) node.classList.add("selected");
  setStatus(`Selected ${id}`);
}

function selectWire(id) {
  clearSelection();
  state.selection = { kind: "wire", id };
  const node = layerWires.querySelector(`[data-wire-id="${id}"]`);
  if (node) node.classList.add("selected");
  setStatus(`Selected wire ${id}`);
}

function deleteSelection() {
  if (!state.selection) return;
  if (state.selection.kind === "symbol") deleteSymbol(state.selection.id);
  else if (state.selection.kind === "wire") deleteWire(state.selection.id);
  state.selection = null;
  setStatus("Deleted");
}

// --------------------------------------------------------------------------
// Canvas events
// --------------------------------------------------------------------------

canvas.addEventListener("dragover", (e) => {
  if (e.dataTransfer.types.includes(DRAG_MIME)) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }
});

canvas.addEventListener("drop", (e) => {
  const src = e.dataTransfer.getData(DRAG_MIME);
  if (!src) return;
  e.preventDefault();
  const { x, y } = clientToCanvas(e.clientX, e.clientY);
  placeSymbol(src, x, y);
});

canvas.addEventListener("mousedown", (e) => {
  const target = /** @type {Element} */ (e.target);

  // pin → handled on click, ignore mousedown
  if (target.classList && target.classList.contains("pin")) return;

  // empty canvas → clear selection (and wire-in-progress on click below)
  if (target.id === "canvas-bg" || target === canvas) return;

  // placed symbol body → start drag
  const placed = target.closest(".placed");
  if (placed) {
    const id = placed.getAttribute("data-sym-id");
    if (!id) return;
    const sym = state.symbols.get(id);
    if (!sym) return;
    const { x, y } = clientToCanvas(e.clientX, e.clientY);
    state.drag = { symId: id, dx: x - sym.x, dy: y - sym.y };
    selectSymbol(id);
    e.preventDefault();
  }
});

window.addEventListener("mousemove", (e) => {
  // wire preview
  if (state.wireStart) {
    const { x, y } = clientToCanvas(e.clientX, e.clientY);
    updatePreviewTo(x, y);
  }
  // symbol drag
  if (state.drag) {
    const { x, y } = clientToCanvas(e.clientX, e.clientY);
    const nx = snap(x - state.drag.dx);
    const ny = snap(y - state.drag.dy);
    moveSymbol(state.drag.symId, nx, ny);
  }
});

window.addEventListener("mouseup", () => {
  state.drag = null;
});

canvas.addEventListener("click", (e) => {
  const target = /** @type {Element} */ (e.target);

  // pin → wire start/finish
  if (target.classList && target.classList.contains("pin")) {
    const symId = target.getAttribute("data-sym-id");
    const pin = target.getAttribute("data-pin");
    if (!symId || !pin) return;
    if (!state.wireStart) startWire(symId, pin);
    else finishWire(symId, pin);
    return;
  }

  // wire path → select
  if (target.classList && target.classList.contains("wire")) {
    const id = target.getAttribute("data-wire-id");
    if (id) selectWire(id);
    return;
  }

  // empty canvas → cancel wire / clear selection
  if (target.id === "canvas-bg" || target === canvas) {
    if (state.wireStart) cancelWire();
    clearSelection();
  }
});

// keyboard
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (state.wireStart) cancelWire();
    else clearSelection();
  } else if (e.key === "Delete" || e.key === "Backspace") {
    // ignore when typing into form fields (none here, but defensive)
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
    if (state.selection) {
      e.preventDefault();
      deleteSelection();
    }
  }
});

// --------------------------------------------------------------------------
// Toolbar
// --------------------------------------------------------------------------

document.getElementById("export-btn").addEventListener("click", () => {
  const data = {
    symbols: [...state.symbols.entries()].map(([id, s]) => ({
      id,
      src: s.src,
      x: s.x,
      y: s.y,
    })),
    wires: state.wires.map((w) => ({ id: w.id, from: w.from, to: w.to })),
  };
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "schematic.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${data.symbols.length} symbols, ${data.wires.length} wires`);
});

document.getElementById("clear-btn").addEventListener("click", () => {
  if (state.symbols.size === 0 && state.wires.length === 0) return;
  if (!confirm("Clear the entire schematic?")) return;
  clearSchematic();
  setStatus("Cleared");
});

/**
 * Drop every symbol and wire without prompting. Used by Clear (after confirm)
 * and by Import (which replaces the current schematic with the loaded one).
 */
function clearSchematic() {
  for (const id of [...state.symbols.keys()]) deleteSymbol(id);
  state.symbols.clear();
  state.wires = [];
  state.selection = null;
  cancelWire();
}

// --- Import ---------------------------------------------------------------

const importBtn = document.getElementById("import-btn");
const importFile = document.getElementById("import-file");

importBtn.addEventListener("click", () => {
  // Reset the input so the same file twice in a row still fires `change`.
  importFile.value = "";
  importFile.click();
});

importFile.addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    const summary = importSchematic(data);
    setStatus(
      `Imported ${summary.symbols} symbols, ${summary.wires} wires` +
        (summary.warnings.length ? ` (${summary.warnings.length} warning${summary.warnings.length === 1 ? "" : "s"} — see console)` : ""),
    );
    if (summary.warnings.length) {
      console.warn("Import warnings:", summary.warnings);
    }
  } catch (err) {
    console.error("Import failed:", err);
    setStatus(`Import failed: ${err.message}`);
  }
});

/**
 * Replace the current schematic with one parsed from the export schema:
 *   { symbols: [{id, src, x, y}], wires: [{from:[symId,pin], to:[symId,pin]}] }
 *
 * Symbols whose `src` is not in the bundle are skipped (warned). Wires
 * referencing missing symbols or pins are skipped (warned). Original ids
 * are preserved so subsequent re-exports stay diff-friendly; `nextSymId`
 * and `nextWireId` are advanced past any numeric suffixes encountered.
 *
 * @returns {{symbols:number, wires:number, warnings:string[]}}
 */
function importSchematic(data) {
  const warnings = [];
  if (!data || typeof data !== "object") {
    throw new Error("not an object");
  }
  const symbols = Array.isArray(data.symbols) ? data.symbols : [];
  const wires = Array.isArray(data.wires) ? data.wires : [];

  clearSchematic();

  let maxSymN = 0;
  let placed = 0;
  for (const s of symbols) {
    if (!s || typeof s.id !== "string" || typeof s.src !== "string"
        || typeof s.x !== "number" || typeof s.y !== "number") {
      warnings.push(`malformed symbol entry: ${JSON.stringify(s)}`);
      continue;
    }
    if (!(window.SYMBOLS && window.SYMBOLS.files && window.SYMBOLS.files[s.src])) {
      warnings.push(`symbol src not in bundle, skipping: ${s.src} (id=${s.id})`);
      continue;
    }
    if (state.symbols.has(s.id)) {
      warnings.push(`duplicate symbol id in file: ${s.id}`);
      continue;
    }
    try {
      instantiateSymbol(s.id, s.src, s.x, s.y);
      placed++;
    } catch (err) {
      warnings.push(`failed to instantiate ${s.id} (${s.src}): ${err.message}`);
      continue;
    }
    const m = /^s(\d+)$/.exec(s.id);
    if (m) maxSymN = Math.max(maxSymN, parseInt(m[1], 10));
  }
  state.nextSymId = maxSymN + 1;

  let maxWireN = 0;
  let wired = 0;
  for (const w of wires) {
    if (!w || !Array.isArray(w.from) || !Array.isArray(w.to)
        || w.from.length !== 2 || w.to.length !== 2) {
      warnings.push(`malformed wire entry: ${JSON.stringify(w)}`);
      continue;
    }
    const [fromSym, fromPin] = w.from;
    const [toSym, toPin] = w.to;
    if (!state.symbols.has(fromSym) || !state.symbols.has(toSym)) {
      warnings.push(`wire references missing symbol: ${fromSym} → ${toSym}`);
      continue;
    }
    if (!state.symbols.get(fromSym).pinsByName.has(fromPin)
        || !state.symbols.get(toSym).pinsByName.has(toPin)) {
      warnings.push(`wire references missing pin: ${fromSym}/${fromPin} → ${toSym}/${toPin}`);
      continue;
    }
    const id = typeof w.id === "string" && w.id ? w.id : `w${state.nextWireId++}`;
    const wire = { id, from: [fromSym, fromPin], to: [toSym, toPin] };
    state.wires.push(wire);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("class", "wire");
    path.setAttribute("data-wire-id", id);
    layerWires.appendChild(path);
    updateWirePath(wire);
    wired++;
    const m = /^w(\d+)$/.exec(id);
    if (m) maxWireN = Math.max(maxWireN, parseInt(m[1], 10));
  }
  state.nextWireId = Math.max(state.nextWireId, maxWireN + 1);

  return { symbols: placed, wires: wired, warnings };
}

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------

buildLibrary();
setStatus("Ready");
