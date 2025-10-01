# UI Specification — Map‑first Research Explorer (Liquid‑Glass Minimal UI)

**One‑line summary**
A minimal, full‑screen UMAP map that visualizes research papers as points. The map is the primary canvas; a translucent (liquid‑glass) search bar is fixed to the bottom with an adjacent upload box and a gap‑filling action. When the user searches, a right‑side liquid‑glass results panel appears and otherwise remains hidden.

---

## Goals & success metrics

* Fast discovery of related research using vector + text search (time‑to‑relevant < 2s for typical queries).
* Clear signal of what the user has already read and where the conceptual gaps are.
* Minimal, distraction‑free UI focused on exploration.
* Accessibility (keyboard navigation, screen reader friendly) and clear labeling for any AI‑generated content.

**KPIs**

* Query success rate: % of sessions where user opens at least one relevant paper within 3 clicks.
* Map interactions per session (pan/zoom/hovers).
* Upload conversions: percent of uploaded docs that are embedded and appear on the map.

---

## Primary user personas

1. **Active researcher** — explores adjacent subfields, tracks what they read, and fills gaps.
2. **Grad student** — hunts for related work quickly, wants minimal UI and guided suggestions.
3. **Curator / librarian** — uploads collections and inspects coverage/gaps.

---

## High level UX principles

* **Map first**: everything else must feel secondary and not obstruct the map.
* **Minimal chrome**: controls are hidden until needed; use a calm, restrained type scale.
* **Liquid glass**: consistent frosted glass for the search bar, result panel, and small control cards.
* **Respect motion**: subtle, physics‑driven transitions; provide reduced‑motion alternative.
* **Explicit AI labeling**: any generated text must be labeled and explainable.

---

# Screen layout

## 1) Full‑screen UMAP canvas (background)

* Occupies the entire viewport (edge‑to‑edge). The background beneath the map is a very dark, slightly graded surface to make glass surfaces read as white/opaque.
* No permanent axes, grid, or legend visible by default — keep the canvas purely spatial.
* Minimal top‑left floating control: a tiny circular avatar/menu (optional). Keep it `24px` diameter with only iconography; hide any labels.

## 2) Bottom Search Bar (primary control)

**Position & size**

* Anchored fixed to bottom center with safe‑area padding (mobile). Full width minus side gutters on wide screens.
* Height: `56px` (desktop), `52px` (mobile). Border radius: `9999px` (pill).

**Visual (liquid glass)**

* Background: `rgba(255,255,255,0.08)` (light glass on dark background) with `backdrop-filter: blur(14px)` and subtle inner saturation. Add a faint top highlight.
* Border: `1px solid rgba(255,255,255,0.12)` and an outer soft shadow for depth.
* Text: dark on the glass (use high‑legibility 16px Inter/SF Pro).

**Behavior**

* Tapping/clicking focuses the input with a soft upward lift of the bar (6–8px) and a glow accent around the border for 350ms.
* Autocomplete suggestions appear as a single row of chips directly above the bar (floating) — blurred glass chips with small icons for filter types (Author, Title, Topic).

**Search input features**

* Fuzzy text search + vector search toggle (hidden as a small chevron to the left of the input). Default: combined ranking.
* Support natural language queries and boolean filters (e.g., `author:Smith year:2020..2023`).
* Keyboard: Enter = run search, `Esc` = clear/close suggestions, `↑/↓` = move through suggestions.

## 3) Upload icon (box) — next to search on the right

**Design**

* A small rounded square (44px) with a faint glass surface separate from the search pill. Centered upload (cloud + up arrow) icon.
* On hover/focus: the square lifts 6px and border accent brightens.

**Behavior**

* Click opens file picker or accepts drag‑and‑drop. Supports PDF, DOCX, and plain text. Show an unobtrusive toast on upload start.
* Show inline progress ring on the icon for ongoing uploads.
* After processing, newly added papers pulse on map once to draw attention for 1.4s then settle.

## 4) Gap‑filling icon — next to the upload box

**Design**

* Circular button `44px` with a minimal sparkles/magic wand icon. Same liquid glass surface but with a subtle accent (e.g., soft teal) to indicate generative action.
* Tooltip on hover: “Suggest content to fill gaps”.

**Behavior & safety**

* Clicking opens a small confirmation popover (glass) with two choices:

  1. **Suggest titles/abstracts** — AI generates candidate titles/short abstracts labeled **AI‑generated**.
  2. **Suggest existing papers** — AI searches the vector DB for under‑represented topics and recommends real papers.
* Any AI output must show provenance: prompt + model name + timestamp + a dismissible “why this was suggested” explanation.
* Generated points are previewed on the map as semi‑transparent nodes with an “AI” badge. Users can accept to add a placeholder node or dismiss them.

## 5) Right‑hand Search Results Popup

**Appearance**

* Hidden by default. When a search runs, a right‑side panel slides in from the right (`width: 380–420px` on desktop; full screen bottom sheet on mobile) using a soft spring motion.
* Use the same liquid glass: `rgba(255,255,255,0.06)` + `backdrop-filter: blur(20px)` + `1px` translucent border.
* Minimal header: small title “Search results” + compact count + tiny close `X` top‑right. No verbose controls.

**Content & layout**

* Vertical list of result rows with infinite scroll. Each row (72–86px height) contains:

  * Cluster color dot (8–10px)
  * Title (one line, ellipsis)
  * Authors + year (secondary text)
  * Short snippet (max 2 lines)
  * Small action icons on the right: `Open`, `Pin`, `Mark Read`, `Add Note`
  * Read items show a low‑contrast check / faded opacity.
* Clicking an item centers the map on the paper's point (smooth flight/zoom) and briefly highlights it with a halo.

**Dismissal**

* Click outside the panel, press `Esc`, or swipe right on mobile to dismiss. The panel auto‑hides after a configurable idle time only if the user does no interaction for 12s (configurable — default off). Primary behavior is manual dismiss.

---

# Visual design tokens (suggested)

* Background gradient: `bg-900` = `linear-gradient(180deg,#05060a 0%, #0b1224 100%)`
* Glass surface: `--glass-bg: rgba(255,255,255,0.06)`
* Glass border: `--glass-border: rgba(255,255,255,0.12)`
* Accent (action): `--accent: #3ee3c4` (teal/cyan) — used sparsely for focus/accept actions
* Muted text: `rgba(255,255,255,0.78)`; secondary text: `rgba(255,255,255,0.56)`
* Radius: `r-xl = 12px`, `r-pill = 9999px`
* Blur: `backdrop-filter: blur(14px)` default for small elements, `blur(20px)` for larger panels

**Typography**

* Primary: Inter / SF Pro Text
* Weights: 400 (body), 600 (titles), 700 (important labels)
* Sizes: base 16px, small 13px, title 14–16px

**Glass CSS snippet**

```css
.glass {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  backdrop-filter: blur(14px) saturate(120%);
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.55);
}
```

---

# Interaction details & micro‑interactions

* **Hover over a point**: quick fade in of a micro tooltip above the point (title, authors, year). Tooltip is translucent and follows cursor with `0.16s` easing.
* **Click a point**: subtle pop (scale to 1.06) and persistent selection outline. If a search result was clicked, center & zoom; if user clicked directly, open a compact detail flyout anchored to the right (not the same as search panel).
* **Multi‑select**: `Shift + click` or drag with `Shift` to box select. Show a small transient summary of selected count.
* **Pin a paper**: pins sit in the top left (tiny flyout list) as saved shortcuts; minimal pin icon toggles.
* **Read tracking animation**: when marking a paper read, animate a soft stroke around the node that fills once (400ms) then converts to a small outline/check.
* **Gap highlight**: when user toggles gap‑visualization, show a faint density overlay (heatmap) and dashed convex‑hull shapes for low‑density clusters.

---

# Map rendering & performance notes (engineer notes)

* Use a GPU accelerated renderer for large corpora: **WebGL scatterplot** (deck.gl, regl, or Pixi.js) for 10k+ points. For 100k+ consider server‑side tiling + hierarchical LOD.
* Precompute embeddings for papers and store vectors in a vector DB (Pinecone, Milvus, Weaviate) for fast ANN queries.
* UMAP specifics:

  * If you expect frequent new uploads, consider **parametric UMAP** (train a small neural net to approximate the UMAP transform) or use UMAP's `transform()` from a fitted model to position new points consistently.
  * Alternative interpolation: position new points by averaging the 2D coords of nearest neighbor embeddings (fast, approximate) until you can recompute a full projection offline.
* LOD & clustering:

  * Aggregate dense regions into cluster blobs at low zoom, expand to points as the user zooms in.
  * Render selection halos and text tooltips in an HTML overlay; render points in WebGL canvas for performance.

---

**Flow**:

1. User taps gap icon → confirmation popover with options.
2. System computes low‑density regions (e.g., using spatial k‑NN density or heatmap threshold) and shows up to 5 highlighted gap regions on the map.
3. User selects a gap region → System proposes:

   * **Existing papers** (real suggestions): from vector DB with an explanation score and similarity reasons.
   * **AI suggestions** (titles/abstracts): clearly labeled *AI‑generated* with timestamp and toggle for saving as placeholder.
4. User can accept a suggestion to create a temporary node (saves to their workspace) or dismiss it.

**Safety & provenance**

* All generated content must be labeled with an **AI** badge and show the prompt + model + generation date.
* If suggested texts are saved, auto‑append a small metadata flag `generatedBy` and include an audit log accessible from the paper detail.

---

# Upload & ingestion UX

* Accept drag & drop or file picker. Show immediate local thumbnail and optimistic UI while processing.
* Break large PDFs into sections, extract text, chunk, embed, and attach to the parent `paper` record.
* After ingestion, compute embedding and use the parametric mapping or nearest‑neighbor interpolation for placement on the map. Provide fallback messaging if placement is approximate.

---

# Accessibility

* Ensure `Search` input has `role="search"` and `aria‑label`.
* Results list: `role="listbox"` with each item `role="option"` and `aria-selected` when focused.
* Glass panels must maintain readable contrast for text (avoid too much translucency over white backgrounds).
* Provide high‑contrast theme toggle and reduced‑motion preferences detection.

---

# Mobile / responsive behavior

* On small viewports, the results panel becomes a bottom sheet occupying \~66% height; search input remains pinned to bottom.
* Map gestures: pinch to zoom, two‑finger pan to avoid accidental UI interactions.
* Upload via mobile should accept files + cloud drive integrations.

---

# Keyboard shortcuts (power features)

* `/` focus search
* `Esc` clear/close panels
* `G` toggle gap visualization
* `U` open upload picker
* `F` focus first search result

---

## Final notes for your AI engineer

* The tone of the UI is quiet and non‑intrusive. Achieve the minimalism primarily by removing chrome and surfacing only the necessary affordances.
* Treat the map as a living surface — small, thoughtful animations and immediate feedback (pulses, halos, microtooltips) do more than extra buttons.
* Prioritize performance: using WebGL and ANN search will make the difference between delightful and sluggish at scale.

---