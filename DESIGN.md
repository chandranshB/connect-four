---
name: Connect Four
description: Clean Material Design 3 paper tabletop Connect Four against locally trained ONNX AI opponents.
colors:
  stone-linen-ground: "#E9E2D2"
  vellum-paper-sheet: "#FAF6EC"
  raised-rag-paper: "#FFFDF8"
  tonal-paper-container: "#EFE9DB"
  recessed-paper-tray: "#E2DAC7"
  ink-primary: "#18202C"
  ink-secondary: "#4D5460"
  slate-board: "#1B2330"
  slate-board-edge: "#273244"
  slate-well: "#0E141D"
  vermilion-coral: "#D63E30"
  saffron-amber: "#E4A31B"
  hint-cobalt: "#2563EB"
typography:
  display:
    fontFamily: "'Figtree Variable', 'Figtree', system-ui, -apple-system, sans-serif"
    fontSize: "1.95rem"
    fontWeight: 800
    lineHeight: 1.08
    letterSpacing: "-0.025em"
  body:
    fontFamily: "'Figtree Variable', 'Figtree', system-ui, -apple-system, sans-serif"
    fontSize: "0.85rem"
    fontWeight: 600
    lineHeight: 1.4
rounded:
  fab: "14px"
  board: "20px"
  arena: "24px"
  pill: "999px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-outlined:
    backgroundColor: "{colors.raised-rag-paper}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.pill}"
    padding: "0 18px"
  button-tonal:
    backgroundColor: "{colors.tonal-paper-container}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.pill}"
    padding: "0 18px"
  button-new-game:
    backgroundColor: "{colors.raised-rag-paper}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.pill}"
    padding: "0 18px 0 9px"
  arena-paper-sheet:
    backgroundColor: "{colors.vellum-paper-sheet}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.arena}"
    padding: "24px 28px 28px"
  board-frame:
    backgroundColor: "{colors.slate-board}"
    rounded: "{rounded.board}"
    padding: "18px"
---

## Overview

Connect Four pairs **Material Design 3 spacing and controls** with **archival paper surfaces** and tiled SVG `feTurbulence` **matte noise** textures (`--noise-paper` and `--noise-matte`). Stripped of technical dashboards, developer metrics, and spoilers, the entire interface centers on one tactile matte slate board resting on a warm rag-paper mat.

## Colors

- **Paper Surfaces:** `#E9E2D2` stone-linen desk ground, `#FAF6EC` elevated top bar and board mat, `#EFE9DB` tonal `Hint` fill, `#FFFDF8` lifted active tabs & `Undo` button, `#E2DAC7` recessed segmented trays.
- **Matte Board & Discs:** `#1B2330` matte obsidian-slate board slab with `#0E141D` recessed wells. Red (`#D63E30`, single inner ring) and Yellow (`#E4A31B`, double concentric rings) discs carry unglazed matte noise finish.
- **Hint Signal:** `#2563EB` dashed target ring appears on the board only when `Hint` is toggled on.

## Typography

- **Display & UI (`Figtree Variable`):** Friendly geometric humanist sans across display headlines (`Connect Four`, `Your Turn`) and crisp Material pill controls, self-hosted locally via `@fontsource-variable/figtree`.

## Layout

- **Material Top App Bar:** `Connect Four` on the left; `Easy / Medium / Pro` difficulty segmented pill, `Red / Yellow` color segmented pill, and the `New Game` paper pill (`42px`, circular matte slate `+` badge) on the right.
- **Single Centered Arena:** Generous centered `624px` Material paper mat (`24px` radius, `24px 28px 28px` padding) containing only the turn status + `Hint` / `Undo` actions, hover drop chute, and the 7x6 matte slate board.

## Elevation & Depth

- Soft Material shadow hierarchy (`--elev-1`, `--elev-2`, `--elev-board`) with `--ease-m3` transitions, radial ink-ripple press feedback (`::after`), gliding segmented paper thumbs (`.seg-thumb`), tactile landing well settle ripples, staggered 4-in-a-row win pops, and tiled `feTurbulence` matte grain across paper sheets, board slab, wells, and discs.

## Shapes

- Full-pill (`999px`) segmented controls and action buttons, `14px` Extended FAB, `20px` matte slate board slab, `24px` paper mat sheet, and circular wells/discs.

## Components

- **Segmented Pill Trays (`.segmented` / `.seg-btn.active`):** `42px` recessed troughs with lifted `#FFFDF8` tabs.
- **Board Actions (`.board-actions`):** Compact `36px` `Hint` (filled tonal / active toggle) and `Undo` (outlined paper pill).
- **Matte 7x6 Board (`.board-frame`):** Interactive column buttons with hover preview disc, gravity drop animation, last-move outline, and 4-in-a-row win ring.

## Do's and Don'ts

- **Do** keep the viewport uncluttered—no technical badges, MCTS sliders, win-probability spoilers, or checkpoint telemetry tables.
- **Do** preserve matte noise + soft layered Material elevation rather than glossy gradients or hard offset shadows.
- **Don't** reintroduce sidebars or telemetry panels unless explicitly requested.
