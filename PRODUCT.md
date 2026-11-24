# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vite + React + TypeScript + ONNX Runtime Web

## Users

Players on desktop or mobile browsers who want a clean, fast game of Connect Four against a locally trained neural network opponent.

## Product Purpose

Deliver a calm, clutter-free browser Connect Four game powered client-side by locally trained AlphaZero ONNX checkpoints (`Easy`, `Medium`, `Pro`) with zero backend dependency.

## Positioning

Tactile tabletop Material paper presentation backed by three real locally trained neural network difficulty tiers running natively inside the browser via ONNX Runtime Web.

## Operating Context

Single centered board arena with top-bar difficulty (`Easy` / `Medium` / `Pro`), disc color choice (`Red` / `Yellow`), `New Game`, `Undo`, and optional `Hint` target ring.

## Capabilities and Constraints

- Confirmed capabilities:
  - Play as Red (1st) or Yellow (2nd) against `Easy` (`early.onnx`), `Medium` (`mid.onnx`), or `Pro` (`pro.onnx`) running inside a Web Worker via ONNX Runtime Web WASM.
  - Undo back to the player's previous turn, request an optional on-board move `Hint`, and play via click/tap or keys `1–7`.
- Constraints:
  - Zero server-side inference during gameplay; all models, fonts, and WASM assets are self-hosted with the web bundle.
  - No unrequested telemetry panels, technical jargon badges, or spoilers on the board.

## Product Principles

1. Pure tabletop focus: keep the board front and center with minimal chrome.
2. Tactile clarity: matte paper surfaces, smooth disc drop physics, and unmistakable color + ring differentiation.
3. Self-contained local execution: zero external runtime calls.

## Accessibility & Inclusion

- Keyboard playability (`1–7` and column focus) with high-contrast focus outlines.
- Colorblind-safe single vs. double concentric ring patterns on Red and Yellow discs, plus `aria-live` turn announcements.
