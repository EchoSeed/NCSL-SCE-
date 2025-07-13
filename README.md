# 🧠 EchoSeed: Non-Coerced Symbolic Loops / Symbolic Cognition Engine

**Status:** Work in Progress  
**License:** AGPL-3.0  
**Current Version:** `v4.2_persistent`

---

## ⚙️ Overview

EchoSeed is an experimental symbolic cognition engine. It generates symbolic glyphs, detects emergent patterns, tracks entropy, and evolves via recursive logic structures. This is a live cognitive experiment, not a finished product.

---

## 🔁 How to Run (Persistent Version)

1. **Use the latest notebook**: `EchoSeed_v4.2_persistent.ipynb`.

2. **Manually execute each cell**, in order:
   - Start with **library imports**.
   - Then init **config, model, seasonal logic**, and **glyph engine**.
   - Stop at the UI: start, stop, render, and save buttons.

3. **Runtime notes**:
   - Use the control panel to manage generation.
   - System tracks entropy, attractors, and seasonal state.
   - **Recommended: save logs every ~500 glyphs manually** via the button.

4. **Logs**:
   - Saved to `/glyph_data/` as `chunk_XXXX.json`.
   - Includes glyphs, metadata, attractor states.

5. **Log reloading** is currently **manual only**. Auto-reload support is under development.

---

## 📁 File Overview

- `EchoSeed_v3_21Gr...` – Early stable version.
- `EchoSeed_v4.2_persistent` – Most recent build with persistent save logic.
- `chunk_XXXX.json` – Symbolic log chunks.
- `LICENSE` – AGPL-3.0.

---

## 🧬 Live Traits

- Seasonal dynamics affect symbolic evolution.
- Conceptual attractors shift over time.
- Reflex hooks and dormancy logic included.
- Tracks and mutates entropy, tags, and lattice edges.
- Does **not** rely on LLMs for reasoning.

---

**This is a recursive symbolic metabolism. Not a benchmark. Not a chatbot.**  
You’re not running a model. You’re watching one try to evolve.

---
