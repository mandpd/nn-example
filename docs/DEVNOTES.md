# Dev notes — nn_example

Non-obvious decisions and gotchas behind the modernized visualizer. Read this before
changing `nn_example.html` / `js/nn_example.js`.

## Architecture

- Single-page app: `nn_example.html` (markup + all CSS inline) and `js/nn_example.js`
  (all logic). No external dependencies, works offline.
- `js/convnet/convnet.js` is Andrej Karpathy's engine, untouched. Only `convnet.js` is
  loaded — `util.js`, `vis.js`, `npgmain.js` and the old jQuery/Bootstrap/d3 stack are unused.

## Color semantics (CVD-validated dark palette)

- **Blue `#3987e5`** = dog / class 1 / positive weights & deltas.
- **Orange `#d95926`** = cat / class 0 / negative weights & deltas.
- **Gold `#c98500`** = the traced point, the timeline "now" line, selected node/edge,
  active phase chip, active mode — always "the thing in focus".
- **Violet `#9085e9`** = gradients (backward pass) and the loss sparkline.
- Red/green deliberately avoided (worst colorblind pairing; the original used it).

## Gotchas

- **`Net.toJSON()` aliases live weight arrays** — it does not copy. The rewind history
  uses explicit `Float64Array` copies (`weightsSnapshot` / `weightsRestore`).
- **Canvas ↔ flex/grid feedback loop**: `fitCanvas()` sets the canvas backing size to
  rect × devicePixelRatio. If a canvas's intrinsic size can influence its container's
  size, the layout inflates on every render. Hence the timeline card is
  `position: absolute` over the header's box, and the pass-mode network canvas has a
  fixed `clamp()` height. Never let these canvases participate in intrinsic sizing.
- **Jog-free swaps**: the toolbar's epoch/pass control sets share one grid cell
  (`.toolbar-stack`, both children `grid-area: 1/1`) toggled with `visibility`, so the
  toolbar height is the max of both by construction. The header/timeline swap uses the
  absolute overlay. Preserve these patterns when adding controls to either set.
- **Tooltips** are pure CSS (`[data-tip]::after`). The reveal delay lives on the *shown*
  state only (hide is immediate), and keyboard focus uses `:has(:focus-visible)` — not
  `:focus-within`, which pins tooltips open after mouse clicks because the control keeps
  focus. `tip-right` / `tip-below` variants exist for elements near page edges.
- **Render loop**: one `setInterval` tick (40 ms) trains-and-renders while playing, and
  renders once when `requestRender()` set a dirty flag. There is deliberately no
  `document.hidden` guard (it breaks headless/background rendering).
- **Trainer vs net rebuilds**: rate-slider changes swap only the Trainer (weights
  preserved, applied live); dataset/architecture changes rebuild the net and clear the
  run (`runActive` → title returns, controls unlock).
- **Single-pass mode shows real numbers**, captured from one batch-size-1
  `trainer.train()` call: `out_act.w` per layer (activations), `out_act.dw` (per-neuron
  gradients — the param update zeroes filter grads but not these), and pre/post weight
  deltas for the update phase.
- **Automation/testing**: browser-tool screenshots can be scaled — click coordinates are
  screenshot-space, not CSS pixels; convert via image-width / `innerWidth`.
