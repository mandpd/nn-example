/* nn_example.js — interactive, layer-by-layer neural net visualizer.
   Engine: Andrej Karpathy's convnetjs (js/convnet/convnet.js).
   UI: dependency-free. The network is configured entirely through the
   controls in nn_example.html; no code editing required. */
'use strict';

/* ---------------- palette (dark surface, CVD-validated) ---------------- */
const C = {
  surface : '#1a1a19',
  ink     : '#ffffff',
  inkSec  : '#c3c2b7',
  muted   : '#898781',
  grid    : '#242422',
  axis    : '#3a3a37',
  // internal names kept from the original cats-vs-dogs fork:
  // DOG/dog = class 1 = blue  = ok ride  (PIREP grade 1–3, ≤ moderate turbulence)
  // CAT/cat = class 0 = orange = rough   (PIREP grade 4–6, ≥ moderate chop → divert)
  dog     : '#3987e5',   // class 1
  cat     : '#d95926',   // class 0
  // blue recedes on the dark surface where orange pops — the heavier blue
  // alpha makes the two regions read at equal strength
  dogWash : 'rgba(57, 135, 229, 0.30)',
  catWash : 'rgba(217, 89, 38, 0.17)',
  tag     : '#c98500',   // alt-click "tagged" point
  mesh    : 'rgba(255, 255, 255, 0.34)',
  meshMini: 'rgba(255, 255, 255, 0.26)',
  spark   : '#9085e9',
};

const DOG = 1, CAT = 0;
const TICK_MS = 40;        // train/render cadence
const EPOCHS_PER_TICK = 10;
const GRID_COLS = 65;      // forward-pass samples per axis (odd, so 0 is sampled)
const GRID_STEP = 2;       // mesh keeps every 2nd sample
const MAX_HIDDEN_LAYERS = 4;

/* ---------------- state ---------------- */
const state = {
  hidden: [4, 4],          // neurons per hidden layer
  activation: 'relu',
  lr: 0.003, momentum: 0.1, batch: 10, l2: 0.001,
  dataset: 'circle',
  overlay: true,           // shade prediction regions
  playing: false,
  mode: 'epoch',           // 'epoch' | 'pass' (single-pass walkthrough)
  lix: 1,                  // selected layer index into net.layers
  d0: 0, d1: 1,            // neuron pair shown in the selected-layer view
  view: { kind: 'layer' }, // inspector mode: layer | node {rowIdx,j} | edge {rowIdx,i,j}
  epoch: 0, loss: null, lossHist: [],
  lossCurve: [],           // {e, v} samples for the timeline's loss trail
};

let net = null, trainer = null;
let data = [], labels = [], tagged = [];
let selectedPt = -1;       // data point whose action bar is open
let field = null;          // last computed forward-pass field
let needsRender = true;
let minis = [];            // pipeline {el, cv} per layer
let archRows = [];         // network-diagram rows {y, idxs} for hit-testing
let archNodes = [];        // network-diagram nodes {x, y, rowIdx, j}
let archEdges = [];        // network-diagram edges {x1,y1,x2,y2, rowIdx, i, j}
let archActs = [];         // network-diagram activation bands {y, rowIdx, actIdx, xMin, xMax}
let archMatX = Infinity;   // left edge of the pass-mode matrix panel (clicks beyond it are ignored)

const LR_STEPS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
const L2_STEPS = [0, 0.0001, 0.001, 0.01];
const BATCH_STEPS = [1, 2, 5, 10, 20, 50];

const $ = (s) => document.querySelector(s);
const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n) => String(n).replace(/\d/g, (c) => SUBS[+c]);
const MONO = '11px ui-monospace, "SF Mono", Menlo, Consolas, monospace';

/* ---------------- datasets ---------------- */
const DATASETS = {
  simple() {
    const pts = [
      [-0.4326, 1.1909, DOG], [3.0, 4.0, DOG], [0.1253, -0.0376, DOG],
      [0.2877, 0.3273, DOG], [-1.1465, 0.1746, DOG], [1.8133, 1.0139, CAT],
      [2.7258, 1.0668, CAT], [1.4117, 0.5593, CAT], [4.1832, 0.3044, CAT],
      [1.8636, 0.1677, CAT], [0.5, 3.2, DOG], [0.8, 3.2, DOG], [1.0, -2.2, DOG],
    ];
    for (const [x, y, l] of pts) { data.push([x, y]); labels.push(l); }
  },
  circle() {
    for (let i = 0; i < 50; i++) {
      const r = convnetjs.randf(0.0, 2.0), t = convnetjs.randf(0, 2 * Math.PI);
      data.push([r * Math.sin(t), r * Math.cos(t)]); labels.push(DOG);
    }
    for (let i = 0; i < 50; i++) {
      const r = convnetjs.randf(3.0, 5.0), t = 2 * Math.PI * i / 50.0;
      data.push([r * Math.sin(t), r * Math.cos(t)]); labels.push(CAT);
    }
  },
  spiral() {
    const n = 100;
    for (let i = 0; i < n; i++) {
      const r = i / n * 5 + convnetjs.randf(-0.1, 0.1);
      const t = 1.25 * i / n * 2 * Math.PI + convnetjs.randf(-0.1, 0.1);
      data.push([r * Math.sin(t), r * Math.cos(t)]); labels.push(DOG);
    }
    for (let i = 0; i < n; i++) {
      const r = i / n * 5 + convnetjs.randf(-0.1, 0.1);
      const t = 1.25 * i / n * 2 * Math.PI + Math.PI + convnetjs.randf(-0.1, 0.1);
      data.push([r * Math.sin(t), r * Math.cos(t)]); labels.push(CAT);
    }
  },
  random() {
    for (let k = 0; k < 40; k++) {
      data.push([convnetjs.randf(-3, 3), convnetjs.randf(-3, 3)]);
      labels.push(convnetjs.randf(0, 1) > 0.5 ? DOG : CAT);
    }
  },
};

function setData(name, animate) {
  state.dataset = name;
  data = []; labels = [];
  DATASETS[name]();
  tagged = data.map(() => false);
  selectPoint(-1);
  clearPass();
  setRunActive(false);
  clearHistory();
  endTimelinePreview(); // a fresh dataset is a reset: the title returns
  state.epoch = 0; state.loss = null; state.lossHist = []; state.lossCurve = [];
  document.querySelectorAll('#datasetSeg button').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.set === name));
  });
  if (animate) animateDataArrival(); else clearDataReveal();
  markDescentDirty();
  requestRender();
}

/* ---------------- network ---------------- */
function buildNet() {
  const defs = [{ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 }];
  for (const n of state.hidden) {
    defs.push({ type: 'fc', num_neurons: n, activation: state.activation });
  }
  defs.push({ type: 'softmax', num_classes: 2 });
  net = new convnetjs.Net();
  net.makeLayers(defs);
  buildTrainer();
  clearPass();
  setRunActive(false); // fresh weights: title returns until the next run starts
  endTimelinePreview();
  clearHistory();
  state.epoch = 0; state.loss = null; state.lossHist = []; state.lossCurve = [];
  state.lix = net.layers.length - 1; // fresh net: start on the final output
  state.view = { kind: 'layer' }; // node/edge references die with the old net
  state.d0 = 0;
  state.d1 = Math.min(1, net.layers[state.lix].out_depth - 1);
  buildPipeline();
  updateSelectionUI();
  markDescentDirty();
  requestRender();
}

function buildTrainer() {
  trainer = new convnetjs.Trainer(net, {
    learning_rate: state.lr, momentum: state.momentum,
    batch_size: state.batch, l2_decay: state.l2,
  });
}

/* --- run state: title/description vs epoch timeline, and control locking --- */
let runActive = false;     // true from the first training step until Reset weights

function setRunActive(on) {
  if (runActive === on) return;
  runActive = on;
  if (on) clearDataReveal(); // training uses every point: snap them all in
  updateHeaderSwap();
  applyRunLocks();
  updatePlayLabel();
  requestRender();
}

/* the timeline replaces the header whenever it's meaningful: during a run, or
   any time single-pass mode is open (each pass advances the epoch count).
   Inference mode never trains, so it always keeps the title. Clicking the
   timeline's Describe card while no run is showing it opens a *preview* that
   lasts until the card is deselected or the page state is reset. */
let timelinePreview = false;

function timelineShown() {
  return state.mode === 'pass'
    || (state.mode === 'epoch' && (runActive || timelinePreview));
}

function endTimelinePreview() {
  if (!timelinePreview) return;
  timelinePreview = false;
  const card = document.getElementById('desc-timeline');
  if (card) card.classList.remove('hl');
  updateHeaderSwap();
  requestRender();
}

function updateHeaderSwap() {
  // visibility swap over the header's absolute box: content below never moves
  const on = timelineShown();
  $('#pageHeader').classList.toggle('swapped-out', on);
  $('#timeline').classList.toggle('swapped-out', !on);
}

/* dataset and architecture are fixed for the duration of a run — the train has
   left the station; Reset weights opens them up again */
function applyRunLocks() {
  document.querySelectorAll('#datasetSeg button').forEach((b) => { b.disabled = runActive; });
  $('#activationSel').disabled = runActive;
  document.querySelectorAll('#layerChips input, #layerChips button')
    .forEach((el) => { el.disabled = runActive; });
  // rate sliders: set before a run, or retune live while it trains — but a
  // paused run is a frozen snapshot, so no edits there
  const frozen = runActive && !state.playing;
  ['lrSlider', 'momentumSlider', 'batchSlider', 'l2Slider']
    .forEach((id) => { $('#' + id).disabled = frozen; });
}

function timelineCeiling(e) {
  let c = 200;
  const mult = [2.5, 2, 2]; // 200 → 500 → 1000 → 2000 → 5000 …
  let i = 0;
  while (c < e * 1.08 + 1) { c *= mult[i % 3]; i++; }
  return c;
}

function drawTimeline() {
  const wrap = $('#timeline');
  if (wrap.classList.contains('swapped-out')) return;
  const { ctx, w, h } = fitCanvas($('#timelineCanvas'));
  ctx.clearRect(0, 0, w, h);
  const padL = 16, padR = 16;
  const baseY = h - 20;
  // while scrubbing, the axis stays pinned to the run's true tip so the
  // scale doesn't shrink under the pointer mid-drag
  const dom = timelineCeiling(scrub ? scrub.tip.epoch : state.epoch);
  const xOf = (e) => padL + e / dom * (w - padL - padR);

  // track + ticks
  ctx.strokeStyle = C.axis;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padL, baseY);
  ctx.lineTo(w - padR, baseY);
  ctx.stroke();
  ctx.font = '9px ui-monospace, Menlo, monospace';
  ctx.fillStyle = C.muted;
  ctx.lineWidth = 1;
  for (const v of niceTicks(0, dom, 9)) {
    if (v > dom + 1e-9) continue;
    const x = xOf(v);
    ctx.beginPath();
    ctx.moveTo(x, baseY - 4);
    ctx.lineTo(x, baseY + 4);
    ctx.stroke();
    ctx.textAlign = x < padL + 8 ? 'left' : x > w - padR - 8 ? 'right' : 'center';
    ctx.fillText(tickLabel(v), x, baseY + 15);
  }
  ctx.textAlign = 'left';

  // progress up to "now"
  const xNow = xOf(state.epoch);
  const live = state.playing || !!(pass && pass.timer);
  if (xNow > padL + 1) {
    const grad = ctx.createLinearGradient(padL, 0, xNow, 0);
    grad.addColorStop(0, 'rgba(201,133,0,0.15)');
    grad.addColorStop(1, C.tag);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(padL, baseY);
    ctx.lineTo(xNow, baseY);
    ctx.stroke();
  }

  // loss trail: the value the run is optimizing, rolling out behind "now".
  // y is normalized to the run's own worst loss, so the shape reads as
  // "how far it has fallen" no matter the dataset
  const curve = state.lossCurve;
  // the y-scale (and the ghost) always come from the full run, so scrubbing
  // back doesn't rescale the trail under the pointer
  const refCurve = scrub ? scrub.tip.curve : curve;
  if (refCurve.length > 1) {
    const top = 26, bot = baseY - 8;
    let vmax = 0;
    for (const p of refCurve) if (p.v > vmax) vmax = p.v;
    if (vmax > 0 && bot > top) {
      const yOf = (v) => bot - (v / vmax) * (bot - top);
      const trace = (c) => {
        ctx.beginPath();
        c.forEach((p, i) => {
          const x = xOf(p.e), y = yOf(p.v);
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.stroke();
      };
      if (scrub) {
        // ghost of the run's true tip: the future the drag can return to
        ctx.save();
        ctx.globalAlpha = 0.3;
        ctx.strokeStyle = C.spark;
        ctx.lineWidth = 1.5;
        trace(refCurve);
        ctx.strokeStyle = C.tag;
        const gx = xOf(scrub.tip.epoch);
        ctx.beginPath();
        ctx.moveTo(gx, baseY + 5);
        ctx.lineTo(gx, 14);
        ctx.stroke();
        ctx.restore();
      }
      if (curve.length > 1) {
        ctx.strokeStyle = C.spark;
        ctx.lineWidth = 1.5;
        trace(curve);
        // leading dot riding the now line
        const tip = curve[curve.length - 1];
        ctx.beginPath();
        ctx.arc(xOf(tip.e), yOf(tip.v), 2.4, 0, Math.PI * 2);
        ctx.fillStyle = C.spark;
        ctx.fill();
      }
    }
  }

  // the "now" line
  ctx.strokeStyle = live ? C.tag : C.inkSec;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xNow, baseY + 5);
  ctx.lineTo(xNow, 14);
  ctx.stroke();
  ctx.beginPath();
  const r = live ? 3.2 + 1.3 * Math.sin(performance.now() / 160) : 3;
  ctx.arc(xNow, baseY, Math.max(r, 1.6), 0, Math.PI * 2);
  ctx.fillStyle = live ? C.tag : C.inkSec;
  ctx.fill();

  ctx.font = '600 11px ui-monospace, Menlo, monospace';
  const lbl = `epoch ${state.epoch.toLocaleString()}${live ? '' : ' · paused'}`;
  const tw = ctx.measureText(lbl).width;
  let lx = xNow + 9;
  if (lx + tw > w - 6) lx = xNow - 9 - tw;
  ctx.fillText(lbl, lx, 20);
  if (state.loss != null) {
    ctx.fillStyle = C.spark;
    const ll = `loss ${state.loss.toFixed(3)}`;
    const lw = ctx.measureText(ll).width;
    let lx2 = xNow + 9;
    if (lx2 + lw > w - 6) lx2 = xNow - 9 - lw;
    ctx.fillText(ll, lx2, 34);
  }
}

/* --- draggable now line: scrub a paused run back through its history --- */
let scrub = null; // { tip, k, moved } while a drag is in flight

function initTimelineScrub() {
  const cv = $('#timelineCanvas');
  const geo = () => {
    const r = cv.getBoundingClientRect();
    return { r, padL: 16, padR: 16, span: r.width - 32 };
  };
  const eligible = () =>
    state.mode === 'epoch' && runActive && history.length > 0 && timelineShown();
  const xNowOf = () => {
    const { padL, span } = geo();
    const dom = timelineCeiling(scrub ? scrub.tip.epoch : state.epoch);
    return padL + state.epoch / dom * span;
  };
  // restore a recorded moment without touching the history stack — the drag
  // is a preview; only pointerup commits
  const applySnap = (s) => {
    weightsRestore(s.weights);
    state.epoch = s.epoch;
    state.loss = s.loss;
    state.lossHist = s.hist;
    state.lossCurve = s.curve;
    requestRender();
  };

  cv.addEventListener('pointerdown', (e) => {
    if (!eligible()) return;
    const { r } = geo();
    if (Math.abs((e.clientX - r.left) - xNowOf()) > 9) return;
    if (state.playing) setPlaying(false); // scrubbing is a paused-world activity
    scrub = {
      tip: {
        epoch: state.epoch, loss: state.loss,
        hist: state.lossHist.slice(), curve: state.lossCurve.slice(),
        weights: weightsSnapshot(),
      },
      k: null, moved: false,
    };
    try { cv.setPointerCapture(e.pointerId); } catch { /* synthetic pointers */ }
    cv.style.cursor = 'w-resize';
    e.preventDefault();
  });

  cv.addEventListener('pointermove', (e) => {
    const { r, padL, span } = geo();
    const x = e.clientX - r.left;
    if (!scrub) {
      cv.style.cursor =
        eligible() && !state.playing && Math.abs(x - xNowOf()) <= 9 ? 'w-resize' : '';
      return;
    }
    const dom = timelineCeiling(scrub.tip.epoch);
    const target = Math.min(scrub.tip.epoch, Math.max(0, (x - padL) / span * dom));
    // snap to the nearest recorded moment: a history snapshot or the tip
    let k = null, best = Math.abs(scrub.tip.epoch - target);
    history.forEach((s, i) => {
      const d = Math.abs(s.epoch - target);
      if (d < best) { best = d; k = i; }
    });
    if (k === scrub.k) return;
    scrub.k = k;
    scrub.moved = true;
    applySnap(k === null ? scrub.tip : history[k]);
  });

  const endScrub = () => {
    if (!scrub) return;
    const { tip, k, moved } = scrub;
    scrub = null;
    if (k === null) {
      applySnap(tip); // released back at the true tip: nothing rewound
    } else {
      // commit: the run continues from here, and the snapshots above this
      // point are spent — exactly like clicking −1 epoch down to this moment
      const s = history[k];
      history = history.slice(0, k);
      applySnap(s);
      buildTrainer(); // momentum memory would be stale after the rewind
      setUndoDisabled(!history.length);
    }
    cv.style.cursor = '';
    // a drag shouldn't double as a panel click (describe-card highlight)
    if (moved) {
      cv.addEventListener('click',
        (e) => { e.stopPropagation(); e.preventDefault(); },
        { capture: true, once: true });
    }
    requestRender();
  };
  cv.addEventListener('pointerup', endScrub);
  cv.addEventListener('pointercancel', endScrub);
}

/* --- training, with a rewindable weight history --- */
let history = [];          // weight snapshots, one per training step
const HISTORY_CAP = 200;

/* convnetjs toJSON() aliases the live weight arrays, so snapshots must deep-copy */
function weightsSnapshot() {
  const out = [];
  net.layers.forEach((L, li) => {
    if (!L.filters) return;
    out.push({
      li,
      filters: L.filters.map((f) => Float64Array.from(f.w)),
      biases: Float64Array.from(L.biases.w),
    });
  });
  return out;
}

function weightsRestore(snap) {
  for (const s of snap) {
    const L = net.layers[s.li];
    s.filters.forEach((fw, j) => {
      const w = L.filters[j].w;
      for (let i = 0; i < fw.length; i++) w[i] = fw[i];
    });
    for (let i = 0; i < s.biases.length; i++) L.biases.w[i] = s.biases[i];
  }
}

function snapshot() {
  history.push({
    epoch: state.epoch, loss: state.loss,
    hist: state.lossHist.slice(), curve: state.lossCurve.slice(),
    weights: weightsSnapshot(),
  });
  if (history.length > HISTORY_CAP) history.shift();
  setUndoDisabled(false);
}

function setUndoDisabled(v) {
  $('#backBtn').disabled = v || state.playing;
  $('#passUndoBtn').disabled = v;
}

function clearHistory() {
  history = [];
  setUndoDisabled(true);
}

function trainEpochs(n) {
  const v = new convnetjs.Vol(1, 1, 2);
  let avloss = 0;
  for (let e = 0; e < n; e++) {
    for (let i = 0; i < data.length; i++) {
      v.w[0] = data[i][0]; v.w[1] = data[i][1];
      avloss += trainer.train(v, labels[i]).loss;
    }
  }
  return avloss / (n * data.length);
}

function recordLoss(avloss, n) {
  state.loss = avloss;
  state.lossHist.push(avloss);
  if (state.lossHist.length > 240) state.lossHist.shift();
  state.epoch += n;
  // the timeline's loss trail spans the whole run, so it can't shift out old
  // values the way lossHist does — decimate instead, keeping full-run coverage
  markDescentDirty(); // weights moved: the landscape needs a re-survey
  state.lossCurve.push({ e: state.epoch, v: avloss });
  if (state.lossCurve.length > 600) {
    const last = state.lossCurve[state.lossCurve.length - 1];
    state.lossCurve = state.lossCurve.filter((_, i) => i % 2 === 0);
    if (state.lossCurve[state.lossCurve.length - 1] !== last) state.lossCurve.push(last);
  }
}

function trainStep() {
  if (!data.length) return;
  snapshot();
  recordLoss(trainEpochs(EPOCHS_PER_TICK), EPOCHS_PER_TICK);
}

function stepForward() {
  if (!data.length || state.playing) return;
  clearPass();
  setRunActive(true);
  snapshot();
  recordLoss(trainEpochs(1), 1);
  requestRender();
}

function stepBack() {
  const s = history.pop();
  if (!s) return;
  clearPass();
  weightsRestore(s.weights);
  buildTrainer(); // momentum memory would be stale after the rewind
  state.epoch = s.epoch;
  state.loss = s.loss;
  state.lossHist = s.hist;
  state.lossCurve = s.curve;
  setUndoDisabled(!history.length);
  markDescentDirty();
  requestRender();
}

/* ---------------- "follow one pass": animated forward/backward walkthrough ---------------- */
let pass = null; // { p, steps, si, acts, grads, deltaByLi, dmax, loss, timer }
const PASS_STEP_MS = 900;
const PASS_PHASES = ['sample', 'forward', 'loss', 'backward', 'update'];

const ICON_PLAY = '<path d="M4 2.5v11l9-5.5z"/>';
const ICON_PAUSE = '<rect x="3" y="2" width="3.5" height="12" rx="1"/><rect x="9.5" y="2" width="3.5" height="12" rx="1"/>';

function setMode(m) {
  if (state.mode === m) return;
  if (m !== 'epoch' && state.playing) setPlaying(false);
  state.mode = m;
  document.querySelectorAll('.mode-seg button').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.mode === m));
  });
  disarmPirep(); // a half-placed report doesn't survive a mode change
  endTimelinePreview(); // previews belong to the mode they were opened in
  $('#panelControls').classList.toggle('pass-mode', m === 'pass');
  $('#panelControls').classList.toggle('infer-mode', m === 'infer');
  document.querySelector('.viz').classList.toggle('passing', m === 'pass');
  // frozen weights descend nowhere: inference hides the landscape everywhere
  $('#panelDescent').hidden = m === 'infer';
  $('#detTabDescent').hidden = m === 'infer';
  $('#detTabDescent3d').hidden = m === 'infer';
  if (m === 'infer' && detailTab !== 'layer') detailTab = 'layer';
  detailApplyTab();
  $('#describePane').dataset.mode = m; // Describe cards follow the visible panels
  // the side panel follows the mode: inference opens the decision guide
  $(m === 'infer' ? '#tabDecide' : '#tabIntro').click();
  updateHeaderSwap();
  clearPass(); // any running animation dies with the mode that owned it
  if (m === 'pass') {
    passIdleUI(); // wait for the Run button — nothing trains on mode entry
    updatePassRunUI();
  } else if (m === 'infer') {
    selectPoint(-1); // the point action bar belongs to the training data
    inferIdleUI();
    updateInferReadout();
  }
  applyFeaturePanelLabels();
  syncPassRowHeight();
  requestRender();
}

/* pass mode hides the feature panel — but that panel (canvas + legend + hint
   chrome) is exactly what pins epoch mode's row height. Measure it off-flow at
   epoch's column width and pin the pass row to the same height, so toggling
   modes never moves the pipeline panel. Only meaningful in the wide 3-column
   layout; the narrower breakpoints restack the panels entirely. */
function syncPassRowHeight() {
  const viz = document.querySelector('.viz');
  if (state.mode !== 'pass' || !window.matchMedia('(min-width: 1521px)').matches) {
    viz.style.removeProperty('--row-h');
    return;
  }
  const feat = $('#panelFeature');
  const colW = (viz.clientWidth - 32) / 2.72; // epoch's 1fr column width
  const old = feat.getAttribute('style') || '';
  feat.style.cssText =
    `display:flex; visibility:hidden; position:absolute; width:${colW}px;`;
  const h = feat.offsetHeight;
  feat.setAttribute('style', old);
  // the detail panel carries a tab-note line the feature panel doesn't have —
  // grant the row its height so the wordiest hints still fit when pinned
  const note = $('#detailTabNote');
  const extra = note ? Math.ceil(note.getBoundingClientRect().height) + 4 : 0;
  viz.style.setProperty('--row-h', (h + extra) + 'px');
}

/* Run/Pause lifecycle for the pass animation */
function passRunToggle() {
  if (!pass) { startPass(); return; }
  if (pass.timer) {
    clearInterval(pass.timer);
    pass.timer = null;              // freeze: phase chips become clickable
  } else {
    pass.timer = setInterval(advancePass, PASS_STEP_MS); // resume (rolls into
    requestRender();                // the next epoch if paused on 'done')
  }
  updatePassRunUI();
}

function updatePassRunUI() {
  const running = !!(pass && pass.timer);
  $('#passRunIcon').innerHTML = running ? ICON_PAUSE : ICON_PLAY;
  $('#passRunLabel').textContent = running ? 'Pause' : 'Run';
  $('#passControls').classList.toggle('pass-live', !!pass);
  $('#passControls').classList.toggle('pass-paused', !!pass && !running);
  updatePassStepUI();
}

/* the ±1 step pair mirrors epoch mode's ±1 epoch buttons: always visible,
   disabled while the animation runs, relabelled at the pass's two ends */
function updatePassStepUI() {
  const back = $('#passStepBackBtn'), fwd = $('#passStepFwdBtn');
  const usable = !!pass && !pass.timer && !pass.inference;
  const atStart = !!pass && pass.si === 0;
  const atEnd = !!pass && pass.si === pass.steps.length - 1;
  back.textContent = atStart ? 'Prior epoch' : '−1 step';
  fwd.textContent = atEnd ? 'Next epoch' : '+1 step';
  back.disabled = !usable || (atStart && history.length < 2);
  fwd.disabled = !usable;
}

function startPass(paused, atEnd) {
  if (state.playing || !data.length) return;
  clearPass();
  setRunActive(true);
  const p = probeIndex();
  snapshot(); // −1 epoch can rewind this single-sample update too

  // one real training pass, batch size forced to 1 so the update lands immediately
  const pre = weightsSnapshot();
  const tmp = new convnetjs.Trainer(net, {
    learning_rate: state.lr, momentum: state.momentum,
    batch_size: 1, l2_decay: state.l2,
  });
  const v = new convnetjs.Vol(1, 1, 2);
  v.w[0] = data[p][0];
  v.w[1] = data[p][1];
  const stats = tmp.train(v, labels[p]);
  const post = weightsSnapshot();
  markDescentDirty(); // the animated sample already moved the weights

  // capture what actually happened: activations, error signals, weight deltas
  const acts = net.layers.map((L) => Float64Array.from(L.out_act.w));
  const grads = net.layers.map((L) => Float64Array.from(L.out_act.dw));
  const deltaByLi = {};
  let dmax = 1e-9;
  post.forEach((s, k) => {
    deltaByLi[s.li] = s.filters.map((fw, j) => {
      const out = new Float64Array(fw.length);
      for (let i = 0; i < fw.length; i++) {
        out[i] = fw[i] - pre[k].filters[j][i];
        if (Math.abs(out[i]) > dmax) dmax = Math.abs(out[i]);
      }
      return out;
    });
  });

  // the matrix annotation replays the forward/backward math, which ran on the
  // PRE-update weights — the net now holds the post-update ones
  const preByLi = {};
  pre.forEach((s) => { preByLi[s.li] = s; });

  const rows = archLayout();
  const steps = [{ phase: 'sample' }];
  rows.forEach((r, ri) => {
    if (ri === 0) steps.push({ phase: 'forward', row: 0, stage: 'lin' });
    else {
      steps.push({ phase: 'forward', row: ri, stage: 'lin' });
      if (r.side) steps.push({ phase: 'forward', row: ri, stage: 'act' });
    }
  });
  steps.push({ phase: 'loss' });
  // the seed step: the loss turns into the output error signal δz = P − y
  steps.push({ phase: 'backward', row: rows.length - 1, seed: true });
  for (let ri = rows.length - 2; ri >= 0; ri--) steps.push({ phase: 'backward', row: ri });
  steps.push({ phase: 'update' });
  steps.push({ phase: 'done' });

  pass = { p, steps, si: 0, acts, grads, deltaByLi, dmax, preByLi, loss: stats.loss, timer: null };
  if (atEnd) pass.si = steps.length - 1; // arriving from a backwards step
  updatePassUI();
  requestRender();
  if (!paused) pass.timer = setInterval(advancePass, PASS_STEP_MS);
  updatePassRunUI();
}

function advancePass() {
  if (!pass) return;
  if (pass.si < pass.steps.length - 1) {
    pass.si++;
    if (pass.inference) updateInferUI(); else updatePassUI();
    requestRender();
    return;
  }
  if (pass.inference) {
    // inference is one forward trip: arrive at the probabilities and stop
    clearInterval(pass.timer);
    pass.timer = null;
    return;
  }
  // 'done' has been on screen for a beat — the run never stops on its own:
  // fast-forward the rest of the epoch, then follow the same point again
  rollPassEpoch();
}

/* complete the current epoch in the background (every other point trains once,
   with the same batch-size-1 stepping the animated pass used), advance the
   epoch counter and loss history, and start the next epoch's animated pass */
function rollPassEpoch(paused) {
  const tmp = new convnetjs.Trainer(net, {
    learning_rate: state.lr, momentum: state.momentum,
    batch_size: 1, l2_decay: state.l2,
  });
  const v = new convnetjs.Vol(1, 1, 2);
  let sum = pass.loss; // the animated pass was this epoch's first sample
  for (let i = 0; i < data.length; i++) {
    if (i === pass.p) continue;
    v.w[0] = data[i][0]; v.w[1] = data[i][1];
    sum += tmp.train(v, labels[i]).loss;
  }
  recordLoss(sum / data.length, 1);
  startPass(paused);
}

/* manual time travel while paused: ±1 animation step, crossing epoch
   boundaries at either end (the chips are read-only progress indicators) */
function passStepForward() {
  if (!pass || pass.timer || pass.inference) return;
  if (pass.si < pass.steps.length - 1) {
    pass.si++;
    updatePassUI();
    requestRender();
  } else {
    rollPassEpoch(true); // "Next epoch": roll forward, stay paused at 'sample'
  }
}

function passStepBackward() {
  if (!pass || pass.timer || pass.inference) return;
  if (pass.si > 0) {
    pass.si--;
    updatePassUI();
    requestRender();
    return;
  }
  // "Prior epoch": rewind this pass's sample update AND the previous epoch,
  // then replay that epoch's pass — deterministic, so it reconstructs exactly
  // what was on screen — frozen at its final step
  if (history.length < 2) return;
  stepBack();
  stepBack();
  startPass(true, true);
}

function clearPass() {
  if (!pass) return;
  clearInterval(pass.timer);
  pass = null;
  if (state.mode === 'pass') passIdleUI();
  if (state.mode === 'infer') inferIdleUI();
  updatePassRunUI();
  requestRender();
}

function passIdleUI() {
  document.querySelectorAll('#passControls [data-phase]')
    .forEach((b) => b.classList.remove('on', 'done'));
  $('#passCaption').textContent =
    'press Run to follow the traced (gold) PIREP through training — one animated pass per epoch, the rest fast-forwarded — until you pause';
}

function passCaption(st) {
  const rows = archLayout();
  const p = pass.p;
  const cls = labels[p] === DOG ? 'ok (grade ≤ 3)' : 'rough (grade ≥ 4)';
  switch (st.phase) {
    case 'sample':
      return `Following one PIREP: (${fmt(data[p][0])}, ${fmt(data[p][1])}) reported ${cls}. Batch size is forced to 1, so this is exactly one report through the net.`;
    case 'forward': {
      const r = rows[st.row];
      if (st.row === 0) return 'Forward · the sample’s two values enter the input layer.';
      if (st.stage === 'lin') return `Forward · ${r.label}: each neuron computes its weighted sum z = Σ w·x + b from the layer above.`;
      return r.side === 'softmax'
        ? 'Forward · softmax turns the two class scores into probabilities that sum to 1.'
        : `Forward · ${r.side} bends each neuron’s z — this is where the non-linearity enters.`;
    }
    case 'loss': {
      const P = pass.acts[net.layers.length - 1][labels[p]];
      const short = labels[p] === DOG ? 'ok' : 'rough';
      return `Loss · the net says P(${short}) = ${fmt(P)}; the PIREP said “${short}”. Loss = −log(${fmt(P)}) = ${pass.loss.toFixed(3)} — the more right the net is, the smaller this gets.`;
    }
    case 'backward':
      if (st.seed) return 'Backward · first the loss becomes an error signal. With softmax + cross-entropy its slope at the output is simply P − y — forecast minus truth. That diff is injected at the output (violet) and now flows back up.';
      return `Backward · the loss gradient flows back through ${rows[st.row].label} along the same weights — brighter violet = more responsibility for the error.`;
    case 'update':
      return `Update · every weight steps against its gradient (learning rate ${state.lr}): blue connections just got stronger, orange got weaker — thickness shows how much.`;
    default:
      return 'Pass complete — the rest of the epoch now fast-forwards in the background, then the next epoch follows this same point again. Pause any time to explore; undo pass rewinds a whole cycle.';
  }
}

function updatePassUI() {
  const st = pass.steps[pass.si];
  const si = PASS_PHASES.indexOf(st.phase);
  document.querySelectorAll('#passControls [data-phase]').forEach((b) => {
    const ci = PASS_PHASES.indexOf(b.dataset.phase);
    b.classList.toggle('on', b.dataset.phase === st.phase);
    b.classList.toggle('done', st.phase === 'done' || (si >= 0 && ci < si));
  });
  $('#passCaption').textContent = passCaption(st);
  updatePassStepUI();
}

/* ---------------- inference mode: one input point, forward pass only ---------------- */
let inferPt = [1.2, -0.8];

function inferProbs() {
  const v = new convnetjs.Vol(1, 1, 2);
  v.w[0] = inferPt[0]; v.w[1] = inferPt[1];
  return net.forward(v, false).w;
}

function updateInferReadout() {
  const w = inferProbs();
  $('#inferPtVal').textContent = `(${fmt(inferPt[0])}, ${fmt(inferPt[1])})`;
  const ok = w[DOG] > w[CAT];
  $('#inferOutVal').innerHTML =
    `<b style="color:${C.dog}">P(ok) ${fmt(w[DOG])}</b> · ` +
    `<b style="color:${C.cat}">P(rough) ${fmt(w[CAT])}</b> → ` +
    `<b style="color:${ok ? C.dog : C.cat}">${ok ? 'stay on plan' : 'divert'}</b>`;
}

function inferIdleUI() {
  $('#inferCaption').textContent =
    'press Run to grade the waypoint — click anywhere on the map to move it along your route';
}

function inferCaption(st) {
  switch (st.phase) {
    case 'sample':
      return `Waypoint (${fmt(inferPt[0])}, ${fmt(inferPt[1])}) enters the net — inference only: no loss, no gradients, no learning. Just the forward pass a deployed model runs.`;
    case 'forward':
      return passCaption(st); // the forward story is identical to training's
    default: {
      const w = pass.acts[net.layers.length - 1];
      const ok = w[DOG] > w[CAT];
      return `Done · P(ok) = ${fmt(w[DOG])}, P(rough) = ${fmt(w[CAT])} — the net grades this waypoint ${ok ? 'moderate turbulence at worst: stay on plan' : 'moderate chop or worse: divert around it'}. Move the waypoint or press Run to replay.`;
    }
  }
}

function updateInferUI() {
  $('#inferCaption').textContent = inferCaption(pass.steps[pass.si]);
}

/* animate the forward pass on the inference point — no trainer involved */
function startInferRun() {
  clearPass();
  const acts = (inferProbs(), net.layers.map((L) => Float64Array.from(L.out_act.w)));
  const rows = archLayout();
  const steps = [{ phase: 'sample' }];
  rows.forEach((r, ri) => {
    if (ri === 0) steps.push({ phase: 'forward', row: 0, stage: 'lin' });
    else {
      steps.push({ phase: 'forward', row: ri, stage: 'lin' });
      if (r.side) steps.push({ phase: 'forward', row: ri, stage: 'act' });
    }
  });
  steps.push({ phase: 'done' });
  pass = {
    p: 0, steps, si: 0, acts, grads: null, deltaByLi: {}, dmax: 1e-9,
    loss: 0, timer: null, inference: true,
  };
  updateInferUI();
  requestRender();
  pass.timer = setInterval(advancePass, PASS_STEP_MS);
}

/* the feature panel doubles as the inference input plane */
const FEAT_TITLE = {
  epoch: ['Airspace map (feature space)',
    'Every PIREP filed so far, plotted by position (km east, km north). In ATC view, every region shows the ride the net would currently forecast there.'],
  infer: ['Next waypoint',
    'One point on your route ahead. Click anywhere to move it; the forecast readout and the gold trace through every diagram follow instantly.'],
};

function applyFeaturePanelLabels() {
  const inf = state.mode === 'infer';
  const [title, tip] = FEAT_TITLE[inf ? 'infer' : 'epoch'];
  const h2 = document.querySelector('#panelFeature h2');
  h2.textContent = title;
  h2.setAttribute('data-tip', tip);
  document.querySelector('#panelFeature .legend').style.display = inf ? 'none' : '';
  $('#pirepAdders').hidden = inf || selectedPt >= 0;
  $('#featInstr').innerHTML = inf
    ? 'click anywhere to move the <b style="color:#c98500">waypoint</b> — the forecast updates live'
    : 'click a data point to tag it';
}

function passRowEmph(ri, n, ps) {
  if (!ps) return { a: 1, hi: null };
  switch (ps.phase) {
    case 'sample': return { a: ri === 0 ? 1 : 0.3, hi: null };
    case 'forward':
      if (ri < ps.row) return { a: 0.8, hi: null };
      if (ri === ps.row) return { a: 1, hi: 'fwd' };
      return { a: 0.22, hi: null };
    case 'loss': return ri === n - 1 ? { a: 1, hi: null } : { a: 0.5, hi: null };
    case 'backward':
      if (ri > ps.row) return { a: 0.7, hi: 'bdone' };
      if (ri === ps.row) return { a: 1, hi: 'bwd' };
      return { a: 0.35, hi: null };
    default: return { a: 1, hi: null };
  }
}

/* ---------------- layer descriptions ---------------- */
const ACT_EXPL = {
  relu: 'ReLU folds the space: anything negative gets clamped flat to zero, so the grid creases along each neuron’s firing boundary.',
  tanh: 'tanh squashes the space smoothly into the (−1, 1) box — far-away regions get compressed toward the edges.',
  sigmoid: 'Sigmoid squashes the space into the (0, 1) box — large values saturate near the walls.',
};

function describeLayers() {
  const Ls = net.layers, out = [];
  const lastFc = Ls.reduce((a, l, i) => (l.layer_type === 'fc' ? i : a), -1);
  let h = 0;
  for (let i = 0; i < Ls.length; i++) {
    const t = Ls[i].layer_type, d = Ls[i].out_depth;
    let e;
    if (t === 'input') {
      e = { short: 'input', title: 'Input · raw feature space',
            expl: 'The untouched inputs x₁, x₂ — a perfectly regular grid. Every layer after this one reshapes this fabric.' };
    } else if (t === 'fc' && i === lastFc) {
      e = { short: 'out', title: 'Output · class scores (linear)',
            expl: 'A final linear layer projects everything down to one score per class.' };
    } else if (t === 'fc') {
      h++;
      e = { short: 'fc' + sub(h), title: `Hidden layer ${h} · linear`,
            expl: 'Weights rotate, scale and shear the grid; the bias slides it around. Straight lines stay straight.' };
    } else if (t === 'softmax') {
      e = { short: 'softmax', title: 'Output · softmax probabilities',
            expl: 'Scores become probabilities that sum to 1. PIREPs past the diagonal grade as ok air.' };
    } else {
      e = { short: t + sub(h), title: `Hidden layer ${h} · ${t}`,
            expl: ACT_EXPL[t] || '' };
    }
    e.dim = d; e.type = t;
    out.push(e);
  }
  return out;
}

function axisLabels() {
  const t = net.layers[state.lix].layer_type;
  if (t === 'input') return { x: 'x₁', y: 'x₂' };
  if (t === 'softmax') return { x: 'P(rough)', y: 'P(ok)' };
  return { x: 'neuron ' + state.d0, y: 'neuron ' + state.d1 };
}

/* ---------------- forward-pass field ---------------- */
function computeField(S) {
  const step = S / (GRID_COLS - 1);
  const nL = net.layers.length;
  const meshN = Math.ceil(GRID_COLS / GRID_STEP);
  const ss = S / 10.4; // px per data unit
  const f = {
    S, ss, meshN, cell: step,
    layers: Array.from({ length: nL }, () => ({ xs: [], ys: [] })),
    sel: { xs: [], ys: [] },
    meshLab: [],
    cells: state.overlay ? [] : null,
    pts: Array.from({ length: nL }, () => []),
    selPts: [],
  };
  const v = new convnetjs.Vol(1, 1, 2);
  const cstep = state.overlay ? 1 : GRID_STEP;
  for (let cx = 0; cx < GRID_COLS; cx += cstep) {
    for (let cy = 0; cy < GRID_COLS; cy += cstep) {
      const px = cx * step, py = cy * step;
      v.w[0] = (px - S / 2) / ss;
      v.w[1] = (py - S / 2) / ss;
      const a = net.forward(v, false);
      const isDog = a.w[DOG] > a.w[CAT];
      if (f.cells) f.cells.push(px, py, isDog ? 1 : 0);
      if (cx % GRID_STEP === 0 && cy % GRID_STEP === 0) {
        for (let li = 0; li < nL; li++) {
          const w = net.layers[li].out_act.w;
          f.layers[li].xs.push(w[0]);
          f.layers[li].ys.push(w.length > 1 ? w[1] : 0);
        }
        const ws = net.layers[state.lix].out_act.w;
        f.sel.xs.push(ws[state.d0]);
        f.sel.ys.push(ws[state.d1]);
        f.meshLab.push(isDog);
      }
    }
  }
  const infer = state.mode === 'infer';
  for (let i = 0; i < data.length; i++) {
    v.w[0] = data[i][0]; v.w[1] = data[i][1];
    net.forward(v, false);
    for (let li = 0; li < nL; li++) {
      const w = net.layers[li].out_act.w;
      f.pts[li].push({ x: w[0], y: w.length > 1 ? w[1] : 0, lab: labels[i], tag: !infer && tagged[i] });
    }
    const ws = net.layers[state.lix].out_act.w;
    f.selPts.push({ x: ws[state.d0], y: ws[state.d1], lab: labels[i], tag: !infer && tagged[i] });
  }
  if (infer) {
    // the inference point rides through every diagram as the (only) gold trace
    v.w[0] = inferPt[0]; v.w[1] = inferPt[1];
    const a = net.forward(v, false);
    const lab = a.w[DOG] > a.w[CAT] ? DOG : CAT;
    for (let li = 0; li < nL; li++) {
      const w = net.layers[li].out_act.w;
      f.pts[li].push({ x: w[0], y: w.length > 1 ? w[1] : 0, lab, tag: true });
    }
    const ws = net.layers[state.lix].out_act.w;
    f.selPts.push({ x: ws[state.d0], y: ws[state.d1], lab, tag: true });
  }
  return f;
}

/* ---------------- canvas helpers ---------------- */
function fitCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function makeScale(min, max, a, b) {
  const dv = max - min;
  if (!(dv > 1e-9)) return () => (a + b) / 2; // dead/constant dimension
  return (v) => a + (v - min) / dv * (b - a);
}

function niceTicks(min, max, count = 5) {
  const span = max - min;
  if (!(span > 1e-9)) return [min];
  const step0 = span / (count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + span * 1e-6; v += step) out.push(v);
  return out;
}

function tickLabel(v) {
  return Math.abs(v) < 1e-9 ? '0' : parseFloat(v.toPrecision(3)).toString();
}

function drawDot(ctx, x, y, r, p) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = p.tag ? C.tag : (p.lab === DOG ? C.dog : C.cat);
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = p.tag ? C.ink : C.surface;
  ctx.stroke();
}

/* Draw one layer's warped grid + data points into a square canvas. */
function drawSpace(ctx, S, xs, ys, n, opt) {
  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] < minx) minx = xs[i];
    if (xs[i] > maxx) maxx = xs[i];
    if (ys[i] < miny) miny = ys[i];
    if (ys[i] > maxy) maxy = ys[i];
  }
  const pad = Math.max(8, S * 0.06);
  const m = opt.ticks
    ? { l: 46, r: 18, t: 26, b: 46 }
    : { l: pad, r: pad, t: pad, b: pad };
  const mx = makeScale(minx, maxx, m.l, S - m.r);
  const my = makeScale(miny, maxy, m.t, S - m.b);
  const X = new Float64Array(xs.length), Y = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) { X[i] = mx(xs[i]); Y[i] = my(ys[i]); }

  if (opt.lab) {
    const cs = (S - m.l - m.r) / (n - 1) + 1.5;
    for (let k = 0; k < X.length; k++) {
      ctx.fillStyle = opt.lab[k] ? C.dogWash : C.catWash;
      ctx.fillRect(X[k] - cs / 2, Y[k] - cs / 2, cs, cs);
    }
  }

  ctx.strokeStyle = opt.meshColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const k = i * n + j;
      if (j + 1 < n) { ctx.moveTo(X[k], Y[k]); ctx.lineTo(X[k + 1], Y[k + 1]); }
      if (i + 1 < n) { ctx.moveTo(X[k], Y[k]); ctx.lineTo(X[k + n], Y[k + n]); }
    }
  }
  ctx.stroke();

  if (opt.pts) for (const p of opt.pts) drawDot(ctx, mx(p.x), my(p.y), opt.r, p);

  if (opt.axes && opt.ticks) {
    const y0 = S - m.b + 8, x0 = m.l - 8;
    ctx.strokeStyle = C.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(m.l, y0); ctx.lineTo(S - m.r, y0);
    ctx.moveTo(x0, m.t); ctx.lineTo(x0, S - m.b);
    ctx.stroke();
    ctx.font = '9px ui-monospace, Menlo, monospace';
    ctx.fillStyle = C.muted;
    ctx.textAlign = 'center';
    for (const v of niceTicks(minx, maxx)) {
      const px = mx(v);
      ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y0 + 4); ctx.stroke();
      ctx.fillText(tickLabel(v), px, y0 + 15);
    }
    ctx.textAlign = 'right';
    for (const v of niceTicks(miny, maxy)) {
      const py = my(v);
      ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x0 - 4, py); ctx.stroke();
      ctx.fillText(tickLabel(v), x0 - 7, py + 3);
    }
    ctx.font = MONO;
    ctx.textAlign = 'center';
    ctx.fillText(opt.axes.x + ' →', (m.l + S - m.r) / 2, S - 8);
    ctx.save();
    ctx.translate(12, (m.t + S - m.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(opt.axes.y + ' →', 0, 0);
    ctx.restore();
    ctx.textAlign = 'left';
  } else if (opt.axes) {
    ctx.fillStyle = C.muted;
    ctx.font = MONO;
    ctx.fillText(opt.axes.x + ' →', 12, S - 10);
    ctx.save();
    ctx.translate(16, S - 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(opt.axes.y + ' →', 0, 0);
    ctx.restore();
  }
}

function subsample(arr, n, s) {
  const m = Math.ceil(n / s), out = new Array(m * m);
  let c = 0;
  for (let i = 0; i < n; i += s) for (let j = 0; j < n; j += s) out[c++] = arr[i * n + j];
  return out;
}

/* ---------------- render ---------------- */
function render() {
  const featFit = fitCanvas($('#featCanvas'));
  field = computeField(featFit.w);
  drawFeature(featFit, field);
  if (!$('#layerCanvas').hidden) drawSelected(fitCanvas($('#layerCanvas')), field);
  drawArch();
  drawMinis(field);
  drawSpark();
  drawDescent();
  if (timelineShown()) drawTimeline();
  $('#lossVal').textContent = state.loss == null ? '—' : state.loss.toFixed(4);
  $('#epochVal').textContent = state.epoch.toLocaleString();
}

function drawFeature({ ctx, w: S }, f) {
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, S, S);

  if (f.cells) {
    const cs = f.cell;
    for (let i = 0; i < f.cells.length; i += 3) {
      ctx.fillStyle = f.cells[i + 2] ? C.dogWash : C.catWash;
      ctx.fillRect(f.cells[i] - cs / 2, f.cells[i + 1] - cs / 2, cs + 0.5, cs + 0.5);
    }
  }

  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let u = -5; u <= 5; u++) {
    const p = S / 2 + u * f.ss;
    ctx.moveTo(p, 0); ctx.lineTo(p, S);
    ctx.moveTo(0, p); ctx.lineTo(S, p);
  }
  ctx.stroke();

  ctx.strokeStyle = C.axis;
  ctx.beginPath();
  ctx.moveTo(0, S / 2); ctx.lineTo(S, S / 2);
  ctx.moveTo(S / 2, 0); ctx.lineTo(S / 2, S);
  ctx.stroke();

  // value ticks on the zero axes, one per data unit
  ctx.font = '9px ui-monospace, Menlo, monospace';
  ctx.fillStyle = C.muted;
  ctx.strokeStyle = C.axis;
  for (let u = -5; u <= 5; u++) {
    if (u === 0) continue;
    const p = S / 2 + u * f.ss;
    if (p < 12 || p > S - 12) continue;
    ctx.beginPath();
    ctx.moveTo(p, S / 2 - 3); ctx.lineTo(p, S / 2 + 3);
    ctx.moveTo(S / 2 - 3, p); ctx.lineTo(S / 2 + 3, p);
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(String(u), p, S / 2 + 15);
    ctx.textAlign = 'right';
    ctx.fillText(String(u), S / 2 - 7, p + 3);
  }
  ctx.textAlign = 'right';
  ctx.fillText('0', S / 2 - 7, S / 2 + 15);
  ctx.textAlign = 'left';

  ctx.fillStyle = C.muted;
  ctx.font = MONO;
  ctx.fillText('x₁ →', S - 38, S / 2 - 8);
  ctx.fillText('x₂', S / 2 + 8, 16);

  if (state.mode === 'infer') {
    // one gold input point with a crosshair — no training data on this plane
    const px = S / 2 + inferPt[0] * f.ss, py = S / 2 + inferPt[1] * f.ss;
    ctx.strokeStyle = 'rgba(201, 133, 0, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(px, 0); ctx.lineTo(px, S);
    ctx.moveTo(0, py); ctx.lineTo(S, py);
    ctx.stroke();
    ctx.setLineDash([]);
    drawDot(ctx, px, py, 6.5, { lab: DOG, tag: true });
    return;
  }

  for (let i = 0; i < data.length; i++) {
    if (reveal && !reveal.shown.has(i)) continue; // this report isn't in yet
    drawDot(ctx, S / 2 + data[i][0] * f.ss, S / 2 + data[i][1] * f.ss, 5,
      { lab: labels[i], tag: tagged[i] });
  }

  // freshly-filed PIREPs wear their tail number + grade for a moment
  ctx.font = '600 10px ui-monospace, Menlo, monospace';
  for (const p of pendingPireps) {
    const px = S / 2 + p.x * f.ss, py = S / 2 + p.y * f.ss;
    ctx.fillStyle = p.lab === DOG ? C.dog : C.cat;
    ctx.fillText(p.text, Math.min(Math.max(px + 9, 4), S - 82), py < 20 ? py + 19 : py - 9);
  }
  ctx.font = MONO;

  // ring the point whose action bar is open
  if (selectedPt >= 0 && selectedPt < data.length) {
    ctx.beginPath();
    ctx.arc(S / 2 + data[selectedPt][0] * f.ss, S / 2 + data[selectedPt][1] * f.ss, 9.5, 0, Math.PI * 2);
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }

  // ring the point being traced by the node/edge detail view
  if (state.view.kind !== 'layer') {
    const p = probeIndex();
    if (p >= 0) {
      ctx.beginPath();
      ctx.setLineDash([4, 3]);
      ctx.arc(S / 2 + data[p][0] * f.ss, S / 2 + data[p][1] * f.ss, 12, 0, Math.PI * 2);
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function drawSelected({ ctx, w: S }, f) {
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, S, S);
  if (state.view.kind === 'node') { drawNodeDetail(ctx, S); return; }
  if (state.view.kind === 'edge') { drawEdgeDetail(ctx, S); return; }
  drawSpace(ctx, S, f.sel.xs, f.sel.ys, f.meshN, {
    lab: state.overlay ? f.meshLab : null,
    meshColor: C.mesh,
    pts: f.selPts, r: 5,
    axes: axisLabels(),
    ticks: true,
  });
}

/* ---------------- node / edge detail views ---------------- */
const fmt = (v) => (Object.is(v, -0) ? 0 : v).toFixed(2);

function probeIndex() {
  if (state.mode === 'infer') return 0; // virtual: probePt() supplies the point
  const t = tagged.findIndex(Boolean);
  return t >= 0 ? t : (data.length ? 0 : -1);
}

/* the coordinates behind a probe index — the inference point in infer mode */
function probePt(p) {
  return state.mode === 'infer' ? inferPt : data[p];
}

function outActOf(row) {
  return net.layers[row.idxs[row.idxs.length - 1]].out_act.w;
}

function forwardProbe(p) {
  const pt = probePt(p);
  const v = new convnetjs.Vol(1, 1, 2);
  v.w[0] = pt[0];
  v.w[1] = pt[1];
  net.forward(v, false);
}

function drawProbeCaption(ctx, S, p) {
  ctx.font = MONO;
  ctx.textAlign = 'left';
  if (state.mode === 'infer') {
    ctx.fillStyle = C.tag;
    ctx.beginPath();
    ctx.arc(16, 17, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = C.muted;
    ctx.fillText(`waypoint (${fmt(inferPt[0])}, ${fmt(inferPt[1])})`, 26, 21);
    return;
  }
  ctx.fillStyle = labels[p] === DOG ? C.dog : C.cat;
  ctx.beginPath();
  ctx.arc(16, 17, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.fillText(
    `tracing PIREP (${fmt(data[p][0])}, ${fmt(data[p][1])}) · ${labels[p] === DOG ? 'ok' : 'rough'}`, 26, 21);
}

function centerText(ctx, S, msg) {
  ctx.font = MONO;
  ctx.fillStyle = C.muted;
  ctx.textAlign = 'center';
  ctx.fillText(msg, S / 2, S / 2);
  ctx.textAlign = 'left';
}

function edgeStyle(ctx, w, wmax) {
  const s = Math.min(Math.abs(w) / (wmax || 1), 1);
  ctx.strokeStyle = w >= 0 ? C.dog : C.cat;
  ctx.globalAlpha = 0.35 + 0.6 * s;
  ctx.lineWidth = 1 + 3 * s;
}

function smallNode(ctx, x, y, r, text, label) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = '#222220';
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.stroke();
  ctx.fillStyle = C.ink;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  if (text != null) ctx.fillText(text, x, y + 3.5);
  if (label) {
    ctx.fillStyle = C.muted;
    ctx.fillText(label, x, y - r - 6);
  }
  ctx.textAlign = 'left';
}

const ACT_FN = {
  relu: { f: (t) => Math.max(0, t), rng: [-0.4, 3] },
  tanh: { f: Math.tanh, rng: [-1.3, 1.3] },
  sigmoid: { f: (t) => 1 / (1 + Math.exp(-t)), rng: [-0.15, 1.15] },
};

function drawActBox(ctx, x, y, wBox, hBox, act, z) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, wBox, hBox, 8);
  else ctx.rect(x, y, wBox, hBox);
  ctx.fillStyle = '#222220';
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = C.inkSec;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(act, x + wBox / 2, y + 13);
  ctx.textAlign = 'left';
  const spec = ACT_FN[act];
  if (!spec) return;
  const gx = x + 10, gy = y + 19, gw = wBox - 20, gh = hBox - 27;
  const dom = [-3, 3];
  const mx = (t) => gx + (t - dom[0]) / (dom[1] - dom[0]) * gw;
  const my = (v) => gy + gh - (v - spec.rng[0]) / (spec.rng[1] - spec.rng[0]) * gh;
  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.beginPath();
  ctx.moveTo(mx(0), gy); ctx.lineTo(mx(0), gy + gh);
  ctx.moveTo(gx, my(0)); ctx.lineTo(gx + gw, my(0));
  ctx.stroke();
  ctx.strokeStyle = C.inkSec;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let k = 0; k <= 40; k++) {
    const t = dom[0] + k / 40 * (dom[1] - dom[0]);
    const px = mx(t), py = my(spec.f(t));
    if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  const zc = Math.max(dom[0], Math.min(dom[1], z));
  ctx.beginPath();
  ctx.arc(mx(zc), my(spec.f(zc)), 3.5, 0, Math.PI * 2);
  ctx.fillStyle = C.tag;
  ctx.fill();
}

function drawCalcLines(ctx, S, weights, inputs, bias, z) {
  const terms = weights.map((w, i) => `${fmt(w)}·${fmt(inputs[i])}`);
  const signed = (v) => `${v < 0 ? '−' : '+'} ${fmt(Math.abs(v))}`;
  const tokens = ['z =', ...terms.map((t, i) => (i ? '+ ' : '') + `(${t})`), signed(bias), `= ${fmt(z)}`];
  ctx.font = '10.5px ui-monospace, Menlo, monospace';
  ctx.fillStyle = C.muted;
  const maxW = S - 28;
  const lines = [];
  let cur = '';
  for (const tok of tokens) {
    const next = cur ? cur + ' ' + tok : tok;
    if (ctx.measureText(next).width > maxW && cur) { lines.push(cur); cur = tok; }
    else cur = next;
  }
  if (cur) lines.push(cur);
  const shown = lines.slice(0, 3);
  if (lines.length > 3) shown[2] += ' …';
  shown.forEach((ln, i) => ctx.fillText(ln, 14, S - 14 - (shown.length - 1 - i) * 15));
}

function drawNodeDetail(ctx, S) {
  const rows = archLayout();
  const { rowIdx, j } = state.view;
  if (rowIdx >= rows.length || j >= rows[rowIdx].count) { centerText(ctx, S, 'selection no longer exists'); return; }
  const p = probeIndex();
  if (p < 0) { centerText(ctx, S, 'add a data point to trace'); return; }
  forwardProbe(p);
  drawProbeCaption(ctx, S, p);

  const row = rows[rowIdx];
  const prevRow = rowIdx > 0 ? rows[rowIdx - 1] : null;
  const nextRow = rowIdx < rows.length - 1 ? rows[rowIdx + 1] : null;
  const isOutput = row.label === 'output';
  const a = outActOf(row)[j];
  const yTop = 88, yBot = S - 100;
  const mid = (yTop + yBot) / 2;
  const yPos = (k, n) => n === 1 ? mid : yTop + k * (yBot - yTop) / (n - 1);

  if (!prevRow) {
    /* --- input feature node --- */
    const xN = Math.round(S * 0.32), xOut = S - 60;
    smallNode(ctx, xN, mid, 24, fmt(a), `x${sub(j + 1)}  (raw input)`);
    if (nextRow) {
      const outW = net.layers[nextRow.fcIdx].filters.map((f) => f.w[j]);
      const wmax = Math.max(...outW.map(Math.abs), 0.1);
      for (let k = 0; k < nextRow.count; k++) {
        const yk = yPos(k, nextRow.count);
        edgeStyle(ctx, outW[k], wmax);
        ctx.beginPath();
        ctx.moveTo(xN + 26, mid);
        ctx.lineTo(xOut - 15, yk);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = outW[k] >= 0 ? C.dog : C.cat;
        ctx.font = '10px ui-monospace, Menlo, monospace';
        ctx.fillText(`×${fmt(outW[k])}`, xN + 26 + (xOut - xN - 41) * 0.55, mid + (yk - mid) * 0.55 - 5);
      }
      for (let k = 0; k < nextRow.count; k++) {
        smallNode(ctx, xOut, yPos(k, nextRow.count), 13, 'n' + k);
      }
      ctx.fillStyle = C.muted;
      ctx.font = MONO;
      ctx.textAlign = 'right';
      ctx.fillText(nextRow.label + ' →', S - 14, 52);
      ctx.textAlign = 'left';
    }
    return;
  }

  /* --- fc neuron (hidden or output) --- */
  const inputs = Array.from(outActOf(prevRow));
  const fcL = net.layers[row.fcIdx];
  const weights = Array.from(fcL.filters[j].w);
  const bias = fcL.biases.w[j];
  const z = fcL.out_act.w[j];
  const xIn = 58, xNeu = Math.round(S * 0.42), xAct = Math.round(S * 0.66), xOut = S - 52;
  const wmaxIn = Math.max(...weights.map(Math.abs), 0.1);

  // incoming edges + weight labels
  for (let i = 0; i < inputs.length; i++) {
    const yi = yPos(i, inputs.length);
    edgeStyle(ctx, weights[i], wmaxIn);
    ctx.beginPath();
    ctx.moveTo(xIn + 14, yi);
    ctx.lineTo(xNeu - 26, mid);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = weights[i] >= 0 ? C.dog : C.cat;
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.fillText(fmt(weights[i]), xIn + 14 + (xNeu - 40 - xIn) * 0.42, yi + (mid - yi) * 0.42 - 5);
  }
  // incoming value nodes
  for (let i = 0; i < inputs.length; i++) {
    smallNode(ctx, xIn, yPos(i, inputs.length), 14, fmt(inputs[i]),
      prevRow.label === 'input' ? `x${sub(i + 1)}` : 'n' + i);
  }
  ctx.fillStyle = C.muted;
  ctx.font = MONO;
  ctx.fillText('from ' + prevRow.label, 14, 52);

  // the neuron: weighted sum + bias
  smallNode(ctx, xNeu, mid, 25, fmt(z), 'Σ w·x + b');
  ctx.fillStyle = C.muted;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`b = ${fmt(bias)}`, xNeu, mid + 39);
  ctx.textAlign = 'left';

  // activation stage
  const actName = row.side || 'linear';
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(xNeu + 26, mid);
  ctx.lineTo(xAct - 46, mid);
  ctx.stroke();
  drawActBox(ctx, xAct - 44, mid - 33, 88, 66, actName, z);
  ctx.fillStyle = C.ink;
  ctx.font = '11px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(isOutput ? `P = ${fmt(a)}` : `a = ${fmt(a)}`, xAct, mid + 48);
  ctx.textAlign = 'left';

  // outgoing fan-out
  if (nextRow) {
    const outW = net.layers[nextRow.fcIdx].filters.map((f) => f.w[j]);
    const wmaxOut = Math.max(...outW.map(Math.abs), 0.1);
    for (let k = 0; k < nextRow.count; k++) {
      const yk = yPos(k, nextRow.count);
      edgeStyle(ctx, outW[k], wmaxOut);
      ctx.beginPath();
      ctx.moveTo(xAct + 44, mid);
      ctx.lineTo(xOut - 14, yk);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    for (let k = 0; k < nextRow.count; k++) {
      smallNode(ctx, xOut, yPos(k, nextRow.count), 12, 'n' + k);
    }
    ctx.fillStyle = C.muted;
    ctx.font = MONO;
    ctx.textAlign = 'right';
    ctx.fillText('to ' + nextRow.label + ' →', S - 14, 52);
    ctx.textAlign = 'left';
  } else {
    ctx.fillStyle = j === DOG ? C.dog : C.cat;
    ctx.font = '12px ui-monospace, Menlo, monospace';
    ctx.fillText(`P(${j === CAT ? 'rough' : 'ok'}) = ${fmt(a)}`, xAct + 54, mid + 4);
  }

  drawCalcLines(ctx, S, weights, inputs, bias, z);
}

function drawEdgeDetail(ctx, S) {
  const rows = archLayout();
  const { rowIdx, i, j } = state.view;
  if (rowIdx >= rows.length || rowIdx < 1) { centerText(ctx, S, 'selection no longer exists'); return; }
  const from = rows[rowIdx - 1], to = rows[rowIdx];
  if (i >= from.count || j >= to.count) { centerText(ctx, S, 'selection no longer exists'); return; }
  const w = net.layers[to.fcIdx].filters[j].w[i];
  const p = probeIndex();
  let x = null, z = null;
  if (p >= 0) {
    forwardProbe(p);
    x = outActOf(from)[i];
    z = net.layers[to.fcIdx].out_act.w[j];
    drawProbeCaption(ctx, S, p);
  }
  const mid = S / 2 - 14;
  const xA = Math.round(S * 0.22), xB = Math.round(S * 0.78);

  ctx.strokeStyle = w >= 0 ? C.dog : C.cat;
  ctx.lineWidth = 1.5 + Math.min(Math.abs(w) / 2, 1) * 6;
  ctx.beginPath();
  ctx.moveTo(xA + 22, mid);
  ctx.lineTo(xB - 22, mid);
  ctx.stroke();

  smallNode(ctx, xA, mid, 21, x == null ? 'n' + i : fmt(x),
    `${from.label} · n${i}`);
  smallNode(ctx, xB, mid, 21, z == null ? 'n' + j : fmt(z),
    `${to.label} · n${j}`);
  ctx.fillStyle = C.muted;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('sends', xA, mid + 34);
  ctx.fillText('Σ w·x + b', xB, mid + 34);

  ctx.fillStyle = C.ink;
  ctx.font = '600 17px ui-monospace, Menlo, monospace';
  ctx.fillText(`w = ${w.toFixed(3)}`, S / 2, mid - 26);
  ctx.font = '11px ui-monospace, Menlo, monospace';
  ctx.fillStyle = w >= 0 ? C.dog : C.cat;
  ctx.fillText(w >= 0 ? 'positive: excites the receiver' : 'negative: inhibits the receiver', S / 2, mid + 58);
  if (x != null) {
    ctx.fillStyle = C.inkSec;
    ctx.fillText(`contribution for the traced point: ${fmt(x)} × ${w.toFixed(3)} = ${fmt(x * w)}`, S / 2, mid + 84);
    ctx.fillStyle = C.muted;
    ctx.fillText(`one of ${from.count} inputs summed into the receiver's z = ${fmt(z)}`, S / 2, mid + 104);
  }
  ctx.textAlign = 'left';
}

function drawMinis(f) {
  for (let li = 0; li < minis.length; li++) {
    const { ctx, w: S } = fitCanvas(minis[li].cv);
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, S, S);
    const xs = subsample(f.layers[li].xs, f.meshN, 2);
    const ys = subsample(f.layers[li].ys, f.meshN, 2);
    drawSpace(ctx, S, xs, ys, Math.ceil(f.meshN / 2), {
      meshColor: C.meshMini,
      pts: f.pts[li], r: 2.5,
    });
  }
}

/* Vertical network diagram: input at top, output at bottom.
   Edge thickness/opacity follow |weight|; blue positive, orange negative. */
function archLayout() {
  const Ls = net.layers;
  const lastFc = Ls.reduce((a, l, i) => (l.layer_type === 'fc' ? i : a), -1);
  const rows = [{ label: 'input', count: 2, idxs: [0] }];
  let h = 0;
  for (let i = 1; i < Ls.length; i++) {
    if (Ls[i].layer_type !== 'fc') continue;
    const idxs = [i];
    let side = '';
    const next = Ls[i + 1];
    if (next && ['relu', 'tanh', 'sigmoid', 'softmax'].includes(next.layer_type)) {
      idxs.push(i + 1);
      side = next.layer_type;
    }
    const label = i === lastFc ? 'output' : 'hidden ' + (++h);
    rows.push({ label, count: Ls[i].out_depth, idxs, fcIdx: i, side });
  }
  return rows;
}

function drawArch() {
  const { ctx, w: W, h: H } = fitCanvas($('#archCanvas'));
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, W, H);
  const rows = archLayout();
  const ps = pass ? pass.steps[pass.si] : null;
  const probe = ps ? pass.p : (state.view.kind !== 'layer' ? probeIndex() : -1);
  if (probe >= 0 && !ps) forwardProbe(probe);

  // single-pass mode: a live pass reserves a right-hand strip for the matrix
  // view of the step being animated (skipped when the canvas is too narrow)
  const showMat = !!ps && state.mode === 'pass' && !pass.inference && W >= 640;
  const annW = showMat ? Math.min(360, Math.max(300, Math.round(W * 0.36))) : 0;
  const netW = W - annW;
  archMatX = showMat ? netW : Infinity;

  const padTop = probe >= 0 ? 44 : 34;
  // a training pass reserves room under the output row for the loss badge
  // and the δz = P − y injection arrows of the seed step
  const isTrainPass = !!ps && !pass.inference;
  const padBot = isTrainPass ? 80 : 42;
  const actGap = 30, actBox = 17;
  const maxCount = Math.max(...rows.map((r) => r.count));
  const spacing = Math.min(52, (netW - 150) / Math.max(maxCount - 1, 1));
  const extra = rows.reduce((s, r) => s + (r.side ? actGap : 0), 0);
  const gapBetween = rows.length > 1 ? (H - padTop - padBot - extra) / (rows.length - 1) : 0;
  const nodeY = [], actY = [];
  let yCur = padTop;
  rows.forEach((r, i) => {
    nodeY[i] = yCur;
    actY[i] = r.side ? yCur + actGap : null;
    yCur = (r.side ? yCur + actGap : yCur) + gapBetween;
  });
  const nodeX = (row, j) => netW / 2 + (j - (row.count - 1) / 2) * spacing;
  const outY = (i) => actY[i] != null ? actY[i] : nodeY[i]; // where a row's output leaves

  archRows = rows.map((r, i) => ({ y: nodeY[i], idxs: r.idxs }));
  archNodes = [];
  archEdges = [];
  archActs = [];
  rows.forEach((r, ri) => {
    for (let j = 0; j < r.count; j++) archNodes.push({ x: nodeX(r, j), y: nodeY[ri], rowIdx: ri, j });
    if (r.side) {
      archActs.push({
        y: actY[ri], rowIdx: ri, actIdx: r.idxs[1],
        xMin: nodeX(r, 0) - actBox, xMax: nodeX(r, r.count - 1) + actBox,
      });
    }
  });

  // global weight scale so edges stay comparable across layers
  let wmax = 0.1;
  for (const r of rows) {
    if (r.fcIdx == null) continue;
    for (const f of net.layers[r.fcIdx].filters) {
      for (const w of f.w) { const a = Math.abs(w); if (a > wmax) wmax = a; }
    }
  }

  for (let ri = 1; ri < rows.length; ri++) {
    const prev = rows[ri - 1], row = rows[ri];
    const filters = net.layers[row.fcIdx].filters;
    let mul = 1, useDelta = false;
    if (ps) {
      if (ps.phase === 'sample') mul = 0.15;
      else if (ps.phase === 'forward') mul = ri < ps.row ? 0.5 : ri === ps.row ? 2.2 : 0.1;
      else if (ps.phase === 'loss') mul = 0.35;
      else if (ps.phase === 'backward') mul = 0.12;
      else if (ps.phase === 'update') useDelta = true;
    }
    for (let j = 0; j < row.count; j++) {
      for (let i = 0; i < prev.count; i++) {
        const w = filters[j].w[i];
        const s = Math.min(Math.abs(w) / wmax, 1);
        const x1 = nodeX(prev, i), y1 = outY(ri - 1);
        const x2 = nodeX(row, j), y2 = nodeY[ri];
        archEdges.push({ x1, y1, x2, y2, rowIdx: ri, i, j });
        if (useDelta) {
          const d = pass.deltaByLi[row.fcIdx][j][i];
          const sD = Math.min(Math.abs(d) / pass.dmax, 1);
          ctx.strokeStyle = d >= 0 ? C.dog : C.cat;
          ctx.globalAlpha = 0.2 + 0.7 * sD;
          ctx.lineWidth = 0.6 + 3.5 * sD;
        } else {
          ctx.strokeStyle = w >= 0 ? C.dog : C.cat;
          ctx.globalAlpha = Math.min((0.12 + 0.6 * s) * mul, 0.95);
          ctx.lineWidth = 0.5 + 3 * s;
        }
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;

  // backward phase: the gradient currently travelling down these edges, in violet
  if (ps && ps.phase === 'backward' && ps.row + 1 < rows.length) {
    ctx.strokeStyle = C.spark;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (const ed of archEdges) {
      if (ed.rowIdx !== ps.row + 1) continue;
      ctx.moveTo(ed.x1, ed.y1);
      ctx.lineTo(ed.x2, ed.y2);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // selected connection drawn on top, in the marker gold
  if (state.view.kind === 'edge') {
    const sel = archEdges.find((ed) =>
      ed.rowIdx === state.view.rowIdx && ed.i === state.view.i && ed.j === state.view.j);
    if (sel) {
      ctx.strokeStyle = C.tag;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sel.x1, sel.y1);
      ctx.lineTo(sel.x2, sel.y2);
      ctx.stroke();
    }
  }

  const fmt1 = (v) => { const s = v.toFixed(1); return s === '-0.0' ? '0.0' : s; };
  const nodeR = probe >= 0 ? 11 : 8;
  const VAL_FONT = '8.5px ui-monospace, Menlo, monospace';
  ctx.font = MONO;
  for (let ri = 0; ri < rows.length; ri++) {
    const row = rows[ri], yN = nodeY[ri];
    const em = passRowEmph(ri, rows.length, ps);
    ctx.globalAlpha = em.a;
    const fcSel = state.view.kind === 'layer' &&
      (row.fcIdx != null ? state.lix === row.fcIdx : state.lix === 0);
    const actSel = state.view.kind === 'layer' && row.side && state.lix === row.idxs[1];

    // stubs from each Σ node down into its activation box
    if (row.side) {
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let j = 0; j < row.count; j++) {
        ctx.moveTo(nodeX(row, j), yN + nodeR);
        ctx.lineTo(nodeX(row, j), actY[ri] - actBox / 2);
      }
      ctx.stroke();
    }

    for (let j = 0; j < row.count; j++) {
      const isSelNode = state.view.kind === 'node' &&
        state.view.rowIdx === ri && state.view.j === j;
      const x = nodeX(row, j);
      ctx.beginPath();
      ctx.arc(x, yN, nodeR, 0, Math.PI * 2);
      ctx.fillStyle = '#222220';
      ctx.fill();
      let strokeC = isSelNode ? C.tag : fcSel ? C.ink : 'rgba(255,255,255,0.28)';
      let strokeW = fcSel || isSelNode ? 1.8 : 1;
      if (em.hi === 'fwd' && (ri === 0 || ps.stage === 'lin')) { strokeC = C.tag; strokeW = 2; }
      if (em.hi === 'bdone') strokeC = 'rgba(144,133,233,0.5)';
      if (em.hi === 'bwd') {
        // violet wash scaled by this neuron's actual share of the error
        const gsrc = pass.grads[row.fcIdx != null ? row.fcIdx : 0];
        let gmax = 1e-9;
        for (let q = 0; q < row.count; q++) if (Math.abs(gsrc[q]) > gmax) gmax = Math.abs(gsrc[q]);
        ctx.fillStyle = `rgba(144,133,233,${0.12 + 0.55 * Math.abs(gsrc[j]) / gmax})`;
        ctx.fill();
        strokeC = C.spark;
        strokeW = 2;
      }
      ctx.lineWidth = strokeW;
      ctx.strokeStyle = strokeC;
      ctx.stroke();
      if (isSelNode) {
        ctx.beginPath();
        ctx.arc(x, yN, nodeR + 4, 0, Math.PI * 2);
        ctx.strokeStyle = C.tag;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (probe >= 0) {
        const val = ps
          ? pass.acts[row.fcIdx != null ? row.fcIdx : 0][j]
          : (row.fcIdx != null ? net.layers[row.fcIdx].out_act.w[j] : probePt(probe)[j]);
        ctx.fillStyle = C.ink;
        ctx.font = VAL_FONT;
        ctx.textAlign = 'center';
        ctx.fillText(fmt1(val), x, yN + 3);
        ctx.textAlign = 'left';
        ctx.font = MONO;
      }
    }

    // the activation band: one box per neuron, clickable as a layer
    if (row.side) {
      const yA = actY[ri];
      for (let j = 0; j < row.count; j++) {
        const x = nodeX(row, j);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x - actBox / 2, yA - actBox / 2, actBox, actBox, 4);
        else ctx.rect(x - actBox / 2, yA - actBox / 2, actBox, actBox);
        ctx.fillStyle = '#222220';
        ctx.fill();
        let aStroke = actSel ? C.ink : 'rgba(255,255,255,0.28)';
        let aWidth = actSel ? 1.8 : 1;
        if (ps && ps.phase === 'forward' && ps.stage === 'act' && ri === ps.row) {
          aStroke = C.tag; aWidth = 2;
        }
        if (ps && ps.phase === 'loss' && row.label === 'output') {
          const isTrue = j === labels[pass.p];
          aStroke = isTrue ? C.tag : 'rgba(255,255,255,0.45)';
          aWidth = isTrue ? 2.4 : 1.2;
        }
        if (em.hi === 'bwd') { aStroke = C.spark; aWidth = 1.6; }
        ctx.lineWidth = aWidth;
        ctx.strokeStyle = aStroke;
        ctx.stroke();
        if (probe >= 0) {
          const av = ps ? pass.acts[row.idxs[row.idxs.length - 1]][j] : outActOf(row)[j];
          ctx.fillStyle = C.inkSec;
          ctx.font = VAL_FONT;
          ctx.textAlign = 'center';
          ctx.fillText(fmt1(av), x, yA + 3);
          ctx.textAlign = 'left';
          ctx.font = MONO;
        } else {
          drawActGlyph(ctx, x, yA, row.side);
        }
      }
      if (probe >= 0 && row.label === 'output') {
        ctx.font = '10px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        for (let j = 0; j < row.count; j++) {
          ctx.fillStyle = j === DOG ? C.dog : C.cat;
          ctx.fillText(`P(${j === CAT ? 'rough' : 'ok'})`, nodeX(row, j), yA + actBox / 2 + 14);
        }
        ctx.textAlign = 'left';
        ctx.font = MONO;
      }
      ctx.fillStyle = actSel ? C.ink : C.muted;
      ctx.textAlign = 'right';
      ctx.fillText(row.side, netW - 10, yA + 4);
      ctx.textAlign = 'left';
    }

    ctx.fillStyle = fcSel ? C.ink : C.muted;
    ctx.fillText(row.label, 10, yN + 4);
  }
  ctx.globalAlpha = 1;

  // the loss lives below the output row: name it there, and on the seed step
  // show its slope P − y being injected back into the output nodes
  if (isTrainPass && (ps.phase === 'loss' || ps.seed)) {
    const out = rows[rows.length - 1];
    const yBadge = H - 16;
    ctx.textAlign = 'center';
    ctx.font = '600 11px ui-monospace, Menlo, monospace';
    if (ps.seed) {
      ctx.fillStyle = C.spark;
      ctx.fillText('δz = P − y · the diff heads back up', netW / 2, yBadge);
      const yTip = (actY[rows.length - 1] ?? nodeY[rows.length - 1]) + 26;
      ctx.strokeStyle = C.spark;
      ctx.lineWidth = 1.4;
      for (let j = 0; j < out.count; j++) {
        const x = nodeX(out, j);
        ctx.beginPath();
        ctx.moveTo(x, yBadge - 12);
        ctx.lineTo(x, yTip + 4);
        ctx.stroke();
        ctx.fillStyle = C.spark;
        ctx.beginPath();
        ctx.moveTo(x, yTip - 1);
        ctx.lineTo(x - 3.5, yTip + 5);
        ctx.lineTo(x + 3.5, yTip + 5);
        ctx.closePath();
        ctx.fill();
      }
    } else {
      const P = pass.acts[net.layers.length - 1][labels[pass.p]];
      ctx.fillStyle = C.tag;
      ctx.fillText(`loss L = −log(${fmt(P)}) = ${pass.loss.toFixed(3)}`, netW / 2, yBadge);
    }
    ctx.textAlign = 'left';
  }

  if (showMat) drawPassMatrixPanel(ctx, netW, W, H, rows);
  if (probe >= 0) drawProbeCaption(ctx, W, probe);
}

function drawActGlyph(ctx, x, y, act) {
  if (act === 'softmax') {
    ctx.fillStyle = C.inkSec;
    ctx.font = '9px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('P', x, y + 3);
    ctx.textAlign = 'left';
    ctx.font = MONO;
    return;
  }
  ctx.strokeStyle = C.inkSec;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  if (act === 'relu') {
    ctx.moveTo(x - 5, y + 3.5);
    ctx.lineTo(x, y + 3.5);
    ctx.lineTo(x + 5, y - 3.5);
  } else {
    ctx.moveTo(x - 5, y + 3.5);
    ctx.bezierCurveTo(x - 1, y + 3.5, x + 1, y - 3.5, x + 5, y - 3.5);
  }
  ctx.stroke();
}

/* ---------------- pass-mode matrix annotation ----------------
   A right-hand strip of the network canvas that shows the actual matrix
   arithmetic behind the animation step on screen: W·a + b = z on the way
   down, δa = Wᵀ·δz on the way back, W + ΔW = W′ at the update. Everything
   is computed from the pass's captured pre-update snapshot so the printed
   numbers really do multiply out. Wide layers are truncated to their
   top-left block with ⋯/⋮/⋱ ellipses. */
const MAT_FONT = '9.5px ui-monospace, Menlo, Consolas, monospace';
const MAT_OP_FONT = '600 11px ui-monospace, Menlo, Consolas, monospace';
const MAT_LBL_FONT = '600 10px ui-monospace, Menlo, Consolas, monospace';
const MAT_ROW_H = 14;
const MAT_MAX_ROWS = 5;    // larger → 4 shown + a ⋮ row
const MAT_MAX_COLS = 3;    // wider  → 2 shown + a ⋯ column

const fmtCell = (v) => {
  if (v === 0) return '0.00';
  const a = Math.abs(v);
  if (a < 0.001) return v.toExponential(0); // tiny deltas/grads: '8e-4', not '-0.0000'
  if (a < 0.01) return v.toFixed(4);
  if (a < 0.1) return v.toFixed(3);
  return v.toFixed(2);
};

function matTrunc(get, nRows, nCols) {
  const rShown = nRows <= MAT_MAX_ROWS ? nRows : MAT_MAX_ROWS - 1;
  const cShown = nCols <= MAT_MAX_COLS ? nCols : MAT_MAX_COLS - 1;
  const cells = [];
  for (let r = 0; r < rShown; r++) {
    const row = [];
    for (let c = 0; c < cShown; c++) row.push(fmtCell(get(r, c)));
    if (cShown < nCols) row.push('⋯');
    cells.push(row);
  }
  if (rShown < nRows) {
    cells.push(cells[0].map((_, c) =>
      cShown < nCols && c === cells[0].length - 1 ? '⋱' : '⋮'));
  }
  return { cells, dims: `${nRows}×${nCols}` };
}

function measureMat(ctx, M) {
  ctx.font = MAT_FONT;
  const cols = M.cells[0].length;
  M.colW = [];
  for (let c = 0; c < cols; c++) {
    let w = 0;
    for (const row of M.cells) w = Math.max(w, ctx.measureText(row[c]).width);
    M.colW.push(w);
  }
  M.w = M.colW.reduce((a, b) => a + b, 0) + (cols - 1) * 8 + 14;
  M.h = M.cells.length * MAT_ROW_H;
}

function drawMat(ctx, x, yMid, M) {
  const top = yMid - M.h / 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + 4, top); ctx.lineTo(x, top);
  ctx.lineTo(x, top + M.h); ctx.lineTo(x + 4, top + M.h);
  ctx.moveTo(x + M.w - 4, top); ctx.lineTo(x + M.w, top);
  ctx.lineTo(x + M.w, top + M.h); ctx.lineTo(x + M.w - 4, top + M.h);
  ctx.stroke();
  ctx.font = MAT_FONT;
  ctx.textAlign = 'right';
  for (let r = 0; r < M.cells.length; r++) {
    const cy = top + r * MAT_ROW_H + MAT_ROW_H / 2 + 3;
    let cx = x + 7;
    for (let c = 0; c < M.cells[0].length; c++) {
      cx += M.colW[c];
      ctx.fillStyle = M.hiRow === r ? C.tag : (M.color || C.inkSec);
      ctx.fillText(M.cells[r][c], cx, cy);
      cx += 8;
    }
  }
  ctx.textAlign = 'center';
  ctx.font = MAT_LBL_FONT;
  if (M.label) {
    ctx.fillStyle = M.labelColor || C.ink;
    ctx.fillText(M.label, x + M.w / 2, top - 7);
  }
  ctx.fillStyle = C.muted;
  ctx.fillText(M.dims, x + M.w / 2, top + M.h + 13);
  ctx.textAlign = 'left';
}

/* lay out matrices and operators left→right, wrapping before an operator when
   a line overflows; each line is centered in the strip */
function drawMatExpr(ctx, x0, x1, yMid, items) {
  const GAP = 9, LABEL_H = 15, DIMS_H = 17, LINE_GAP = 14;
  for (const it of items) {
    if (it.kind === 'op') {
      ctx.font = MAT_OP_FONT;
      it.w = ctx.measureText(it.text).width;
      it.h = 0;
    } else measureMat(ctx, it);
  }
  // units = optional leading operator + the matrix it applies to
  const units = [];
  let pend = [];
  for (const it of items) {
    pend.push(it);
    if (it.kind !== 'op') { units.push(pend); pend = []; }
  }
  if (pend.length) units.push(pend);
  const unitW = (u) => u.reduce((s, it) => s + it.w, 0) + (u.length - 1) * GAP;
  const maxW = x1 - x0;
  const lines = [];
  let line = [], w = 0;
  for (const u of units) {
    const need = (line.length ? GAP : 0) + unitW(u);
    if (line.length && w + need > maxW) { lines.push(line); line = []; w = 0; }
    line.push(u);
    w += (line.length > 1 ? GAP : 0) + unitW(u);
  }
  if (line.length) lines.push(line);
  const lineH = (ln) =>
    Math.max(...ln.flat().map((it) => it.h || 0), MAT_ROW_H) + LABEL_H + DIMS_H;
  const totH = lines.reduce((s, ln) => s + lineH(ln), 0) + (lines.length - 1) * LINE_GAP;
  let y = yMid - totH / 2;
  for (const ln of lines) {
    const lh = lineH(ln);
    const mid = y + LABEL_H + (lh - LABEL_H - DIMS_H) / 2;
    const lw = ln.reduce((s, u) => s + unitW(u), 0) + (ln.length - 1) * GAP;
    let x = x0 + Math.max(0, (maxW - lw) / 2);
    for (const u of ln) {
      for (const it of u) {
        if (it.kind === 'op') {
          ctx.font = MAT_OP_FONT;
          ctx.fillStyle = C.muted;
          ctx.fillText(it.text, x, mid + 4);
        } else drawMat(ctx, x, mid, it);
        x += it.w + GAP;
      }
    }
    y += lh + LINE_GAP;
  }
}

/* the numbers behind the current animation step */
function passMatrixSpec(st, rows) {
  const A = pass.acts;
  const preF = (li) => {
    const s = pass.preByLi && pass.preByLi[li];
    return s ? s.filters : net.layers[li].filters.map((f) => f.w);
  };
  const preB = (li) => {
    const s = pass.preByLi && pass.preByLi[li];
    return s ? s.biases : net.layers[li].biases.w;
  };
  const vec = (arr, label, opts) =>
    ({ kind: 'mat', label, ...matTrunc((r) => arr[r], arr.length, 1), ...opts });
  const mat = (get, nR, nC, label, opts) =>
    ({ kind: 'mat', label, ...matTrunc(get, nR, nC), ...opts });
  const op = (t) => ({ kind: 'op', text: t });
  const RES = { color: C.ink, labelColor: C.tag };

  switch (st.phase) {
    case 'sample':
      return {
        title: 'sample',
        sub: 'the PIREP as a 2×1 column vector',
        items: [vec(A[0], 'x')],
        footer: 'x₁ = km east · x₂ = km north',
      };
    case 'forward': {
      const row = rows[st.row];
      if (st.row === 0) {
        return {
          title: 'forward · input',
          sub: 'x enters the input layer',
          items: [vec(A[0], 'x')],
        };
      }
      if (st.stage === 'lin') {
        const prev = rows[st.row - 1];
        const aIn = A[prev.idxs[prev.idxs.length - 1]];
        const F = preF(row.fcIdx);
        const inLab = prev.label === 'input' ? 'x' : 'a';
        return {
          title: `forward · ${row.label}`,
          sub: `z = W·${inLab} + b — one row of W per neuron`,
          items: [
            mat((r, c) => F[r][c], row.count, prev.count, 'W'),
            op('·'), vec(aIn, inLab),
            op('+'), vec(Array.from(preB(row.fcIdx)), 'b'),
            op('='), vec(A[row.fcIdx], 'z', RES),
          ],
        };
      }
      const z = A[row.fcIdx], a = A[row.idxs[1]];
      if (row.side === 'softmax') {
        return {
          title: 'forward · softmax',
          sub: 'P = exp(z) / Σ exp(z)',
          items: [vec(z, 'z'), op('→'), vec(a, 'P', RES)],
          footer: 'two probabilities that sum to 1',
        };
      }
      return {
        title: `forward · ${row.side}`,
        sub: `a = ${row.side}(z), applied element-wise`,
        items: [vec(z, 'z'), op('→'), vec(a, 'a', RES)],
      };
    }
    case 'loss': {
      const P = A[net.layers.length - 1];
      const y = labels[pass.p];
      const onehot = Array.from(P, (_, i) => (i === y ? 1 : 0));
      return {
        title: 'loss · cross-entropy',
        sub: 'L = −log P[y] — compare P with the truth y',
        items: [vec(Array.from(P), 'P', { hiRow: y }), vec(onehot, 'y', { hiRow: y })],
        footer: `L = −log(${fmt(P[y])}) = ${pass.loss.toFixed(3)}`,
        graph: { p: P[y], loss: pass.loss },
      };
    }
    case 'backward': {
      if (st.seed) {
        const P = A[net.layers.length - 1];
        const y = labels[pass.p];
        const onehot = Array.from(P, (_, i) => (i === y ? 1 : 0));
        const out = rows[rows.length - 1];
        return {
          title: 'backward · loss gradient',
          sub: 'δz = P − y — the diff IS the error signal',
          items: [
            vec(Array.from(P), 'P'), op('−'), vec(onehot, 'y', { hiRow: y }),
            op('='), vec(pass.grads[out.fcIdx], 'δz', RES),
          ],
          footer: 'softmax + cross-entropy make ∂L/∂z exactly this diff; it now flows back through every layer',
        };
      }
      const from = rows[st.row], to = rows[st.row + 1];
      const F = preF(to.fcIdx);
      const dz = pass.grads[to.fcIdx];
      const da = pass.grads[from.idxs[from.idxs.length - 1]];
      const outLab = st.row === 0 ? 'δx' : 'δa';
      return {
        title: `backward · ${from.label}`,
        sub: `${outLab} = Wᵀ·δz — the same weights, transposed`,
        items: [
          mat((r, c) => F[c][r], from.count, to.count, 'Wᵀ'),
          op('·'), vec(dz, 'δz'),
          op('='), vec(da, outLab, RES),
        ],
        footer: `δz is ${to.label}'s error signal; W is ${to.label}'s weight matrix`,
      };
    }
    case 'update':
    case 'done': {
      const row = rows[1];
      const li = row.fcIdx;
      const F = preF(li);
      const d = pass.deltaByLi[li];
      const cur = net.layers[li].filters;
      const nC = rows[0].count;
      return {
        title: st.phase === 'done' ? 'pass complete · weights updated' : `update · ${row.label}`,
        sub: 'W ← W + ΔW, where ΔW ≈ −η·∂L/∂W',
        items: [
          mat((r, c) => F[r][c], row.count, nC, 'W'),
          op('+'), mat((r, c) => d[r][c], row.count, nC, 'ΔW'),
          op('='), mat((r, c) => cur[r].w[c], row.count, nC, 'W′', RES),
        ],
        footer: `η = ${state.lr} · shown for ${row.label} — every layer updates the same way`,
      };
    }
    default:
      return null;
  }
}

function drawPassMatrixPanel(ctx, x0, x1, H, rows) {
  const spec = passMatrixSpec(pass.steps[pass.si], rows);
  if (!spec) return;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0 + 0.5, 14);
  ctx.lineTo(x0 + 0.5, H - 14);
  ctx.stroke();

  const px0 = x0 + 18, px1 = x1 - 14;
  ctx.textAlign = 'left';
  ctx.font = MAT_LBL_FONT;
  ctx.fillStyle = C.tag;
  ctx.fillText(spec.title.toUpperCase(), px0, 26);
  ctx.fillStyle = C.muted;
  ctx.font = MAT_FONT;
  ctx.fillText(spec.sub, px0, 42);

  // footer, word-wrapped to at most two lines above the bottom edge
  let footH = 0;
  if (spec.footer) {
    const lines = [];
    let cur = '';
    for (const word of spec.footer.split(' ')) {
      const next = cur ? cur + ' ' + word : word;
      if (ctx.measureText(next).width > px1 - px0 && cur) { lines.push(cur); cur = word; }
      else cur = next;
    }
    if (cur) lines.push(cur);
    const shown = lines.slice(0, 2);
    if (lines.length > 2) shown[1] += ' …';
    ctx.fillStyle = C.muted;
    shown.forEach((ln, i) => ctx.fillText(ln, px0, H - 14 - (shown.length - 1 - i) * 14));
    footH = shown.length * 14 + 8;
  }

  // the loss step also plots L = −log p, with the net's actual P[y] marked
  const graphH = spec.graph ? Math.min(150, Math.max(110, Math.round(H * 0.3))) : 0;
  drawMatExpr(ctx, px0, px1, 46 + (H - 46 - footH - graphH - 20) / 2, spec.items);
  if (spec.graph) drawLossCurve(ctx, px0, px1, H - footH - graphH - 10, graphH, spec.graph);
}

function drawLossCurve(ctx, x0, x1, yTop, hBox, g) {
  const gx = x0 + 26, gw = x1 - x0 - 40, gy = yTop + 10, gh = hBox - 40;
  const LMAX = 3;
  const mx = (p) => gx + p * gw;
  const my = (L) => gy + gh - Math.min(L, LMAX) / LMAX * gh;
  ctx.strokeStyle = C.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(gx, gy); ctx.lineTo(gx, gy + gh); ctx.lineTo(gx + gw, gy + gh);
  ctx.stroke();
  ctx.strokeStyle = C.inkSec;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  const pMin = Math.exp(-LMAX); // where −log p leaves the plotted range
  for (let k = 0; k <= 60; k++) {
    const p = pMin + k / 60 * (1 - pMin);
    const px = mx(p), py = my(-Math.log(p));
    if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  // dashed guides to the axes, then the net's current (P[y], L)
  const dx = mx(Math.max(g.p, pMin)), dy = my(g.loss);
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(gx, dy); ctx.lineTo(dx, dy); ctx.lineTo(dx, gy + gh);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = C.tag;
  ctx.beginPath();
  ctx.arc(dx, dy, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = MAT_FONT;
  ctx.fillStyle = C.muted;
  ctx.fillText('L = −log p', gx + 6, gy + 4);
  ctx.textAlign = 'center';
  ctx.fillText('0', gx, gy + gh + 12);
  ctx.fillText('1', gx + gw, gy + gh + 12);
  ctx.fillText('P[y] → confident & correct', gx + gw / 2, gy + gh + 12);
  ctx.textAlign = 'left';
  ctx.fillStyle = C.tag;
  ctx.fillText(`(${fmt(g.p)}, ${g.loss.toFixed(2)})`, Math.min(dx + 8, x1 - 60), dy - 6);
}

function segDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len2)) : 0;
  const cx = x1 + t * dx, cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

function onArchClick(e) {
  const px = e.offsetX, py = e.offsetY;
  if (px >= archMatX) return; // the matrix panel is display-only
  for (const n of archNodes) {
    if ((px - n.x) ** 2 + (py - n.y) ** 2 <= 13 * 13) { selectNode(n.rowIdx, n.j); return; }
  }
  for (const ab of archActs) {
    if (Math.abs(py - ab.y) <= 11 && px >= ab.xMin - 6 && px <= ab.xMax + 6) {
      selectLayer(ab.actIdx);
      return;
    }
  }
  let bestE = null, bestED = 6;
  for (const ed of archEdges) {
    const d = segDist(px, py, ed.x1, ed.y1, ed.x2, ed.y2);
    if (d < bestED) { bestED = d; bestE = ed; }
  }
  if (bestE) { selectEdge(bestE.rowIdx, bestE.i, bestE.j); return; }
  let best = -1, bestD = 26;
  for (let i = 0; i < archRows.length; i++) {
    const d = Math.abs(py - archRows[i].y);
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best >= 0) selectLayer(archRows[best].idxs[0]);
}

function drawSpark() {
  const { ctx, w, h } = fitCanvas($('#sparkCanvas'));
  ctx.clearRect(0, 0, w, h);
  const hist = state.lossHist;
  if (hist.length < 2) return;
  let min = Infinity, max = -Infinity;
  for (const v of hist) { if (v < min) min = v; if (v > max) max = v; }
  const dv = Math.max(max - min, 1e-9);
  ctx.strokeStyle = C.spark;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const x = i / (hist.length - 1) * (w - 2) + 1;
    const y = h - 3 - (hist[i] - min) / dv * (h - 6);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

/* ---------------- gradient descent landscape ----------------
   The loss surface lives in one dimension per weight, so it can't be drawn —
   but a 2-D slice through the current weights can be evaluated for real:
   d1 points along the run's own recent travel (so the picture is oriented
   along the descent), d2 is a fixed random perpendicular, and every grid
   cell is the true training loss at w0 + (a·d1 + b·d2)·r. The survey runs
   in small time-boxed chunks so training and rendering never stall, and the
   live weights are restored before every yield. */
const DESCENT_N = 29;              // grid cells per axis (odd: center sampled)
const DESCENT_COOLDOWN_MS = 700;   // min gap between two surveys
let descent = null;                // finished survey {w0,d1,d2,r,grid,N,dims,min,max,minA,minB}
let descentDirty = true;
let descentBusy = false;
let descentLast = 0;
let descentRandA = null, descentRandB = null; // sticky directions → stable view

function markDescentDirty() { descentDirty = true; }

function flatWeightCount() {
  let n = 0;
  for (const L of net.layers) {
    if (L.filters) n += L.filters.length * L.filters[0].w.length + L.biases.w.length;
  }
  return n;
}

function flatWeights() {
  const out = new Float64Array(flatWeightCount());
  let k = 0;
  for (const L of net.layers) {
    if (!L.filters) continue;
    for (const f of L.filters) for (const w of f.w) out[k++] = w;
    for (const b of L.biases.w) out[k++] = b;
  }
  return out;
}

function setFlatWeights(arr) {
  let k = 0;
  for (const L of net.layers) {
    if (!L.filters) continue;
    for (const f of L.filters) for (let i = 0; i < f.w.length; i++) f.w[i] = arr[k++];
    for (let i = 0; i < L.biases.w.length; i++) L.biases.w[i] = arr[k++];
  }
}

/* flatten a history snapshot in the same order as flatWeights() */
function flatFromSnap(snap) {
  const out = [];
  for (const s of snap) {
    for (const fw of s.filters) out.push(...fw);
    out.push(...s.biases);
  }
  return out;
}

function descentDirections() {
  const w0 = flatWeights();
  const n = w0.length;
  const freshRand = () => Float64Array.from({ length: n }, () => convnetjs.randn(0, 1));
  if (!descentRandA || descentRandA.length !== n) descentRandA = freshRand();
  if (!descentRandB || descentRandB.length !== n) descentRandB = freshRand();

  // d1: where the run has been travelling (oldest snapshot → now); random until
  // there is a run
  let d1 = null, travel = 0;
  if (history.length) {
    const old = flatFromSnap(history[0].weights);
    if (old.length === n) {
      d1 = new Float64Array(n);
      let s = 0;
      for (let i = 0; i < n; i++) { d1[i] = w0[i] - old[i]; s += d1[i] * d1[i]; }
      travel = Math.sqrt(s);
      if (travel < 1e-6) d1 = null;
    }
  }
  if (!d1) d1 = Float64Array.from(descentRandA);
  const norm = (v) => {
    let s = 0;
    for (const x of v) s += x * x;
    s = Math.sqrt(s) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= s;
  };
  norm(d1);
  // d2: the sticky random direction, made perpendicular to d1
  const d2 = Float64Array.from(descentRandB);
  let dot = 0;
  for (let i = 0; i < n; i++) dot += d2[i] * d1[i];
  for (let i = 0; i < n; i++) d2[i] -= dot * d1[i];
  norm(d2);
  // slice half-width: wide enough to hold the recent path, never microscopic
  const r = Math.min(Math.max(travel * 1.25, 0.8), 8);
  return { w0, d1, d2, r };
}

function descentLossHere(v, step) {
  let sum = 0, cnt = 0;
  for (let k = 0; k < data.length; k += step) {
    v.w[0] = data[k][0];
    v.w[1] = data[k][1];
    const P = net.forward(v, false).w[labels[k]];
    sum += -Math.log(Math.max(P, 1e-12));
    cnt++;
  }
  return sum / cnt;
}

function descentCompute() {
  descentBusy = true;
  descentDirty = false;
  const { w0, d1, d2, r } = descentDirections();
  const N = DESCENT_N;
  const surf = { w0, d1, d2, r, N, dims: w0.length, grid: new Float64Array(N * N) };
  const step = data.length > 120 ? 2 : 1;
  const tmp = new Float64Array(w0.length);
  const v = new convnetjs.Vol(1, 1, 2);
  let row = 0;
  const chunk = () => {
    // the net was rebuilt or the data emptied mid-survey: abandon, try again
    if (!data.length || flatWeightCount() !== surf.dims) {
      descentBusy = false;
      descentDirty = true;
      return;
    }
    const live = flatWeights();
    const t0 = performance.now();
    while (row < N && performance.now() - t0 < 12) {
      const b = (row / (N - 1)) * 2 - 1;
      for (let cx = 0; cx < N; cx++) {
        const a = (cx / (N - 1)) * 2 - 1;
        for (let i = 0; i < surf.dims; i++) tmp[i] = w0[i] + (a * d1[i] + b * d2[i]) * r;
        setFlatWeights(tmp);
        surf.grid[row * N + cx] = descentLossHere(v, step);
      }
      row++;
    }
    setFlatWeights(live);
    if (row < N) { setTimeout(chunk, 0); return; }
    surf.min = Infinity;
    surf.max = -Infinity;
    let minAt = 0;
    surf.grid.forEach((L, i) => {
      if (L < surf.min) { surf.min = L; minAt = i; }
      if (L > surf.max) surf.max = L;
    });
    surf.minA = ((minAt % N) / (N - 1)) * 2 - 1;
    surf.minB = (Math.floor(minAt / N) / (N - 1)) * 2 - 1;
    descent = surf;
    descentBusy = false;
    descentLast = performance.now();
    $('#descentNote').textContent =
      `${surf.dims} weights · slice half-width ${r.toFixed(2)} · loss ${surf.min.toFixed(2)}–${surf.max.toFixed(2)} · dark = low`;
    requestRender();
  };
  setTimeout(chunk, 0);
}

/* start a survey when one is due; called every tick */
function descentTick() {
  if ($('#panelDescent').hidden || !data.length) return;
  if (descentBusy || !descentDirty) return;
  if (performance.now() - descentLast < DESCENT_COOLDOWN_MS) return;
  descentCompute();
}

/* the detail panel hosts the landscape as its second (map) and third (3-d) tabs */
let detailTab = 'layer'; // 'layer' | 'descent' | 'descent3d'

const DETAIL_TABS = {
  layer: {
    btn: '#detTabLayer', cv: '#layerCanvas', hint: '#layerExpl',
    note: '(select the layer to view from the selection in the panel below)',
  },
  descent: {
    btn: '#detTabDescent', cv: '#detailDescentCanvas', hint: '#detailDescentHint',
    note: '(see below for a description of this landscape)',
    title: 'Gradient descent · loss landscape',
  },
  descent3d: {
    btn: '#detTabDescent3d', cv: '#detailDescent3dCanvas', hint: '#detailDescent3dHint',
    note: '(drag to rotate · see below for a description of this landscape)',
    title: 'Gradient descent · loss surface',
  },
};

function detailApplyTab() {
  for (const [k, t] of Object.entries(DETAIL_TABS)) {
    const on = detailTab === k;
    $(t.btn).setAttribute('aria-selected', String(on));
    $(t.cv).hidden = !on;
    $(t.hint).hidden = !on;
  }
  const t = DETAIL_TABS[detailTab];
  $('#detailTabNote').textContent = t.note;
  if (t.title) {
    $('#layerTitle').textContent = t.title;
    $('#cycleBtn').hidden = true;
  } else {
    updateSelectionUI(); // restores the selection's title, hint and ⟳ button
  }
  requestRender();
}

/* inspecting a component is a statement of intent: bring its view forward */
function detailShowLayerTab() {
  if (detailTab === 'layer') return;
  detailTab = 'layer';
  detailApplyTab();
}

function drawDescent() {
  if (!$('#panelDescent').hidden) drawDescentInto($('#descentCanvas'));
  if (!$('#detailDescentCanvas').hidden) drawDescentInto($('#detailDescentCanvas'));
  if (!$('#detailDescent3dCanvas').hidden) drawDescent3DInto($('#detailDescent3dCanvas'));
}

function drawDescentInto(cv) {
  const { ctx, w: S } = fitCanvas(cv);
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, S, S);
  if (descent && descent.dims !== flatWeightCount()) descent = null;
  if (!descent) { centerText(ctx, S, 'mapping the loss landscape…'); return; }
  const { grid, N, w0, d1, d2, r, min, max } = descent;
  const pad = 22;
  const span = S - pad * 2;
  const cs = span / N;
  const px = (a) => pad + (a + 1) / 2 * span; // a,b ∈ [−1,1] → canvas
  const dv = Math.max(max - min, 1e-9);

  // posterized heatmap: bright violet ridge, near-black valley — the contour
  // bands read as a topo map without marching squares
  for (let gy = 0; gy < N; gy++) {
    for (let gx = 0; gx < N; gx++) {
      const t = (grid[gy * N + gx] - min) / dv;
      const tq = Math.round(t * 11) / 11;
      ctx.fillStyle = `rgba(144, 133, 233, ${(0.04 + 0.72 * tq).toFixed(3)})`;
      ctx.fillRect(pad + gx * cs, pad + gy * cs, cs + 0.5, cs + 0.5);
    }
  }

  // the lowest surveyed point: a white ring to settle into
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(px(descent.minA), px(descent.minB), 7, 0, Math.PI * 2);
  ctx.stroke();

  // recent trajectory + the live weights, projected onto the slice
  const proj = (flat) => {
    let a = 0, b = 0;
    for (let i = 0; i < flat.length; i++) {
      const d = flat[i] - w0[i];
      a += d * d1[i];
      b += d * d2[i];
    }
    return [a / r, b / r];
  };
  const clip = (t) => Math.max(-1.04, Math.min(1.04, t));
  ctx.strokeStyle = C.spark;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  let started = false;
  const from = Math.max(0, history.length - 60);
  for (let h = from; h < history.length; h++) {
    const snap = flatFromSnap(history[h].weights);
    if (snap.length !== w0.length) continue;
    const [a, b] = proj(snap);
    const x = px(clip(a)), y = px(clip(b));
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  const [na, nb] = proj(flatWeights());
  const nx = px(clip(na)), ny = px(clip(nb));
  if (started) ctx.lineTo(nx, ny);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = C.tag;
  ctx.beginPath();
  ctx.arc(nx, ny, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#141413';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.font = MONO;
  ctx.fillStyle = C.muted;
  ctx.fillText('d₁ · along the run’s travel →', pad, S - 7);
  ctx.save();
  ctx.translate(13, S - pad);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('d₂ · random ⊥ →', 0, 0);
  ctx.restore();
}

/* ---- 3-d surface view of the same survey: loss vertical over (d1, d2) ---- */
let d3yaw = -0.65;   // azimuth, radians — drag horizontally
let d3elev = 1.05;   // elevation above the horizon — drag vertically

/* bilinear height of the surveyed surface at slice coords (a, b) ∈ [−1, 1] */
function descentHeightAt(a, b) {
  const { grid, N } = descent;
  const gx = (Math.max(-1, Math.min(1, a)) + 1) / 2 * (N - 1);
  const gy = (Math.max(-1, Math.min(1, b)) + 1) / 2 * (N - 1);
  const x0 = Math.floor(gx), y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, N - 1), y1 = Math.min(y0 + 1, N - 1);
  const fx = gx - x0, fy = gy - y0;
  return grid[y0 * N + x0] * (1 - fx) * (1 - fy) + grid[y0 * N + x1] * fx * (1 - fy)
    + grid[y1 * N + x0] * (1 - fx) * fy + grid[y1 * N + x1] * fx * fy;
}

function drawDescent3DInto(cv) {
  const { ctx, w: S } = fitCanvas(cv);
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, S, S);
  if (descent && descent.dims !== flatWeightCount()) descent = null;
  if (!descent) { centerText(ctx, S, 'mapping the loss landscape…'); return; }
  const { grid, N, w0, d1, d2, r, min, max } = descent;
  const dv = Math.max(max - min, 1e-9);
  const ZH = 0.85; // world height of the loss axis
  const cosY = Math.cos(d3yaw), sinY = Math.sin(d3yaw);
  const cosE = Math.cos(d3elev), sinE = Math.sin(d3elev);
  const scale = (S - 90) / (2 * Math.SQRT2);
  const cx = S / 2, cy = S / 2 + ZH * cosE * scale * 0.5;
  // orthographic: yaw about the loss axis, then tilt by elevation
  const proj = (X, Y, Z) => {
    const rx = X * cosY - Y * sinY;
    const ry = X * sinY + Y * cosY;
    return [cx + rx * scale, cy + (ry * sinE - Z * cosE) * scale, ry * cosE + Z * sinE];
  };
  const hOf = (i) => (grid[i] - min) / dv * ZH;

  // project every vertex once
  const VX = new Float64Array(N * N), VY = new Float64Array(N * N), VD = new Float64Array(N * N);
  for (let gy = 0; gy < N; gy++) {
    for (let gx = 0; gx < N; gx++) {
      const i = gy * N + gx;
      const [px2, py2, dep] = proj((gx / (N - 1)) * 2 - 1, (gy / (N - 1)) * 2 - 1, hOf(i));
      VX[i] = px2; VY[i] = py2; VD[i] = dep;
    }
  }

  // painter's algorithm: sort cells far → near by mean corner depth
  const cells = [];
  for (let gy = 0; gy < N - 1; gy++) {
    for (let gx = 0; gx < N - 1; gx++) {
      const i0 = gy * N + gx, i1 = i0 + 1, i2 = i0 + N, i3 = i2 + 1;
      cells.push([VD[i0] + VD[i1] + VD[i2] + VD[i3], i0, i1, i3, i2]);
    }
  }
  cells.sort((p, q) => p[0] - q[0]);

  const ramp = (t) => {
    const m = (lo, hi) => Math.round(lo + (hi - lo) * t);
    return `rgb(${m(30, 144)}, ${m(30, 133)}, ${m(29, 233)})`;
  };
  ctx.lineJoin = 'round';
  for (const [, i0, i1, i3, i2] of cells) {
    const t = ((grid[i0] + grid[i1] + grid[i2] + grid[i3]) / 4 - min) / dv;
    ctx.fillStyle = ramp(0.06 + 0.9 * t);
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(VX[i0], VY[i0]);
    ctx.lineTo(VX[i1], VY[i1]);
    ctx.lineTo(VX[i3], VY[i3]);
    ctx.lineTo(VX[i2], VY[i2]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // white ring on the lowest surveyed vertex
  let minAt = 0;
  for (let i = 1; i < grid.length; i++) if (grid[i] < grid[minAt]) minAt = i;
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(VX[minAt], VY[minAt], 6, 0, Math.PI * 2);
  ctx.stroke();

  // trajectory + live weights, draped on the surface (bilinear heights)
  const slice = (flat) => {
    let a = 0, b = 0;
    for (let i = 0; i < flat.length; i++) {
      const d = flat[i] - w0[i];
      a += d * d1[i];
      b += d * d2[i];
    }
    return [Math.max(-1, Math.min(1, a / r)), Math.max(-1, Math.min(1, b / r))];
  };
  const drape = (a, b) => proj(a, b, (descentHeightAt(a, b) - min) / dv * ZH + 0.015);
  ctx.strokeStyle = C.spark;
  ctx.lineWidth = 1.6;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  let started = false;
  for (let h = Math.max(0, history.length - 60); h < history.length; h++) {
    const snap = flatFromSnap(history[h].weights);
    if (snap.length !== w0.length) continue;
    const [px2, py2] = drape(...slice(snap));
    if (!started) { ctx.moveTo(px2, py2); started = true; }
    else ctx.lineTo(px2, py2);
  }
  const [na, nb] = slice(flatWeights());
  const [nx, ny] = drape(na, nb);
  if (started) ctx.lineTo(nx, ny);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = C.tag;
  ctx.beginPath();
  ctx.arc(nx, ny, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#141413';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // axis labels pinned to the base square's projected edges
  ctx.font = MONO;
  ctx.fillStyle = C.muted;
  ctx.textAlign = 'center';
  const [ax, ay] = proj(1.28, 0, 0);
  ctx.fillText('d₁ →', ax, ay);
  const [bx, by] = proj(0, 1.28, 0);
  ctx.fillText('d₂ →', bx, by);
  ctx.textAlign = 'left';
  ctx.fillText('loss ↑ · drag to rotate', 12, 18);
}

/* drag on the 3-d canvas orbits the camera */
function initDescent3dDrag() {
  const cv = $('#detailDescent3dCanvas');
  let last = null;
  cv.addEventListener('pointerdown', (e) => {
    last = [e.clientX, e.clientY];
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', (e) => {
    if (!last) return;
    d3yaw += (e.clientX - last[0]) * 0.01;
    d3elev = Math.max(0.2, Math.min(1.45, d3elev + (e.clientY - last[1]) * 0.008));
    last = [e.clientX, e.clientY];
    requestRender();
  });
  const end = () => { last = null; };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);
}

function requestRender() { needsRender = true; }

/* ---------------- pipeline strip ---------------- */
/* tiny activation glyphs for the connector chips: axes + curve + a gold dot
   at the origin, echoing the node-detail diagrams */
const ACT_ICON = {
  relu:    { axes: 'M10 1 V15 M2 12 H18', curve: 'M2 12 H10 L18 3',       dot: [10, 12] },
  tanh:    { axes: 'M10 1 V15 M2 8 H18',  curve: 'M2 14 C8 14 12 2 18 2', dot: [10, 8] },
  sigmoid: { axes: 'M10 1 V15 M2 8 H18',  curve: 'M2 13 C8 13 12 3 18 3', dot: [10, 8] },
};

/* the chip between mini i−1 and mini i names the operation that produced i */
function pipeOpEl(prev, inf) {
  const el = document.createElement('span');
  el.className = 'pipe-op';
  el.setAttribute('aria-hidden', 'true');
  if (inf.type === 'fc' && prev.type === 'input') {
    el.textContent = '(x₁, x₂)';
    el.setAttribute('data-tip', 'The raw input — a position (km east, km north) flowing into the first layer of weighted sums.');
  } else if (inf.type === 'fc') {
    el.textContent = 'Σ w·x + b';
    el.setAttribute('data-tip', 'Linear step: each neuron takes a weighted sum of the previous layer’s outputs, plus its bias.');
  } else if (inf.type === 'softmax') {
    el.textContent = 'eᶻ / Σeᶻ';
    el.setAttribute('data-tip', 'Softmax: exponentiate the class scores and normalize, turning them into probabilities that sum to 1.');
  } else {
    const ic = ACT_ICON[inf.type] || ACT_ICON.relu;
    el.innerHTML =
      `<svg viewBox="0 0 20 16" width="20" height="16" fill="none" stroke="currentColor" `
      + `stroke-linecap="round" stroke-linejoin="round">`
      + `<path d="${ic.axes}" stroke-width="1" opacity="0.35"/>`
      + `<path d="${ic.curve}" stroke-width="1.6"/>`
      + `<circle cx="${ic.dot[0]}" cy="${ic.dot[1]}" r="2" fill="#c98500" stroke="none"/>`
      + `</svg>${inf.type}`;
    el.setAttribute('data-tip', `Activation: ${inf.type} bends each neuron’s weighted sum — this is where the non-linearity enters.`);
  }
  return el;
}

function buildPipeline() {
  const host = $('#pipeline');
  host.innerHTML = '';
  minis = [];
  const info = describeLayers();
  info.forEach((inf, i) => {
    if (i > 0) host.appendChild(pipeOpEl(info[i - 1], inf));
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'pipe-item';
    item.setAttribute('aria-label', `Inspect ${inf.title}`);
    item.setAttribute('data-tip', inf.expl);
    if (i > info.length * 0.6) item.classList.add('tip-right');
    const cv = document.createElement('canvas');
    const nm = document.createElement('span');
    nm.className = 'nm';
    nm.textContent = `${inf.short} (${inf.dim})`;
    item.appendChild(cv);
    item.appendChild(nm);
    item.addEventListener('click', () => selectLayer(i));
    host.appendChild(item);
    minis.push({ el: item, cv });
  });
}

function selectLayer(i) {
  state.view = { kind: 'layer' };
  state.lix = i;
  state.d0 = 0;
  state.d1 = Math.min(1, net.layers[i].out_depth - 1);
  detailShowLayerTab();
  updateSelectionUI();
  requestRender();
}

function selectNode(rowIdx, j) {
  const rows = archLayout();
  state.view = { kind: 'node', rowIdx, j };
  state.lix = rows[rowIdx].idxs[0];
  detailShowLayerTab();
  updateSelectionUI();
  requestRender();
}

function selectEdge(rowIdx, i, j) {
  const rows = archLayout();
  state.view = { kind: 'edge', rowIdx, i, j };
  state.lix = rows[rowIdx].fcIdx;
  detailShowLayerTab();
  updateSelectionUI();
  requestRender();
}

function updateSelectionUI() {
  // the landscape tabs own the header while up; the selection keeps
  // driving the filmstrip highlight underneath
  if (detailTab !== 'layer') {
    $('#cycleBtn').hidden = true;
    minis.forEach((m, i) => m.el.classList.toggle('sel', i === state.lix));
    return;
  }
  const v = state.view;
  const btn = $('#cycleBtn');
  let title = '', expl = '';
  if (v.kind === 'node') {
    const rows = archLayout();
    const row = rows[v.rowIdx];
    btn.hidden = true;
    if (v.rowIdx === 0) {
      title = `input · x${sub(v.j + 1)}`;
      expl = 'One raw input feature of the traced point. It fans out to every neuron in the first layer, each scaling it with its own learned weight.';
    } else if (row.label === 'output') {
      title = `output · ${v.j === CAT ? 'rough' : 'ok'} score`;
      expl = 'A class-score neuron: it sums weighted evidence from the layer before, and softmax turns the two scores into probabilities that sum to 1.';
    } else {
      title = `${row.label} · neuron ${v.j}`;
      expl = `One neuron: each incoming value is scaled by its learned weight, summed with the bias b, passed through ${row.side || 'the activation'} — and the result fans out to the next layer. Training nudges every w and b to lower the loss.`;
    }
  } else if (v.kind === 'edge') {
    const rows = archLayout();
    const from = rows[v.rowIdx - 1], to = rows[v.rowIdx];
    btn.hidden = true;
    title = `weight · ${from.label} n${v.i} → ${to.label} n${v.j}`;
    expl = 'A single learned weight. It scales the sender’s output on its way into the receiver: positive (blue) pushes the receiver up, negative (orange) pulls it down, and the size sets how strongly. Every training step nudges it against the loss gradient.';
  } else {
    const info = describeLayers()[state.lix];
    title = info.title;
    expl = info.expl;
    const depth = net.layers[state.lix].out_depth;
    btn.hidden = depth <= 2;
    btn.textContent = `⟳ neurons ${state.d0}·${state.d1}`;
  }
  $('#layerTitle').textContent = title;
  $('#layerExpl').textContent = expl;
  minis.forEach((m, i) => m.el.classList.toggle('sel', i === state.lix));
}

function cycleNeurons() {
  const depth = net.layers[state.lix].out_depth;
  state.d0 = (state.d0 + 1) % depth;
  state.d1 = (state.d1 + 1) % depth;
  updateSelectionUI();
  requestRender();
}

/* ---------------- architecture controls ---------------- */
function renderLayerChips() {
  const host = $('#layerChips');
  host.innerHTML = '';
  state.hidden.forEach((n, i) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const lab = document.createElement('span');
    lab.textContent = 'L' + (i + 1);
    const rng = document.createElement('input');
    rng.type = 'range';
    rng.min = 2; rng.max = 8; rng.step = 1; rng.value = n;
    rng.className = 'chip-slider';
    rng.setAttribute('aria-label', `Neurons in hidden layer ${i + 1}`);
    const val = document.createElement('b');
    val.className = 'chip-val';
    val.textContent = n;
    rng.addEventListener('input', () => { val.textContent = rng.value; });
    rng.addEventListener('change', () => {   // commit on release: rebuild resets weights
      state.hidden[i] = +rng.value;
      buildNet();
    });
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'x';
    x.textContent = '×';
    x.setAttribute('aria-label', `Remove hidden layer ${i + 1}`);
    x.addEventListener('click', () => {
      state.hidden.splice(i, 1);
      renderLayerChips();
      buildNet();
    });
    chip.append(lab, rng, val, x);
    host.appendChild(chip);
  });
  if (state.hidden.length < MAX_HIDDEN_LAYERS) {
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'add-layer';
    add.textContent = '+ layer';
    add.setAttribute('data-tip',
      'Add another hidden layer. Depth lets the net compose folds — each layer bends the already-bent space of the one before.');
    add.addEventListener('click', () => {
      state.hidden.push(4);
      renderLayerChips();
      buildNet();
    });
    host.appendChild(add);
  }
  applyRunLocks();
}

/* ---------------- data editing ---------------- */
const HIT_RADIUS_PX = 9;

function pointAt(xt, yt, ss) {
  let mink = -1, mind = Infinity;
  for (let k = 0; k < data.length; k++) {
    const dx = (data[k][0] - xt) * ss, dy = (data[k][1] - yt) * ss;
    const d = dx * dx + dy * dy;
    if (d < mind) { mind = d; mink = k; }
  }
  return mind <= HIT_RADIUS_PX * HIT_RADIUS_PX ? mink : -1;
}

function selectPoint(k) {
  selectedPt = k;
  const bar = $('#ptActions'), instr = $('#featInstr'), adders = $('#pirepAdders');
  if (k < 0 || k >= data.length) {
    selectedPt = -1;
    bar.hidden = true;
    instr.hidden = false;
    adders.hidden = state.mode === 'infer';
    return;
  }
  bar.hidden = false;
  instr.hidden = true;
  adders.hidden = true;
  $('#ptLabel').textContent =
    `${labels[k] === DOG ? 'ok' : 'rough'} PIREP (${fmt(data[k][0])}, ${fmt(data[k][1])})`;
  $('#btnTag').textContent = tagged[k] ? 'untag' : 'tag to trace';
}

/* new PIREPs land like radio calls: tail number + grade first, then the label
   fades and only the colored dot remains (mirrors the intro video) */
let pendingPireps = []; // { x, y, lab, text }

/* weather-change animation: the fresh dataset's reports arrive one at a time,
   in random order, each announcing its tail number + grade before settling
   into a dot — the intro video played out on the real airspace map */
let reveal = null; // { shown: Set<index>, timer } while arriving

function clearDataReveal() {
  if (!reveal) return;
  clearInterval(reveal.timer);
  reveal = null;
  pendingPireps = [];
  requestRender();
}

function animateDataArrival() {
  clearDataReveal();
  if (!data.length) return;
  const order = data.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  let k = 0;
  reveal = { shown: new Set(), timer: null };
  // ~4.5 s total whatever the dataset size; with dense datasets only a
  // sample of reports get the tail-number callout (as in the intro video —
  // labelling all 200 would just be noise)
  const step = Math.max(18, Math.min(160, 4500 / order.length));
  const labelEvery = Math.max(1, Math.ceil(order.length / 28));
  reveal.timer = setInterval(() => {
    const i = order[k++];
    reveal.shown.add(i);
    if (k % labelEvery === 0 || order.length <= 28) {
      const grade = (labels[i] === DOG ? 1 : 4) + Math.floor(Math.random() * 3);
      const entry = { x: data[i][0], y: data[i][1], lab: labels[i], text: `${randTail()} · ${grade}` };
      pendingPireps.push(entry);
      setTimeout(() => {
        pendingPireps = pendingPireps.filter((e) => e !== entry);
        requestRender();
      }, 1250);
    }
    if (k >= order.length) {
      clearInterval(reveal.timer);
      reveal = null; // all reports in; the last labels fade on their own
    }
    requestRender();
  }, step);
}

function addPirep(cls, x, y) {
  if (x == null) { // the random chip drops its report anywhere
    x = (Math.random() * 2 - 1) * 4.6;
    y = (Math.random() * 2 - 1) * 4.6;
  }
  const lab = cls == null ? (Math.random() < 0.5 ? DOG : CAT) : cls;
  data.push([x, y]);
  labels.push(lab);
  tagged.push(false);
  markDescentDirty(); // the surface is loss-over-THIS-data
  if (reveal) reveal.shown.add(data.length - 1); // a hand-filed report lands at once
  const grade = (lab === DOG ? 1 : 4) + Math.floor(Math.random() * 3);
  const entry = { x, y, lab, text: `${randTail()} · ${grade}` };
  pendingPireps.push(entry);
  setTimeout(() => {
    pendingPireps = pendingPireps.filter((e) => e !== entry);
    requestRender();
  }, 1400);
  requestRender();
}

/* ok / rough chips arm a placement: crosshair cursor + blinking chip until the
   user clicks the map to drop the report there */
let armedPirep = null; // DOG | CAT | null

function armPirep(cls, btn) {
  const rearm = armedPirep !== cls;
  disarmPirep();
  if (!rearm) return; // clicking the armed chip again cancels
  armedPirep = cls;
  document.body.classList.add('placing');
  btn.classList.add('arming');
}

function disarmPirep() {
  armedPirep = null;
  document.body.classList.remove('placing');
  document.querySelectorAll('.pirep-add.arming').forEach((b) => b.classList.remove('arming'));
}

function onCanvasEdit(e) {
  const S = e.currentTarget.getBoundingClientRect().width;
  const ss = S / 10.4;
  const xt = (e.offsetX - S / 2) / ss;
  const yt = (e.offsetY - S / 2) / ss;
  if (state.mode === 'infer') {
    inferPt = [xt, yt];
    clearPass(); // a moved point invalidates the running animation
    updateInferReadout();
    requestRender();
    return;
  }
  if (armedPirep != null) {
    addPirep(armedPirep, xt, yt); // the armed report lands where you click
    disarmPirep();
    return;
  }
  // otherwise clicks only select existing PIREPs
  selectPoint(pointAt(xt, yt, ss));
  requestRender();
}

/* ---------------- play / reset ---------------- */
function setPlaying(on) {
  if (on) {
    clearPass();
    setRunActive(true);
  }
  state.playing = on;
  // ±1 epoch stay visible so nothing shifts; they only work while paused
  $('#stepBtn').disabled = on;
  setUndoDisabled(!history.length);
  const btn = $('#playBtn');
  btn.setAttribute('aria-pressed', String(on));
  $('#playIcon').innerHTML = on ? ICON_PAUSE : ICON_PLAY;
  updatePlayLabel();
  applyRunLocks(); // pausing freezes the rate sliders too
}

/* "Train" only while no run is active (fresh weights); once the train has left
   the station a pause reads "Resume" until Reset weights */
function updatePlayLabel() {
  $('#playLabel').textContent = state.playing ? 'Pause' : (runActive ? 'Resume' : 'Train');
}

/* ---------------- side panel: tabs + background-click highlighting ---------------- */
function initSidePanel() {
  const tabs = [
    ['tabIntro', 'introPane'],
    ['tabDescribe', 'describePane'],
    ['tabExplore', 'explorePane'],
    ['tabDelve', 'delvePane'],
    ['tabDecide', 'decidePane'],
  ];
  const activate = (name) => {
    tabs.forEach(([t, p]) => {
      const on = t === name;
      $('#' + t).setAttribute('aria-selected', String(on));
      $('#' + p).hidden = !on;
    });
  };
  tabs.forEach(([t]) => $('#' + t).addEventListener('click', () => activate(t)));

  // Next chips walk the tabs in order (Decide wraps back to Intro), each
  // landing at the top of the next pane
  document.querySelectorAll('.side-next .next-chip').forEach((b) => {
    b.addEventListener('click', () => {
      $('#' + b.dataset.next).click();
      const pane = tabs.find(([t]) => t === b.dataset.next)[1];
      $('#' + pane).scrollTop = 0;
    });
  });

  // clicking a panel's non-interactive background highlights its Describe card
  // (and flashes the panel itself); Controls/Network map to their pass-mode
  // cards while in single-pass mode
  let hlTimer = null;
  let panelTimer = null;
  const MAP = {
    panelControls: () => (state.mode === 'pass' ? 'desc-pass-controls'
      : state.mode === 'infer' ? 'desc-infer' : 'desc-controls'),
    panelFeature: () => (state.mode === 'infer' ? 'desc-infer' : 'desc-feature'),
    panelNetwork: () => (state.mode === 'pass' ? 'desc-pass-network' : 'desc-network'),
    panelDetail: () => 'desc-detail',
    panelPipeline: () => 'desc-pipeline',
    panelDescent: () => 'desc-descent',
    timeline: () => 'desc-timeline',
  };
  // these panels' canvases are pure display, so they don't block the lookup
  const CANVAS_OK = new Set(['timeline', 'panelDescent']);
  for (const [secId, cardIdOf] of Object.entries(MAP)) {
    const sec = document.getElementById(secId);
    sec.addEventListener('click', (e) => {
      if (!CANVAS_OK.has(secId)
        && e.target.closest('button, canvas, select, input, label, a')) return;
      activate('tabDescribe');
      const card = document.getElementById(cardIdOf());
      if (card.id !== 'desc-timeline') endTimelinePreview();
      document.querySelectorAll('.side-card.hl').forEach((c) => c.classList.remove('hl'));
      card.classList.add('hl');
      card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      clearTimeout(hlTimer);
      hlTimer = setTimeout(() => card.classList.remove('hl'), 2600);
      document.querySelectorAll('.hl-panel').forEach((p) => p.classList.remove('hl-panel'));
      sec.classList.add('hl-panel');
      clearTimeout(panelTimer);
      panelTimer = setTimeout(() => sec.classList.remove('hl-panel'), 2600);
    });
  }

  // …and the reverse: clicking a Describe card flashes the page panel(s) it
  // documents. Mode scoping already hides any card whose panel is off screen,
  // so the static map stays truthful; desc-infer covers two panels because the
  // forward map sends both there in inference mode.
  const RMAP = {
    'desc-controls': ['panelControls'],
    'desc-pass-controls': ['panelControls'],
    'desc-infer': ['panelControls', 'panelFeature'],
    'desc-timeline': ['timeline'],
    'desc-feature': ['panelFeature'],
    'desc-pass-network': ['panelNetwork'],
    'desc-network': ['panelNetwork'],
    'desc-detail': ['panelDetail'],
    'desc-pipeline': ['panelPipeline'],
    'desc-descent': ['panelDescent'],
  };
  for (const [cardId, secIds] of Object.entries(RMAP)) {
    const card = document.getElementById(cardId);
    card.addEventListener('click', (e) => {
      if (e.target.closest('a, button')) return;
      // unique case: with no run on screen, the timeline card toggles a live
      // preview of the timeline in the title's place. Clicking the selected
      // card again deselects it and the title returns (unless a run has since
      // made the timeline permanent — updateHeaderSwap re-applies the rules).
      if (cardId === 'desc-timeline' && timelinePreview) {
        endTimelinePreview();
        document.querySelectorAll('.hl-panel').forEach((p) => p.classList.remove('hl-panel'));
        return;
      }
      const previewing = cardId === 'desc-timeline' && !timelineShown();
      if (previewing) {
        timelinePreview = true;
        updateHeaderSwap();
        requestRender();
      } else if (cardId !== 'desc-timeline') {
        endTimelinePreview();
      }
      document.querySelectorAll('.side-card.hl').forEach((c) => c.classList.remove('hl'));
      card.classList.add('hl');
      clearTimeout(hlTimer);
      // a previewing card stays selected until explicitly deselected
      if (!previewing) hlTimer = setTimeout(() => card.classList.remove('hl'), 2600);
      document.querySelectorAll('.hl-panel').forEach((p) => p.classList.remove('hl-panel'));
      let scrolled = false;
      for (const sid of secIds) {
        const sec = document.getElementById(sid);
        // the timeline only exists on screen while a run (or preview) shows it
        if (sid === 'timeline' && sec.classList.contains('swapped-out')) continue;
        sec.classList.add('hl-panel');
        if (!scrolled) {
          sec.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          scrolled = true;
        }
      }
      clearTimeout(panelTimer);
      panelTimer = setTimeout(() => {
        document.querySelectorAll('.hl-panel').forEach((p) => p.classList.remove('hl-panel'));
      }, 2600);
    });
  }

  // every other card — Intro, Explore, Delve, Decide — is clickable too, even
  // with no page panel to point at: the click just moves the highlight there
  const wired = new Set(Object.keys(RMAP));
  document.querySelectorAll('.side-card').forEach((card) => {
    if (wired.has(card.id)) return;
    card.addEventListener('click', (e) => {
      if (e.target.closest('a, button')) return;
      endTimelinePreview();
      document.querySelectorAll('.side-card.hl').forEach((c) => c.classList.remove('hl'));
      document.querySelectorAll('.hl-panel').forEach((p) => p.classList.remove('hl-panel'));
      card.classList.add('hl');
      clearTimeout(hlTimer);
      hlTimer = setTimeout(() => card.classList.remove('hl'), 2600);
    });
  });
}

/* ---------------- one-off welcome "video" modal ---------------- */
/* PIREPs arrive over time: tail number + grade flash up, then settle into the
   colored dot the feature space will show from then on. Decorative, so
   Math.random is fine. */
let modalTimers = [];
const mt = (fn, ms) => modalTimers.push(setTimeout(fn, ms));
function clearModalTimers() { modalTimers.forEach(clearTimeout); modalTimers = []; }

function buildModalScene() {
  clearModalTimers();
  const svg = $('#modalNet');
  const NS = 'http://www.w3.org/2000/svg';
  const W = 580, H = 260;
  let grid = '';
  for (let x = 20; x < W; x += 40) grid += `M${x} 0V${H}`;
  for (let y = 20; y < H; y += 40) grid += `M0 ${y}H${W}`;
  svg.innerHTML = `<path d="${grid}" stroke="#2c2c2a" stroke-width="1" fill="none" opacity="0.55"/>`;

  // two interleaved spiral bands of reports — the cyclone dataset in miniature
  const cx = W / 2, cy = H / 2 - 10;
  const pts = [];
  for (let k = 0; k < 9; k++) {
    const t = 0.55 + k * 0.42, r = 13 + k * 10.5;
    pts.push({ x: cx + r * Math.cos(t) * 1.65, y: cy + r * Math.sin(t) * 0.72,
      ok: false, g: 4 + (k % 3) });
    pts.push({ x: cx + r * Math.cos(t + Math.PI) * 1.65, y: cy + r * Math.sin(t + Math.PI) * 0.72,
      ok: true, g: 1 + (k % 3) });
  }
  // position-only calls with no ride value ("· –") scattered anywhere: once
  // every echo is grey, they bury the spiral in what reads as random traffic
  for (let k = 0; k < 12; k++) {
    pts.push({ x: 26 + Math.random() * (W - 52), y: 14 + Math.random() * (H - 62), g: null });
  }
  // shuffle so graded and ungraded calls interleave
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]];
  }
  const addPt = (p) => {
    const col = p.g == null ? '#898781' : p.ok ? C.dog : C.cat;
    const dot = document.createElementNS(NS, 'circle');
    dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y);
    dot.setAttribute('r', 5.5);
    dot.setAttribute('fill', col);
    svg.appendChild(dot);
    const t = document.createElementNS(NS, 'text');
    t.textContent = `${randTail()} · ${p.g == null ? '–' : p.g}`;
    t.setAttribute('x', Math.min(Math.max(p.x + 8, 6), W - 70));
    t.setAttribute('y', p.y < 20 ? p.y + 17 : p.y - 8);
    t.setAttribute('fill', col);
    t.setAttribute('font-size', '9.5');
    t.setAttribute('font-family', 'ui-monospace,Menlo,monospace');
    t.setAttribute('font-weight', '600');
    svg.appendChild(t);
    // the echo goes grey: pilots hear the report, but they don't keep a map —
    // hiding the pattern is what makes "should it divert?" a real question
    mt(() => { t.remove(); dot.setAttribute('r', 4); dot.setAttribute('fill', '#6e6c66'); }, 1050);
  };

  const finale = () => {
    const route = document.createElementNS(NS, 'path');
    route.setAttribute('d', `M-8 232 C 150 212, 230 158, ${cx} ${cy} S 470 62, 588 44`);
    route.setAttribute('stroke', C.tag);
    route.setAttribute('stroke-width', '1.6');
    route.setAttribute('stroke-dasharray', '6 5');
    route.setAttribute('fill', 'none');
    svg.appendChild(route);
    const plane = document.createElementNS(NS, 'path');
    plane.setAttribute('d', 'M0 -10 L2.2 -3 L11 3 L11 5.5 L2.2 3 L1.6 8.5 L5 11 L5 13 '
      + 'L0 11.6 L-5 13 L-5 11 L-1.6 8.5 L-2.2 3 L-11 5.5 L-11 3 L-2.2 -3 Z');
    plane.setAttribute('fill', '#e8e6dc');
    plane.setAttribute('transform', 'translate(30 226) rotate(83)');
    svg.appendChild(plane);
    const q = document.createElementNS(NS, 'text');
    q.textContent = 'should it divert — and if so, how?';
    q.setAttribute('x', W / 2); q.setAttribute('y', 251);
    q.setAttribute('text-anchor', 'middle');
    q.setAttribute('fill', C.tag);
    q.setAttribute('font-size', '11');
    q.setAttribute('font-family', 'ui-monospace,Menlo,monospace');
    q.setAttribute('font-weight', '700');
    svg.appendChild(q);
  };

  // no reduced-motion fallback: this only ever plays on first load or after an
  // explicit "replay video" click, so the viewer has opted into the animation
  pts.forEach((p, i) => mt(() => addPt(p), 350 + i * 210));
  mt(finale, 350 + pts.length * 210 + 750);
}

function randTail() {
  return 'N' + (100 + Math.floor(Math.random() * 900))
    + String.fromCharCode(65 + Math.floor(Math.random() * 26))
    + String.fromCharCode(65 + Math.floor(Math.random() * 26));
}

function dismissModal(followUp) {
  $('#introModal').classList.add('dismissed');
  clearModalTimers();
  // the follow-up (start training, highlight the intro, ...) waits until the
  // full-screen switch has played out, so it isn't competing for the eye
  fsAttention(followUp);
}

function showModal() {
  $('#introModal').classList.remove('dismissed');
  buildModalScene();
  $('#modalClose').focus();
}

function initModal() {
  // every exit lands on the Intro tab with "What you're looking at" spotlit;
  // the Train button additionally sets the run going
  const introSpotlight = () => {
    $('#tabIntro').click();
    const card = document.getElementById('intro-what');
    card.classList.add('hl');
    card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    setTimeout(() => card.classList.remove('hl'), 2600);
  };
  $('#modalClose').addEventListener('click', () => dismissModal(introSpotlight));
  $('#modalTrain').addEventListener('click', () => {
    dismissModal(() => { introSpotlight(); setPlaying(true); });
  });
  $('#modalIntro').addEventListener('click', () => dismissModal(introSpotlight));
  // no backdrop or Esc dismissal: leaving the modal is an explicit choice
  // (✕, Train, or Intro)
  $('#replayBtn').addEventListener('click', showModal);
  showModal();
}

/* ---------------- full-screen banner ---------------- */
const ICON_EXPAND =
  '<path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/>'
  + '<path d="M2 2l4 4M14 2l-4 4M14 14l-4-4M2 14l4-4"/>';
const ICON_COMPRESS =
  '<path d="M6 2v4H2M10 2v4h4M14 10h-4v4M2 10h4v4"/>'
  + '<path d="M6 6L2 2M10 6l4-4M10 10l4 4M6 10l-4 4"/>';

function initFullscreen() {
  const sync = () => {
    const fs = !!document.fullscreenElement;
    $('#fsLabel').textContent = fs
      ? 'full screen — click here (or press esc) to exit'
      : 'switching to full screen to give the best experience';
    const icon = fs ? ICON_COMPRESS : ICON_EXPAND;
    $('#fsIcon').innerHTML = icon;
    $('#fsIconTop').innerHTML = icon;
  };
  const toggle = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  };
  $('#fsBar').addEventListener('click', toggle);
  $('#fsBtnTop').addEventListener('click', toggle);
  $('#fsBar').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });
  document.addEventListener('fullscreenchange', sync);
  sync();
}

/* after the intro modal closes: blink the toggle white for four seconds, then
   auto-switch to full screen and retire the banner. The dismissed-button's
   follow-up runs only after that, so it doesn't compete with the switch — and
   the toggle stays available if the user wants to switch back. */
let fsAttnTimer = null;
function fsAttention(followUp) {
  const btn = $('#fsBtnTop');
  const bar = $('#fsBar');
  const done = () => { if (followUp) followUp(); };
  if (document.fullscreenElement) {
    bar.classList.add('gone'); // nothing left to sell
    done();
    return;
  }
  // pitch already delivered on an earlier dismissal (they may have exited
  // full screen on purpose) — don't force the switch a second time
  if (bar.classList.contains('gone')) { done(); return; }
  // the pitch only works if it's on screen: surface the banner and toggle
  window.scrollTo({ top: 0, behavior: 'smooth' });
  btn.classList.add('attn');
  clearTimeout(fsAttnTimer);
  fsAttnTimer = setTimeout(() => {
    btn.classList.remove('attn');
    // 4 s still sits inside the dismissing click's transient-activation
    // window (~5 s in Chrome), so the auto-switch is allowed to succeed
    const switching = document.fullscreenElement
      ? Promise.resolve()
      : document.documentElement.requestFullscreen();
    switching
      .then(() => bar.classList.add('gone'))
      .catch(() => {}) // switch refused: leave the banner as the manual handle
      .finally(done);
  }, 4000);
}

/* ---------------- init ---------------- */
function init() {
  setData(state.dataset);
  renderLayerChips();
  buildNet();

  $('#playBtn').addEventListener('click', () => setPlaying(!state.playing));
  $('#resetBtn').addEventListener('click', () => buildNet());
  $('#stepBtn').addEventListener('click', stepForward);
  $('#backBtn').addEventListener('click', stepBack);
  document.querySelectorAll('.mode-seg button').forEach((b) => {
    b.addEventListener('click', () => setMode(b.dataset.mode));
  });
  $('#passRunBtn').addEventListener('click', passRunToggle);
  $('#passResetBtn').addEventListener('click', () => buildNet());
  $('#passUndoBtn').addEventListener('click', stepBack);
  $('#inferRunBtn').addEventListener('click', startInferRun);
  $('#passStepBackBtn').addEventListener('click', passStepBackward);
  $('#passStepFwdBtn').addEventListener('click', passStepForward);
  window.addEventListener('resize', syncPassRowHeight);
  initTimelineScrub();
  $('#cycleBtn').addEventListener('click', cycleNeurons);
  $('#detTabLayer').addEventListener('click', () => { detailTab = 'layer'; detailApplyTab(); });
  $('#detTabDescent').addEventListener('click', () => { detailTab = 'descent'; detailApplyTab(); });
  $('#detTabDescent3d').addEventListener('click', () => { detailTab = 'descent3d'; detailApplyTab(); });
  initDescent3dDrag();
  initSidePanel();
  initModal();
  initFullscreen();
  applyFeaturePanelLabels();

  document.querySelectorAll('#datasetSeg button').forEach((b) => {
    b.addEventListener('click', () => setData(b.dataset.set, $('#animCheck').checked));
  });

  $('#activationSel').addEventListener('change', (e) => {
    state.activation = e.target.value;
    buildNet();
  });
  // sliders apply live on drag; trainer swap keeps the learned weights
  const slider = (id, valId, key, steps, fmt) => {
    const el = $(id), out = $(valId);
    el.addEventListener('input', () => {
      const v = steps ? steps[+el.value] : +el.value;
      state[key] = v;
      out.textContent = fmt(v);
      buildTrainer();
    });
  };
  slider('#lrSlider', '#lrVal', 'lr', LR_STEPS, String);
  slider('#momentumSlider', '#momentumVal', 'momentum', null, (v) => v.toFixed(2));
  slider('#batchSlider', '#batchVal', 'batch', BATCH_STEPS, String);
  slider('#l2Slider', '#l2Val', 'l2', L2_STEPS, String);

  // one view state, two segs (epoch + inference control sets):
  // pilot view = reports only; ATC view = the net's full forecast overlay
  const setOverlay = (on) => {
    state.overlay = on;
    document.querySelectorAll('#viewSeg button, #viewSeg2 button').forEach((b) => {
      b.setAttribute('aria-pressed', String((b.dataset.view === 'atc') === on));
    });
    requestRender();
  };
  document.querySelectorAll('#viewSeg button, #viewSeg2 button').forEach((b) => {
    b.addEventListener('click', () => setOverlay(b.dataset.view === 'atc'));
  });

  const featCv = $('#featCanvas');
  featCv.addEventListener('pointerdown', onCanvasEdit);
  $('#archCanvas').addEventListener('pointerdown', onArchClick);

  $('#addOkBtn').addEventListener('click', (e) => armPirep(DOG, e.currentTarget));
  $('#addRoughBtn').addEventListener('click', (e) => armPirep(CAT, e.currentTarget));
  $('#addRandBtn').addEventListener('click', () => { disarmPirep(); addPirep(null); });
  $('#btnTag').addEventListener('click', () => {
    if (selectedPt < 0) return;
    const turnOn = !tagged[selectedPt];
    tagged.fill(false); // only one point can be traced at a time
    tagged[selectedPt] = turnOn;
    selectPoint(selectedPt);
    requestRender();
  });
  $('#btnRemove').addEventListener('click', () => {
    if (selectedPt < 0) return;
    data.splice(selectedPt, 1);
    labels.splice(selectedPt, 1);
    tagged.splice(selectedPt, 1);
    selectPoint(-1);
    markDescentDirty();
    requestRender();
  });
  $('#btnDismiss').addEventListener('click', () => {
    selectPoint(-1);
    requestRender();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    // in full screen, Esc belongs to the browser (exit full screen) — don't
    // also fire the app's Escape behaviors
    if (document.fullscreenElement) return;
    if (!$('#introModal').classList.contains('dismissed')) {
      return; // the modal only closes via its own buttons or the ✕
    }
    if (armedPirep != null) {
      disarmPirep();
      return;
    }
    if (selectedPt >= 0) {
      selectPoint(-1);
      requestRender();
    } else if (state.mode !== 'epoch') {
      setMode('epoch');
    }
  });

  const ro = new ResizeObserver(requestRender);
  ro.observe(featCv);
  ro.observe($('#layerCanvas'));
  ro.observe($('#archCanvas'));
  ro.observe($('#timelineCanvas'));
  ro.observe($('#descentCanvas'));

  setInterval(() => {
    if (state.playing) {
      trainStep();
      render();
      needsRender = false;
    } else if (needsRender) {
      render();
      needsRender = false;
    }
    descentTick();
  }, TICK_MS);
}

document.addEventListener('DOMContentLoaded', init);
