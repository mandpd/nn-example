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
  dog     : '#3987e5',   // class 1
  cat     : '#d95926',   // class 0
  dogWash : 'rgba(57, 135, 229, 0.17)',
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
  lr: 0.01, momentum: 0.1, batch: 10, l2: 0.001,
  dataset: 'circle',
  overlay: true,           // shade prediction regions
  playing: false,
  mode: 'epoch',           // 'epoch' | 'pass' (single-pass walkthrough)
  lix: 1,                  // selected layer index into net.layers
  d0: 0, d1: 1,            // neuron pair shown in the selected-layer view
  view: { kind: 'layer' }, // inspector mode: layer | node {rowIdx,j} | edge {rowIdx,i,j}
  epoch: 0, loss: null, lossHist: [],
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

function setData(name) {
  state.dataset = name;
  data = []; labels = [];
  DATASETS[name]();
  tagged = data.map(() => false);
  selectPoint(-1);
  clearPass();
  setRunActive(false);
  clearHistory();
  state.epoch = 0; state.loss = null; state.lossHist = [];
  document.querySelectorAll('#datasetSeg button').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.set === name));
  });
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
  clearHistory();
  state.epoch = 0; state.loss = null; state.lossHist = [];
  if (state.lix >= net.layers.length) state.lix = 1;
  state.view = { kind: 'layer' }; // node/edge references die with the old net
  state.d0 = 0;
  state.d1 = Math.min(1, net.layers[state.lix].out_depth - 1);
  buildPipeline();
  updateSelectionUI();
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
  // visibility swap inside a shared grid cell: the row keeps the header's height
  $('#pageHeader').classList.toggle('swapped-out', on);
  $('#timeline').classList.toggle('swapped-out', !on);
  applyRunLocks();
  requestRender();
}

/* dataset and architecture are fixed for the duration of a run — the train has
   left the station; Reset weights opens them up again */
function applyRunLocks() {
  document.querySelectorAll('#datasetSeg button').forEach((b) => { b.disabled = runActive; });
  $('#activationSel').disabled = runActive;
  document.querySelectorAll('#layerChips input, #layerChips button')
    .forEach((el) => { el.disabled = runActive; });
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
  const dom = timelineCeiling(state.epoch);
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
  const live = state.playing;
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
    hist: state.lossHist.slice(), weights: weightsSnapshot(),
  });
  if (history.length > HISTORY_CAP) history.shift();
  setUndoDisabled(false);
}

function setUndoDisabled(v) {
  $('#backBtn').disabled = v;
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
  setUndoDisabled(!history.length);
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
  if (m === 'pass' && state.playing) setPlaying(false);
  state.mode = m;
  document.querySelectorAll('.mode-seg button').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.mode === m));
  });
  $('#panelControls').classList.toggle('pass-mode', m === 'pass');
  document.querySelector('.viz').classList.toggle('passing', m === 'pass');
  if (m === 'pass') {
    passIdleUI(); // wait for the Run button — nothing trains on mode entry
    updatePassRunUI();
  } else {
    clearPass();
  }
  requestRender();
}

/* Run/Pause lifecycle for the pass animation */
function passRunToggle() {
  if (!pass) { startPass(); return; }
  if (pass.timer) {
    clearInterval(pass.timer);
    pass.timer = null;              // freeze: phase chips become clickable
  } else if (pass.si >= pass.steps.length - 1) {
    startPass();                     // finished: Run starts another pass
    return;
  } else {
    pass.timer = setInterval(advancePass, PASS_STEP_MS); // resume
  }
  updatePassRunUI();
}

function updatePassRunUI() {
  const running = !!(pass && pass.timer);
  $('#passRunIcon').innerHTML = running ? ICON_PAUSE : ICON_PLAY;
  $('#passRunLabel').textContent = running ? 'Pause' : 'Run';
  document.querySelectorAll('#passControls [data-phase]')
    .forEach((b) => { b.disabled = running || !pass; });
}

function startPass() {
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
  for (let ri = rows.length - 2; ri >= 0; ri--) steps.push({ phase: 'backward', row: ri });
  steps.push({ phase: 'update' });
  steps.push({ phase: 'done' });

  pass = { p, steps, si: 0, acts, grads, deltaByLi, dmax, loss: stats.loss, timer: null };
  updatePassUI();
  requestRender();
  pass.timer = setInterval(advancePass, PASS_STEP_MS);
  updatePassRunUI();
}

function advancePass() {
  if (!pass) return;
  if (pass.si < pass.steps.length - 1) {
    pass.si++;
    updatePassUI();
    requestRender();
  }
  if (pass.si >= pass.steps.length - 1 && pass.timer) {
    clearInterval(pass.timer);
    pass.timer = null;
    updatePassRunUI(); // animation finished: Run reappears, chips unlock
  }
}

function jumpPassPhase(phase) {
  if (!pass) return;
  const i = pass.steps.findIndex((s) => s.phase === phase);
  if (i < 0) return;
  pass.si = i;
  updatePassUI();
  requestRender();
  clearInterval(pass.timer);
  pass.timer = setInterval(advancePass, PASS_STEP_MS);
  updatePassRunUI();
}

function clearPass() {
  if (!pass) return;
  clearInterval(pass.timer);
  pass = null;
  if (state.mode === 'pass') passIdleUI();
  updatePassRunUI();
  requestRender();
}

function passIdleUI() {
  document.querySelectorAll('#passControls [data-phase]')
    .forEach((b) => b.classList.remove('on', 'done'));
  $('#passCaption').textContent =
    'press Run to train once on the traced (gold) point and watch the pass unfold';
}

function passCaption(st) {
  const rows = archLayout();
  const p = pass.p;
  const cls = labels[p] === DOG ? 'dog' : 'cat';
  switch (st.phase) {
    case 'sample':
      return `Following one training sample: (${fmt(data[p][0])}, ${fmt(data[p][1])}) — a ${cls}. Batch size is forced to 1, so this is exactly one pass through the net.`;
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
      return `Loss · the net says P(${cls}) = ${fmt(P)}; the truth is “${cls}”. Loss = −log(${fmt(P)}) = ${pass.loss.toFixed(3)} — the more right the net is, the smaller this gets.`;
    }
    case 'backward':
      return `Backward · the loss gradient flows back through ${rows[st.row].label} along the same weights — brighter violet = more responsibility for the error.`;
    case 'update':
      return `Update · every weight steps against its gradient (learning rate ${state.lr}): blue connections just got stronger, orange got weaker — thickness shows how much.`;
    default:
      return 'Pass complete — the net learned a little from this one example. Click a chip to replay a phase, or run another pass and watch the loss on this sample shrink.';
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
            expl: 'Scores become probabilities that sum to 1. Points past the diagonal are called dogs.' };
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
  if (t === 'softmax') return { x: 'P(cat)', y: 'P(dog)' };
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
  for (let i = 0; i < data.length; i++) {
    v.w[0] = data[i][0]; v.w[1] = data[i][1];
    net.forward(v, false);
    for (let li = 0; li < nL; li++) {
      const w = net.layers[li].out_act.w;
      f.pts[li].push({ x: w[0], y: w.length > 1 ? w[1] : 0, lab: labels[i], tag: tagged[i] });
    }
    const ws = net.layers[state.lix].out_act.w;
    f.selPts.push({ x: ws[state.d0], y: ws[state.d1], lab: labels[i], tag: tagged[i] });
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
  const layerFit = fitCanvas($('#layerCanvas'));
  field = computeField(featFit.w);
  drawFeature(featFit, field);
  drawSelected(layerFit, field);
  drawArch();
  drawMinis(field);
  drawSpark();
  if (runActive) drawTimeline();
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

  for (let i = 0; i < data.length; i++) {
    drawDot(ctx, S / 2 + data[i][0] * f.ss, S / 2 + data[i][1] * f.ss, 5,
      { lab: labels[i], tag: tagged[i] });
  }

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
  const t = tagged.findIndex(Boolean);
  return t >= 0 ? t : (data.length ? 0 : -1);
}

function outActOf(row) {
  return net.layers[row.idxs[row.idxs.length - 1]].out_act.w;
}

function forwardProbe(p) {
  const v = new convnetjs.Vol(1, 1, 2);
  v.w[0] = data[p][0];
  v.w[1] = data[p][1];
  net.forward(v, false);
}

function drawProbeCaption(ctx, S, p) {
  ctx.font = MONO;
  ctx.textAlign = 'left';
  ctx.fillStyle = labels[p] === DOG ? C.dog : C.cat;
  ctx.beginPath();
  ctx.arc(16, 17, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.fillText(
    `tracing point (${fmt(data[p][0])}, ${fmt(data[p][1])}) · ${labels[p] === DOG ? 'dog' : 'cat'}`, 26, 21);
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
    ctx.fillText(`P(${j === CAT ? 'cat' : 'dog'}) = ${fmt(a)}`, xAct + 54, mid + 4);
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

  const padTop = probe >= 0 ? 44 : 34;
  const padBot = 42;
  const actGap = 30, actBox = 17;
  const maxCount = Math.max(...rows.map((r) => r.count));
  const spacing = Math.min(52, (W - 150) / Math.max(maxCount - 1, 1));
  const extra = rows.reduce((s, r) => s + (r.side ? actGap : 0), 0);
  const gapBetween = rows.length > 1 ? (H - padTop - padBot - extra) / (rows.length - 1) : 0;
  const nodeY = [], actY = [];
  let yCur = padTop;
  rows.forEach((r, i) => {
    nodeY[i] = yCur;
    actY[i] = r.side ? yCur + actGap : null;
    yCur = (r.side ? yCur + actGap : yCur) + gapBetween;
  });
  const nodeX = (row, j) => W / 2 + (j - (row.count - 1) / 2) * spacing;
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
          : (row.fcIdx != null ? net.layers[row.fcIdx].out_act.w[j] : data[probe][j]);
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
          ctx.fillText(`P(${j === CAT ? 'cat' : 'dog'})`, nodeX(row, j), yA + actBox / 2 + 14);
        }
        ctx.textAlign = 'left';
        ctx.font = MONO;
      }
      ctx.fillStyle = actSel ? C.ink : C.muted;
      ctx.textAlign = 'right';
      ctx.fillText(row.side, W - 10, yA + 4);
      ctx.textAlign = 'left';
    }

    ctx.fillStyle = fcSel ? C.ink : C.muted;
    ctx.fillText(row.label, 10, yN + 4);
  }
  ctx.globalAlpha = 1;

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

function segDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len2)) : 0;
  const cx = x1 + t * dx, cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

function onArchClick(e) {
  const px = e.offsetX, py = e.offsetY;
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

function requestRender() { needsRender = true; }

/* ---------------- pipeline strip ---------------- */
function buildPipeline() {
  const host = $('#pipeline');
  host.innerHTML = '';
  minis = [];
  const info = describeLayers();
  info.forEach((inf, i) => {
    if (i > 0) {
      const a = document.createElement('span');
      a.className = 'pipe-arrow';
      a.textContent = '→';
      a.setAttribute('aria-hidden', 'true');
      host.appendChild(a);
    }
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
  updateSelectionUI();
  requestRender();
}

function selectNode(rowIdx, j) {
  const rows = archLayout();
  state.view = { kind: 'node', rowIdx, j };
  state.lix = rows[rowIdx].idxs[0];
  updateSelectionUI();
  requestRender();
}

function selectEdge(rowIdx, i, j) {
  const rows = archLayout();
  state.view = { kind: 'edge', rowIdx, i, j };
  state.lix = rows[rowIdx].fcIdx;
  updateSelectionUI();
  requestRender();
}

function updateSelectionUI() {
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
      title = `output · ${v.j === CAT ? 'cat' : 'dog'} score`;
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
  const bar = $('#ptActions'), instr = $('#featInstr');
  if (k < 0 || k >= data.length) {
    selectedPt = -1;
    bar.hidden = true;
    instr.hidden = false;
    return;
  }
  bar.hidden = false;
  instr.hidden = true;
  $('#ptLabel').textContent =
    `${labels[k] === DOG ? 'dog' : 'cat'} (${fmt(data[k][0])}, ${fmt(data[k][1])})`;
  $('#btnClass').textContent = `make it a ${labels[k] === DOG ? 'cat' : 'dog'}`;
  $('#btnTag').textContent = tagged[k] ? 'untag' : 'tag to trace';
}

function onCanvasEdit(e) {
  const S = e.currentTarget.getBoundingClientRect().width;
  const ss = S / 10.4;
  const xt = (e.offsetX - S / 2) / ss;
  const yt = (e.offsetY - S / 2) / ss;
  const k = pointAt(xt, yt, ss);
  if (k >= 0) {
    selectPoint(k);
  } else {
    data.push([xt, yt]);
    labels.push(DOG);
    tagged.push(false);
    selectPoint(-1);
  }
  requestRender();
}

/* ---------------- play / reset ---------------- */
function setPlaying(on) {
  if (on) {
    clearPass();
    setRunActive(true);
  }
  state.playing = on;
  $('#stepBtn').hidden = on;
  $('#backBtn').hidden = on;
  const btn = $('#playBtn');
  btn.setAttribute('aria-pressed', String(on));
  $('#playIcon').innerHTML = on ? ICON_PAUSE : ICON_PLAY;
  $('#playLabel').textContent = on ? 'Pause' : 'Train';
}

/* ---------------- side panel: tabs + background-click highlighting ---------------- */
function initSidePanel() {
  const tabs = [
    ['tabDescribe', 'describePane'],
    ['tabExplore', 'explorePane'],
    ['tabDelve', 'delvePane'],
  ];
  const activate = (name) => {
    tabs.forEach(([t, p]) => {
      const on = t === name;
      $('#' + t).setAttribute('aria-selected', String(on));
      $('#' + p).hidden = !on;
    });
  };
  tabs.forEach(([t]) => $('#' + t).addEventListener('click', () => activate(t)));

  // clicking a panel's non-interactive background highlights its Describe card
  let hlTimer = null;
  const MAP = {
    panelControls: 'desc-controls',
    panelFeature: 'desc-feature',
    panelNetwork: 'desc-network',
    panelDetail: 'desc-detail',
    panelPipeline: 'desc-pipeline',
  };
  for (const [secId, cardId] of Object.entries(MAP)) {
    document.getElementById(secId).addEventListener('click', (e) => {
      if (e.target.closest('button, canvas, select, input, label, a')) return;
      activate('tabDescribe');
      const card = document.getElementById(cardId);
      document.querySelectorAll('.side-card.hl').forEach((c) => c.classList.remove('hl'));
      card.classList.add('hl');
      card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      clearTimeout(hlTimer);
      hlTimer = setTimeout(() => card.classList.remove('hl'), 2600);
    });
  }
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
  $('#passUndoBtn').addEventListener('click', stepBack);
  document.querySelectorAll('#passControls [data-phase]').forEach((b) => {
    b.addEventListener('click', () => jumpPassPhase(b.dataset.phase));
  });
  $('#cycleBtn').addEventListener('click', cycleNeurons);
  initSidePanel();

  document.querySelectorAll('#datasetSeg button').forEach((b) => {
    b.addEventListener('click', () => setData(b.dataset.set));
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

  $('#overlayChk').addEventListener('change', (e) => {
    state.overlay = e.target.checked;
    requestRender();
  });

  const featCv = $('#featCanvas');
  featCv.addEventListener('pointerdown', onCanvasEdit);
  $('#archCanvas').addEventListener('pointerdown', onArchClick);

  $('#btnClass').addEventListener('click', () => {
    if (selectedPt < 0) return;
    labels[selectedPt] = labels[selectedPt] === DOG ? CAT : DOG;
    selectPoint(selectedPt); // refresh labels
    requestRender();
  });
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
    requestRender();
  });
  $('#btnDismiss').addEventListener('click', () => {
    selectPoint(-1);
    requestRender();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (selectedPt >= 0) {
      selectPoint(-1);
      requestRender();
    } else if (state.mode === 'pass') {
      setMode('epoch');
    }
  });

  const ro = new ResizeObserver(requestRender);
  ro.observe(featCv);
  ro.observe($('#layerCanvas'));
  ro.observe($('#archCanvas'));
  ro.observe($('#timelineCanvas'));

  setInterval(() => {
    if (state.playing) {
      trainStep();
      render();
      needsRender = false;
    } else if (needsRender) {
      render();
      needsRender = false;
    }
  }, TICK_MS);
}

document.addEventListener('DOMContentLoaded', init);
