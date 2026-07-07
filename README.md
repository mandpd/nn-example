# nn-example · how a neural net bends space

A tiny neural network learning to recognize patterns in data, live in your browser —
a heavily re-imagined version of Andrej Karpathy's
[classify2d demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
from [convnetjs](https://cs.stanford.edu/people/karpathy/convnetjs/). The original
convnetjs engine is used unmodified; everything around it — the scenario, the
visualizations, the interaction model — is new.

> **To view, click [here](https://mandpd.github.io/nn-example/nn_example.html)**

The sections below mirror the in-app **info tabs** (Intro · Describe · Explore ·
Delve inside · Decide) in the side panel.

---

## Intro — the scenario

### What you're looking at

The setting is an airspace map. Aircraft ahead of you have been filing **PIREPs** —
pilot reports of the ride, graded 1–6 from light turbulence to extreme. Grades
**1–3** (up to moderate turbulence, shown blue) are tolerable for a passenger
flight; grades **4–6** (moderate chop or worse, shown orange) are worth the fuel
cost of a diversion. Each PIREP is just a position and a grade — a dot on a map.
Turbulence is invisible; the only way to know about air nobody has flown through is
to *infer* it from the reports around it.

Each PIREP is one dot at position (x₁ = km east, x₂ = km north). The network's
whole job is to reconstruct the turbulence pattern from those scattered reports — a
real image classifier does the same thing with millions of inputs instead of two;
we use two so you can watch every step of it happen. The question the net exists to
answer: **should the flight divert — and if so, how?**

### Training vs. running

A network's life has two phases. **Training**: show it PIREPs whose ride is known,
measure how wrong its map is (the *loss*), and nudge every weight to be slightly
less wrong — then repeat, thousands of times. **Running** (inference): freeze the
weights and ask the finished net about waypoints nobody has reported on. The two
mode panels at the top right of the page mirror exactly this split — and this is
the answer to "why not just use the forecast?": every forecast map *is* the output
of a model like this one; the ground truth underneath is always scattered point
reports.

### Three modes, three insights

- **Epoch** — training at full speed. Watch the forecast boundary form on the
  airspace map, the loss fall, and each layer bend space in the filmstrip. This is
  also where you shape the experiment: file new PIREPs, change the weather pattern,
  the layers, the activation, the rates.
- **Single pass** — training in slow motion. One PIREP's forward pass, loss,
  backprop and weight update play out over an enlarged network diagram, phase by
  phase, with the real numbers. This is where the learning rule itself becomes
  visible.
- **Inference** — the trained net at work. Place a waypoint anywhere on the map and
  watch the forward pass turn it into P(ok) and P(rough) — and a call: stay on
  plan, or divert. Nothing learns here; this is what "using a model" means.

### Finding your way

Nearly everything on the page is clickable. In any mode, click the **background of
a panel** and the **Describe** tab jumps to that panel's card — the cards change
with the mode, so click around in each one. Click **circles, connections and
activation boxes** in the network diagram, or any **filmstrip stage**, to open its
detail view. The **Explore** tab holds guided experiments; the **Delve inside** tab
is a set of directed walkthroughs for single-pass mode, one phase of the learning
rule at a time.

---

## Describe — what each panel shows

### Controls

Everything that defines the experiment. **Train** runs gradient descent
continuously; while paused, **+1 / −1 epoch** step training forward one pass at a
time or rewind it — weights, loss and the epoch counter are restored exactly. The
**mode** chips switch between epoch, single pass, and inference. **Reset weights**
starts over from new random values.

**Weather** swaps the pattern behind the PIREPs — front, storm cell, cyclone bands,
patternless scattered convection, or **rare rough** (85 ok / 10 rough reports, the
class-imbalance setting where weighted cross-entropy and focal loss earn their
keep). **save csv / load csv** round-trip the reports as plain text — one row per
PIREP, `x0,x1,y`, with the position in map units (the visible airspace spans
±5.2) and `y` = 1 for ok, 0 for rough (`ok`/`rough` also accepted; header row
optional; up to 1000 rows). Export a preset to study or seed your own file, then
load any distribution the presets can't draw — XORs, checkerboards, real data
projected to 2-D — and it becomes the weather, with label noise and the
validation hold-out applying on top just like a preset. **Label noise**
regenerates the data with 10% or 25% of the grades filed wrong — the scenario
label smoothing and KL's soft targets are built for. The
**hidden layers** sliders and the **activation** menu change the architecture and
rebuild the net from scratch. The **loss** menu swaps what "wrong" means (see
[choosing the loss function](#choosing-the-loss-function)), the **optimizer** menu
swaps how the weights step downhill — sgd, nesterov, adagrad, windowgrad, or
adadelta (momentum only feeds sgd and nesterov, so it greys out under the adaptive
methods; adadelta ignores the learning rate entirely) — and the **learning rate,
momentum, batch size and L2** sliders retune the descent — all live, without
touching what has been learned — watch the timeline's loss trail react.

**Validation · hold out 25%** keeps a random quarter of the PIREPs out of training
(a fresh draw each time the box is ticked). The held-out
reports turn **grey** on the airspace map — display-only: they can't be selected,
tagged or traced — and their loss is the second number in the Loss stat and the
dashed teal trail on the timeline — the number training never sees. When teal lifts away while violet
keeps falling, the net is memorizing, not learning. **Miss · false alarm** counts
the two ways to be wrong over every plotted report: rough air graded ok (the
dangerous error), and ok air graded rough (costly but safe) — the scoreboard for
the imbalance-aware losses. **Model save / load** writes the trained weights and
dials to a JSON file and restores them later, and **⧉ share setup** (top right)
copies a link that reproduces the whole experiment — weather, architecture, loss,
optimizer, dials, noise, hold-out — from fresh weights.

Once a run starts, the page title gives way to the epoch timeline, and the dataset
and architecture controls fade and lock until **Reset weights**. What can still be
clicked keeps its raised background.

### Epoch timeline

Takes the title's place once a run starts. The gold line is **now**: every panel on
the page is a snapshot of this exact moment in an evolving training process. The
scale stretches as the epoch count climbs, and "paused" marks a frozen simulation.
The violet trail unrolling behind the now-line is the **loss** — the average error
the training is pushing down, drawn against the run's own worst value. With the
validation hold-out on, a dashed teal trail rides alongside: the held-out points'
loss, the run's honest score.
**+1 / −1 epoch** move the now-line with them — including backwards. While paused
you can also **drag the now-line** itself: scrub back through the recorded history
(about the last 200 snapshots) and forward again — every panel follows live, and
letting go commits the run to that moment.

### Airspace map (feature space)

The map is the net's feature space: every position is a possible input
(x₁ = km east, x₂ = km north); each dot is one PIREP — blue ok (grade 1–3), orange
divert-grade (4–6). In **ATC view**, the background shows the ride the net
currently forecasts everywhere; the decision boundary is where the colors meet —
**pilot view** shows only the reports, all a crew actually has.

File new reports with the chips: pick **+ ok** or **+ rough**, then click the map
to place it (**+ random** drops one anywhere) — each lands as a tail number and
grade before settling into its colored dot. Click any PIREP for its action bar:
**tag it to trace** (it turns gold and its values flow through the Network and
detail panels), or remove it. Every edit changes the training data immediately.

### Neural network

The architecture: input at the top, prediction at the bottom. Circles are each
neuron's weighted sum (Σ w·x + b); the small boxes beneath are its activation
stage; every line is one learned weight — blue positive, orange negative,
thicker = stronger — redrawn live as the net trains.

Click a **circle** to open that neuron's arithmetic, a **line** to inspect a single
weight, an **activation box** or a row label to see how that stage warps space.
While a component is selected, the traced point's actual values are printed at
every stage of the diagram.

### Layer space (detail)

Shows whatever is selected. For a **layer**: how it reshapes the previous stage's
output — the warped grid with real value axes and the data points carried along.
For a **neuron**: its inputs, weights, bias, activation curve and fan-out, computed
for the traced point. For a **weight**: its value, sign and contribution. When a
layer has more than two neurons, the ⟳ button cycles which pair forms the axes.

The panel is tabbed: **layer space** is this inspector, and the **loss map (2d)**
and **loss map (3d)** tabs hold the gradient-descent landscape (below) — handy for
watching the dot descend while a run plays. Clicking a tab also opens its card in
the side panel; selecting any layer, neuron or weight flips back to the inspector.

### Loss map (2d) and loss map (3d) tabs

**Loss map (2d)** is a contour map of the training loss around the net's current
weights, seen from directly above: **d₁** (x-axis) points along the direction the
run has been travelling, **d₂** (y-axis) is a fixed random perpendicular, and every
cell's shade is the real loss with the weights moved to that spot — **dark = low**,
bright violet = high. The **gold dot** is the current weights, sliding downhill
between re-surveys; the **violet trail** is the recent path projected onto the
slice; the **white ring** marks the lowest mapped point nearby.

**Loss map (3d)** draws the same survey as terrain: d₁ on x, d₂ on y, and the
**loss as height** — valleys are good weight settings, ridges are bad ones. **Drag
to rotate**: horizontal orbits the view, vertical tilts it toward or away from the
horizon. The violet trail is draped over the terrain it actually descended, the
gold dot sits on the surface at the current weights' own loss, and the white ring
marks the lowest surveyed vertex. Train and watch the bowl deepen while the dot
rolls to its floor — gradient descent, literally.

### Log book tab

Every flight keeps a log. The **camera button** in the detail panel's header files
a snapshot of whichever view is up — layer space, loss map (2d) or (3d) — as a
pixel copy *plus every setting that produced it*: epoch, loss, weather,
architecture, activation, and the optimizer dials. The **log book** tab plays the
entries back as a slideshow: ‹ › to page, ▶ to auto-advance, **save** to download
the PNG, **remove** to tear out a page. Entries persist in your browser between
visits (up to 24 — oldest pages fall out first). Use it to compare runs that no
longer exist: two different random starts, the same weather at two learning rates,
the landscape before and after a long train.

### The network, layer by layer

The whole forward pass as a filmstrip: each thumbnail is the same square of space
after one more stage. Linear layers rotate, scale and shear it; activations fold
(relu) or squash (tanh, sigmoid) it; softmax lines everything up on the probability
diagonal. Read left to right to watch the two classes come apart. Click any stage
to open it in the detail panel.

### Gradient descent · loss landscape

Training is descent on a surface: every possible setting of the weights has a
loss, and gradient descent walks downhill. The catch is that this net has ~40
weights, so the true surface lives in ~40 dimensions. The landscape panel (bottom
of the page) does the honest next-best thing: it takes a **2-D slice** through the
current weights — one axis (d₁) pointing along the direction the run has actually
been travelling, the other (d₂) a fixed random perpendicular — and evaluates the
*real training loss* at every point of that plane. Dark is low loss; bright violet
is high.

The **gold dot** is the net's current weights, sliding downhill between re-maps;
the **violet trail** is the recent path of the run projected onto the slice; the
**white ring** marks the lowest point of the mapped neighborhood. Watch a run
settle: the trail shortens and the dot comes to rest inside the dark basin on the
ring — a (local) minimum. Reset the weights and the net drops somewhere new on a
different part of the surface; raise the learning rate and watch the dot overshoot
the valley instead of settling into it. The map re-surveys itself as the run moves,
and the same view is available beside the network diagram via the detail panel's
**loss map (2d)** and **loss map (3d)** tabs.

### Single pass — run controls and the stage

**Run** follows the traced (gold) PIREP through training, and like epoch mode it
never stops on its own: each cycle animates that report's real training step —
**sample → forward → loss → backward → update**, batch size forced to 1 — then
fast-forwards the rest of the epoch in the background and rolls straight into the
next one. **Pause** freezes it and unlocks **−1 / +1 step**: walk the animation
backwards and forwards one step at a time while the phase chips light up to show
where you are. At the first step the back button reads **Prior epoch** and rewinds
into the end of the previous epoch's pass; at the final step the forward button
reads **Next epoch** and rolls into the next one, still paused. **undo pass**
rewinds one whole cycle. The narration line spells out what the current phase is
doing, with the real numbers.

On the expanded network diagram: during **forward**, each row lights up as the
sample's real activations arrive; **loss** rings the reported grade's softmax box
and prints the selected loss for the truth; **backward** washes neurons violet by their share of the
blame; **update** redraws the connections that just changed — blue got stronger,
orange got weaker, thickness shows how much. Everything stays clickable mid-pass.

### Matrix view (the arithmetic)

While a pass runs, the right side of the network panel shows the calculation the
highlighted layer is executing — with the pass's real numbers, so every row
multiplies out. A layer is one line of linear algebra: **z = W·a + b**. The weight
matrix **W** has one *row per neuron* and one *column per input* (its dimensions
are printed underneath); **a** is the previous layer's output as a column vector;
**b** is the bias vector — one nudge per neuron. Row × column + bias gives each
neuron's weighted sum in **z**.

The next step applies the **activation element-wise**: z → a through
relu/tanh/sigmoid (relu simply zeroes the negatives), or softmax turning the final
scores into the two probabilities in **P**. When a layer is too wide to print, the
view keeps the top-left corner of the matrix and marks the rest with ⋯ and ⋮ — the
true size stays in the dims label. The same panel then re-uses this notation
backwards: δa = Wᵀ·δz during backprop, and W + ΔW = W′ at the update.

### The loss function

The loss turns "how wrong was that?" into a single number. With the default,
cross-entropy, that number is **L = −log P[y]** — where P[y] is the probability the
net gave the ride the PIREP actually reported. At the **loss** step the matrix view
plots the selected loss as a curve over P[y] with the net's own (P[y], L) marked in
gold, and a badge names the loss under the output row — the loss lives just below
the net, comparing its output with the truth.

The shape of −log is the whole point: right and confident (P[y] → 1) costs nearly
nothing; unsure costs a little; **confidently wrong** (P[y] → 0) runs up the cliff
toward infinity. That asymmetry is what makes training spend its effort on the
reports it is getting wrong — a coin-flip answer sits near L = −log 0.5 ≈ 0.69,
the value a fresh net hovers at.

### Choosing the loss function

Cross-entropy is the standard choice for classification, but not the only feasible
one. The right loss depends on what you want the classifier to *care about* — and
because this demo outputs ok-vs-rough probabilities, you can watch each candidate
reshape the same training run. The **Loss** menu applies live: the readout jumps to
the new scale, the loss maps re-survey their terrain, and in single-pass mode the
loss step shows the new formula with real numbers. Every option still trains
through the same softmax head, so the probabilities every view reads stay
available; only what "wrong" means changes.

**Cross-entropy** (default) · L = −log P[y]. It lines up perfectly with
probability outputs: high probability on the correct class costs nearly nothing,
and confident wrong predictions run up a cliff toward infinity. Best
general-purpose choice — every alternative below is a deliberate trade against it.

**Weighted cross-entropy** · the same −log, but each class carries a price: here a
missed *rough* report costs **2.5×** a false alarm. For the aviation metaphor that
asymmetry is very reasonable — failing to warn about severe turbulence is worse
than diverting unnecessarily. Watch the boundary get pushed into blue territory:
the net would rather cry wolf than miss chop.

**Label-smoothed cross-entropy** · trains against softened targets,
**0.95 / 0.05** instead of 1 / 0 (ε = 0.1). PIREPs are subjective — smoothing says
"do not become infinitely certain from imperfect labels." The net keeps a sliver
of doubt everywhere, which often generalizes better; the loss never reaches 0,
even when perfectly right.

**KL divergence** · matches a full target *distribution* rather than a hard class
— here **0.9 / 0.1**, as if 9 of 10 pilots agreed on each grade. The gradient is
the same P − q diff as cross-entropy with soft labels; the value differs — it
reads 0 exactly when P equals the target. Useful whenever the label itself is
uncertain or comes from another model (distillation).

**Focal loss** · cross-entropy times **(1−P[y])²** (γ = 2). Reports the net
already gets right are damped toward zero loss, so training spends its effort on
the hard or rare ones. Made for imbalance — the **rare rough** weather preset is
its home turf: plain cross-entropy coasts on the easy ok region while focal keeps
digging at the 10 stragglers, with the miss counter as the scoreboard. The trade:
it gives up gradient on easy examples it has nearly won.

**Mean squared error** · treat the label as a number and punish (P − truth)².
Simple and feasible, but usually worse for classification: predicting 0.01 for the
true class is "bad" to MSE (loss ≈ 2) where cross-entropy calls it catastrophic
(loss ≈ 4.6) — and its gradient fades at saturated outputs, so confidently-wrong
nets learn slowly. Expect slower, less decisive training here.

**Hinge loss** · the classic SVM loss, scored on the raw class scores z, not the
probabilities: L = max(0, 1 − z[correct] + z[wrong]). Don't just call it rough —
call it rough *by a comfortable margin*. Once a report clears the margin its
gradient switches off entirely, so well-classified PIREPs stop shaping the
boundary at all.

**Squared hinge** · the same margin, squared:
max(0, 1 − z[correct] + z[wrong])². Large margin violations hurt much more, and
the loss fades smoothly to zero at the margin instead of meeting it at a kink — a
common variant in margin-based classifiers.

A rough ranking for this demo: **cross-entropy** as the general-purpose default ·
**weighted CE** if rough air is rare or safety-critical · **focal** when many easy
examples drown a few hard ones · **hinge** for margin-based classification ·
**MSE** feasible but usually not ideal. Cross-entropy stays popular because it
rewards exactly what a probability output should do: put mass on the truth. In
single-pass mode the loss step, its plotted curve, and the injected error signal
δz all follow the selection.

### Backprop · the diff flows back

Backprop starts with a beautiful coincidence: for softmax + cross-entropy (the
default loss), the slope of the loss at the output is simply **δz = P − y** — the
forecast minus the truth. No mystery quantity: *the diff itself is the error
signal*. The violet arrows under the output row show it being injected, and the
matrix view computes it from the real vectors. Other losses inject their own δz —
a class-weighted diff, a softened-target diff, a damped one, or hinge's flat ±1 —
and the seed step spells out whichever is active.

From there the diff walks back one layer at a time: **δa = Wᵀ·δz** — the very same
weights that carried values forward, now transposed, split the blame among the
neurons that supplied them. Each neuron's violet wash is its actual share;
activations gate the flow (a relu that output 0 passes no blame — dead neurons
don't learn). When the diff reaches the bottom, every weight knows its own slope,
and **update** steps each one against it: W ← W − η·∂L/∂W.

### Inference

Training is frozen — this mode asks the net for the ride ahead. The airspace map
becomes a single **waypoint** on your route: click anywhere to move it, and the
forecast readout updates instantly with P(ok) and P(rough) from the current
weights — plus the call a passenger flight would make: **stay on plan**, or
**divert**. The **divert threshold** slider is that call's dial: divert whenever
P(rough) is at or above the line. At 0.50 it is the plain "whichever is larger"
rule; slide it down and the orange divert region swallows the map (cautious), up
and only near-certain chop diverts. Training decides what the net believes — this
slider is policy, what you do about it. **Run** animates the forward pass — the
waypoint's two coordinates travel layer by layer through the network diagram until
they become the two probabilities. Nothing trains, no loss, no gradients: this is
the whole life of a deployed network.

---

## Explore — guided experiments, as to-do lists

Every exploration in the Explore tab is an ordered **to-do list**. Click any step
(or walk with the **‹ prev / next ›** buttons under each card) and the control it
talks about lights up on the main page — the mode and detail tab switch too if the
step needs them — while the steps behind you get ticked off. The current
walkthroughs:

**Watch a boundary being found.** Storm cell, ATC view, a few weight resets to see
different random starting guesses, then train and replay moments with
**+1 / −1 epoch**. Training is the loss dragging a random guess into shape.

**Give the cyclone more capacity.** Two layers of 4 usually can't wrap the spiral
bands; more neurons give the net more directions to fold space. The filmstrip
shows where the extra capacity gets spent.

**Two resets, two valleys.** Train to convergence, snapshot the **loss map (2d)**
with the camera, reset, train, snapshot again, and flip between the log-book
entries: same data, same architecture, two different valleys. That's why
practitioners train with several seeds.

**Rare rough · when misses matter.** The imbalanced preset, judged by the
**miss · false alarm** counter: plain cross-entropy coasts on the easy ok region;
weighted CE reprices the classes and trades false alarms for fewer misses; focal
mutes the easy examples so training digs at the 10 hard ones. Which trade would
you fly with?

**Noisy labels want soft targets.** 25% label noise plus the validation hold-out:
watch plain cross-entropy dig into the noise while the val curve stalls, then see
label smoothing keep train and val closer together.

**Overfit, then regularize.** Hold-out on, scattered weather, maximum capacity,
L2 at 0: violet keeps falling while dashed teal lifts away — that gap is
overfitting. Raise L2 mid-run and watch the gap close.

**Race the optimizers.** The loss map as a racetrack: sgd's steady steps,
momentum's overshoot, adadelta ignoring the learning rate, adagrad stalling late
as its history accumulates.

**Pick the divert line.** Inference mode: park the waypoint near the boundary and
slide the **divert threshold** — same net, same beliefs, different appetite for
risk. Find the waypoint where policy, not weather, decides.

**Change the ingredients, keep the receipt.** Activations, learning-rate abuse,
mid-run loss swaps — then **⧉ share setup** to copy a link that reproduces the
experiment, and **save net / load net** to keep the trained weights themselves.

**Beyond this demo.** Explore the
[other convnetjs examples](https://cs.stanford.edu/people/karpathy/convnetjs/) —
including a full convolutional network classifying the real
[CIFAR-10 images](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)
(heavy on slower machines) — or read the
[library docs](https://cs.stanford.edu/people/karpathy/convnetjs/docs.html).

---

## Delve inside — one training pass, up close

All of these use **single pass** mode. Every number it shows is captured from one
real batch-size-1 training step, not a simulation.

**Anatomy of one pass.** Click a PIREP and **tag it to trace**, then flip the mode
chip to **single pass**. The feature space steps aside and the network diagram
becomes a stage. Press **Run**: sample → forward → loss → backward → update, each
phase chip lighting up as it happens. Pause and use **−1 / +1 step** to walk the
phases at your own pace, in either direction. Esc returns to epoch mode.

**Watch a mistake get corrected.** Tag a PIREP sitting in the wrong-colored region
and run a pass. At **loss**, the true class's softmax box rings gold and the
caption spells out the selected loss — with the default cross-entropy,
loss = −log P(truth), large when the net is confidently wrong.
Then watch **update**: thick orange slashes the weights that fed the wrong answer
through the active neurons, while thick blue reinforces the path toward the right
one. That is the whole learning rule, drawn.

**Blame only flows through live neurons.** During **backward**, each neuron's
violet wash is its real share of the error — brighter means more responsibility.
Find a relu box showing 0.0: that neuron is "dead" for this report, so no blame
reaches it, and in the update phase its weights barely move. Learning flows only
through neurons that actually fired.

**Loss shrinks pass by pass.** Let the run loop and watch the loss in the narration
drop with each cycle — the same PIREP, a little easier to grade every epoch. Raise
the learning rate (in epoch mode) and repeat: bigger weight steps in the update
phase, faster drop, less stability. **undo pass** rewinds any cycle you want to see
again.

**Confident reports barely teach.** Tag a PIREP deep inside a correctly-colored
region and run a pass: P(truth) ≈ 1, loss ≈ 0, and in **update** the connections
barely glow — almost nothing changes. Gradient descent spends its effort where it
is wrong, which is why the reports near the boundary are the ones that shape the
net.

---

## Decide — the inference layout as a flight-deck decision

**The call.** The rule from the flight deck: assuming fuel isn't an issue, a
passenger flight takes the fuel cost of a diversion for **moderate chop or worse
(grade 4–6)**, but rides out **moderate turbulence or less (grade 1–3)**. The net
turns your waypoint into two probabilities, and the forecast readout thresholds
them into exactly that call: **stay on plan** when P(ok) wins, **divert** when
P(rough) does. A classifier's output isn't trivia — it exists to be turned into a
decision by someone with something at stake.

**Reading the screen.** **Next waypoint** (left panel) is one point on your route
ahead — click anywhere to move it; the gold crosshair is where you're asking about.
The **Waypoint** readout gives its coordinates, and **Forecast** gives
P(ok) · P(rough) and the call, live as you move. **Run** replays the question in
slow motion — the same forward pass a deployed model runs, with the real numbers
printed at every neuron. **Pilot view / ATC view** toggles between what a crew
actually has (the reports) and the net's full forecast map.

**Flying the route.** Train first — the forecast is only as good as the PIREPs it
learned from. Then flip to inference and walk your waypoint along the route you'd
actually fly. Where P(rough) spikes, probe sideways: in ATC view the divert answer
is visible as a corridor of blue. Try the **cyclone** weather and you'll find the
honest answer is sometimes not "go around everything" but "thread the gap between
the bands" — and sometimes, on **scattered** days, the net has nothing trustworthy
to say at all.

---

## Origins

This is a re-imagining of Andrej Karpathy's
[classify2d demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
for the [convnetjs](https://cs.stanford.edu/people/karpathy/convnetjs/) library.
The convnetjs engine (`js/convnet/convnet.js`) is included unmodified and does all
the actual training and inference; see the
[convnetjs documentation](https://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
for how nets are defined and trained.

## Running the example

Download the [repo](https://github.com/mandpd/nn-example) and open
`nn_example.html` in a web browser. There are no build steps and no external
dependencies — it works offline.
