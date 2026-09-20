# Nervous System Visualization — FAQ

Notes on the live "brain" panel in the web platform (`server/`, `web/`),
for explaining what's real, what's stylized, and why it looks the way it
does. Findings below are backed by direct measurements on the actual
trained checkpoint and real traced neuron data — not guesses.

## Is the center panel showing real computation, or is it a canned animation?

Real, live computation — recomputed every single frame, not a replay.

- **Dot/line positions**: real anatomical coordinates and real traced
  skeletons for actual neurons from the MCNS connectome dataset (fetched
  from Janelia's public GCS bucket).
- **Brightness**: the *actual* per-neuron membrane activation computed by
  the trained model's forward pass on that exact game frame — the true
  internal state of the network reacting to the current situation.
- **Synaptic wires** (in the simpler live-training dashboard variant): real
  trained synapse weights, redrawn each frame between whichever neurons are
  currently most active and their real strongest partners.

What's cosmetic, not biological fact: the translucent hull shapes (a
convex-hull outline standing in for a brain-surface mesh we don't have
data for), the auto-rotate camera motion, and the bloom/glow shader.

## Does the model "see" the game as pixels?

No. The connectome's input is a 12-number physics vector, not an image:
the horizontal distance and top/bottom gap edges for the next three pipes
(9 numbers), plus the bird's own y-position, vertical velocity, and
rotation (3 numbers). The rendered game frame shown on screen is purely
for the human viewer — the network never touches pixel data.

## Do neurons stay "on" after firing, or does activity persist?

Underlying model: **no persistent memory between frames.** Every game
step, the membrane state starts at zero, the current 12-number reading is
injected, it propagates through the sparse synapse graph for 4 internal
ticks (`sim_steps=4`), an action is read out — then that state is
discarded. The next frame starts from zero again. There is no carried-over
"thought" between frames; it's a fresh reflex computation each time
(~every 33ms).

The continuous-looking glow in the UI is a deliberate **display-only**
smoothing effect (rise instantly, decay ~18%/frame) so activity doesn't
strobe distractingly at the true instantaneous rate. A toggle
("Smoothed glow: ON/OFF") switches to the raw, undamped values if you want
to see literally what the network computed that exact frame.

## Why does one region look so much brighter than the rest?

This has two separate causes — one is real network behavior, one **was**
a sampling artifact on our end that has since been fixed. Being honest
about both is important.

### Cause 1 (real): sensory neurons run hot almost by definition

They receive the raw input current directly, every single frame.
Everything else (motor + interneurons) only lights up if signal manages
to propagate to it within just 4 internal hops (`sim_steps=4`) before the
decision is made. So brightness roughly tracks "how many synaptic hops
from the input." Measured over 150 live frames: **14 of the 20 most
consistently-active neurons were sensory**, only 3 motor and 3 interneuron.

This effect is real but *moderate*: the hottest 5% of neurons hold about
**10% of total activation mass** (not a single hub monopolizing
everything), and the "hot set" isn't frozen — frame to frame it overlaps
only ~50% with itself, meaning a consistent hot core plus a shifting
situational set that responds to the actual game state.

### Cause 2 (known limitation): sampling bias in the training pool

The real fly brain is bilaterally symmetric — well-established fact,
independently verifiable (e.g. the Nature MCNS paper reports <0.4%
asymmetry across the central brain). So a lopsided-looking render is
**not** the biology; it's something we introduced.

The cause is on our end: we built the 20,000-neuron training subgraph by
"snowball" sampling — BFS-expanding outward from a **single random seed
neuron** through real synapses. That preserves realistic local synapse
density (why we did it that way), but it biases the pool toward whichever
hemisphere the seed happened to sit in.

Measured directly on the loaded checkpoint:
- Soma-position skew on the x-axis (median offset from mid-range,
  normalised by range) = **0.14** — pool median sits well to one side
  instead of near the anatomical midline.
- Actual rendered arbor-vertex skew (what the browser draws) = **0.21 on
  x, 0.32 on z** — the neurons' branches extend outward even more
  asymmetrically than their somas suggest.

**This cannot be fixed at the display layer.** An earlier attempt
tried to rebalance by picking display neurons more evenly across x —
that produced a symmetric selection of somas, but the render was still
lopsided because the underlying pool of trained neurons genuinely has
its arbors reaching further in one direction. You can't invent trained
neurons on the missing side that were never in the training subgraph
to begin with.

**FIXED (2026-09-15) in `src/connectome/loader.py::_snowball_sample_bilateral`.**
The real dataset has a populated per-neuron `Soma side` column (81,199
right / 79,102 left / 392 center — genuinely balanced, real anatomical
metadata, not derived/fabricated). The sampler now grows two independent
snowballs — one seeded from a real "left"-labeled neuron, one from a real
"right"-labeled neuron, each restricted to its own hemisphere and
targeting half the neuron budget — instead of one snowball from a single
random seed.

Verified two ways after the fix, at n_neurons=20,000:
- Resulting subgraph: exactly **10,000 left / 10,000 right** by the real
  Soma-side label.
- Re-measured on a fresh random sample of 299 real fetched skeletons from
  that subgraph: **x-axis (left-right) skew dropped from 0.21 to 0.013**
  — a >15x improvement, now essentially symmetric.

**A correction to the framing above**: the z-axis skew (0.32) mentioned
earlier is *not* part of this bug and was never fixable this way — z
corresponds to the head-to-abdomen body axis (this dataset combines
brain + ventral nerve cord), and a nervous system running from head to
tail is *supposed* to be elongated along that axis. Bilateral symmetry
is specifically a left-right property; there's no biological reason to
expect head-tail symmetry. Conflating the two earlier was a mistake.

This fix has been validated but **not yet retrained** — it only takes
effect on the *next* training run (the currently-deployed checkpoint
still reflects the old, single-seed sample). Retraining takes about the
same wall-clock time as before (~2 hours at the 20,000-neuron budget on
a rented GPU).

## Why does a dense cluster still remain in specific places (even after the L/R fix)?

Genuine biology — this part isn't an artifact and doesn't depend on
subsampling either.

Checking the actual rendered points (not just each neuron's single
anatomical coordinate, but every point along its full traced branching
arbor) explains it: **individual neurons' branches reach far outside
their own central position** (the branch-point spread is roughly 2-3x
wider than the spread of neurons' central coordinates). And critically:
**75 of the 400 rendered neurons** (measured before we upgraded to 2,000)
all have branches passing through the same single dense region.

That's what a real neuropil looks like. In real insect brains, a neuron's
cell body typically sits off to the side, connected by a single thin
fiber to where it actually does its synaptic work: a dense hub where many
*different* neurons' branches converge and overlap. What's rendered is
dozens of real, individually-traced fly neurons genuinely overlapping in
the same real anatomical volume.

## Is the clustering caused by using only 20,000 (or 2,000) neurons instead of the full ~166,700?

The dense hubs themselves aren't caused by subsampling — they'd be there
in the full connectome too, since they're a property of each individual
real traced neuron's actual shape (soma-far-from-arbor anatomy).

Subsampling *does* affect two things: (1) it **exaggerated the L/R
imbalance** described above until we fixed the picker, and (2) it makes
sparse peripheral regions look thinner/emptier than they would with the
full population (fewer neurons contributing long single-fiber projections
outward), so the hub-vs-periphery contrast is somewhat starker at 2,000
than it would be at 166,700 — but the hubs are real either way.

## Quick summary if someone asks in one sentence

"The dense hubs are real fly neuropils — anatomical regions where many
different individually-traced neurons' branches converge — plus sensory
neurons (which are always driven directly by the input) glow more than
distant neurons in this shallow 4-hop network; and there's a known
sampling artifact making one hemisphere look emptier than the other
(caused by seeding our 20K-neuron training subgraph from a single random
neuron instead of two bilateral ones), which needs retraining to fix
properly."
