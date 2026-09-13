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

This is real signal, not a bug — and it comes from network architecture,
not from which physical "side" of the brain it is.

**Sensory neurons run hot almost by definition.** They receive the raw
input current directly, every single frame. Everything else (motor +
interneurons) only lights up if signal manages to propagate to it within
just 4 internal hops (`sim_steps=4`) before the decision is made. So
brightness roughly tracks "how many synaptic hops from the input."
Measured over 150 live frames: **14 of the 20 most consistently-active
neurons were sensory**, only 3 motor and 3 interneuron.

This effect is real but *moderate*, not extreme: the hottest 5% of
neurons hold about **10% of total activation mass** (not a single
neuron/hub monopolizing everything), and the "hot set" isn't frozen either
— frame to frame it overlaps only ~50% with itself, meaning there's a
consistent hot core plus a shifting situational set that responds to the
actual game state.

Important: this hop-distance effect determines *which* neurons are
bright, but does **not** by itself explain spatial clustering — measured
correlation between activation strength and 3D position was ~0 on every
axis, and hot neurons are spread across the same volume as everything
else, not bunched together.

## So why does the bright region stay in one place even when you rotate the view?

Because it's real 3D geometry, not a camera/projection illusion (a
projection artifact would break apart or move as you rotate — it doesn't).

Checking the actual rendered points (not just each neuron's single
anatomical coordinate, but every point along its full traced branching
arbor) explains it: **individual neurons' branches reach far outside their
own central position** (the branch-point spread is roughly 2-3x wider
than the spread of neurons' central coordinates). And critically: **75 of
the 400 rendered neurons** all have branches passing through the same
single dense region.

That's not an artifact — that's what a real neuropil looks like. In real
insect brains, a neuron's cell body typically sits off to the side,
connected by a single thin fiber to where it actually does its synaptic
work: a dense hub where many *different* neurons' branches converge and
overlap. What's rendered is 75 real, individually-traced fly neurons
genuinely overlapping in the same real anatomical volume.

## Is the clustering caused by only using 20,000 (or 400) neurons instead of the full ~166,700?

**No — the hub itself isn't created by subsampling.** It's a property of
each individual real traced neuron's actual shape (soma-far-from-arbor
anatomy), which would be exactly the same with the full connectome.

**But subsampling does exaggerate the *contrast*.** With only 400 of
~166,700 real neurons shown, the sparse peripheral regions (long,
single-fiber projections reaching outward) have much less material to
fill them in, so they look thinner/sparser by comparison than they would
with the full population — while the naturally-dense hub stays
comparatively rich even at 400. So: real biology causes the hub, but
neuron count affects how stark the hub-vs-periphery contrast looks.

## Quick summary if someone asks in one sentence

"The bright cluster is a real fly neuropil — a hub where many different
real, individually-traced neurons' branches converge, seen because
sensory-input neurons (which are always driven directly) plus their
few-hop neighbors light up while distant neurons don't; showing fewer
neurons makes the sparse surrounding regions look emptier by comparison,
but doesn't create the hub itself."
