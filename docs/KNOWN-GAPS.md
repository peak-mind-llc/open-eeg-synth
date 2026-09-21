# Known gaps and room for improvement

What this package does not do well yet, and what we know is wrong or thin. It is a living list: add a
line when a review, a measurement or a consumer finds something, and remove a line when it is fixed
(say in which version). `docs/DESIGN.md` describes the package as built; this file is the honest
counterweight to it. Numbers here were measured on 0.2.0 unless a line says otherwise.

## 1. How real the signal is

- **The realism test's seeds were used to tune the recipe.** The 24 seeds it measures were part of the
  tuning objective, so its pass is not independent evidence. On fresh seeds, roughly a quarter to a
  half of 24-seed sets pass every line outside the two named gaps. Improving this means tuning against
  one set of seeds and gating on another.
- **Two named gaps.** Eyes-closed Laplacian theta and eyes-open bipolar beta are each too high or too
  low in the nearest-distance bin, and the test excuses them by name. Eyes-closed Laplacian beta is the
  next line at risk; it passes by about 0.001 on one fresh pool.
- **Eyes-closed theta dominance is not gated.** Frontal-midline theta leads in about 62 % of seeds with
  the eyes closed, against a gate of 60 % for eyes open. A louder or more focal network can push a
  temporal channel to the top, so a future recipe change should re-check it.
- **A left-right bias from the template head.** Right posterior alpha sits about 0.7-0.9 dB below left,
  and the sensorimotor rhythm is slightly asymmetric, because the template's C3 is 8.6 mm off C4's
  mirror. Real recordings vary, so this is not obviously wrong, but it is systematic and unmodelled.
- **Long rhythm lags scramble phase rather than model conduction.** Beta at 80 ms and alpha at 50 ms
  spread patches across most of a cycle. The measured cost is a lower left-right occipital alpha
  correlation (about 0.68, negative in 1 of 240 seeds). A physiological lag model would be better.
- **The realism reference is 20 subjects of one public dataset** (PhysioNet eegmmidb), cleaned with ICA
  blink removal. Two subjects drift under newer MNE and scikit-learn versions. A second, independent
  reference would make the bands trustworthy.

## 2. Artifacts and scalp patterns

- **Channels missing from the eye-map file are guessed.** The fit uses trustworthy neighbours and never
  exceeds the map's peak, but on unusual montages it can still be off: up to about 3.4x for eye
  movement when only one distant anchor is usable, and worse on non-10-20 layouts (a 128-channel
  product layout, a spherical head, or electrodes moved 5 mm). The 19-channel engine never hits this,
  because all 19 channels are in the file.
- **The clip at the map's peak understates far-lateral eye movement.** F9 and F10 would read about 1.1
  times the peak; they are held at 1.0. A wider map, or eye dipoles in the head model, would fix it
  properly.
- **Truth channel lists can look lopsided.** A blink lists F7 without F8 for some subjects, because the
  measured map is asymmetric and the cut is a fixed 0.3 of the peak.
- **Artifact size does not vary between people.** Every subject blinks at the same typical size (about
  120 µV at Fp1), so no case has a strong blinker or someone who barely blinks; the same holds for jaw
  tension. Only blink-to-blink variation exists. The fix is a subject-level size drawn from a measured
  spread: the script that derived the eye maps already finds every blink per subject in the public
  recordings, so the between-person spread can come from data.
- **Only blinks and eye movements vary per subject.** Jaw tension uses one analytic shape with no
  between-subject variation, and the dead channel has none by nature.
- **The head model has no eye or muscle sources.** Both come from measured scalp maps and analytic
  shapes bolted on, which is why the two points above exist.
- **Missing artifact types.** Loose-lead coordination, bridged pairs, movement, heartbeat bleed and
  pulse remain planned. Electrode pop, contact noise, noisy channel, mains hum and sweat drift now
  exist, using explicitly labelled engineering ranges that still need calibration against reviewed
  recordings.

## 3. Engine, scheduler and identity

- **The fingerprint does not cover eyes-closed artifact rates.** A 20 s eyes-closed recording draws no
  eye movement or jaw tension, so zeroing those two rates changes nothing it checks. Everything else,
  including each eyes-open artifact, is covered.
- **The exclusive-artifact scheduler is quadratic** in the number of busy spans it still holds, and its
  pending queue is unbounded if a plug-in's rate times its event length exceeds one. Pruning keeps
  ordinary runs small, but a pathological plug-in can still grow both.
- **Re-attaching an artifact to a scheduler that has pruned another artifact's spans is refused.**
  That is deliberate, because replaying from zero would overlap peers, but it is a real restriction on
  long-running hosts.
- **An exclusive plug-in that is bound but never rendered still consumes schedule.** Not reachable
  through the engine or the stream today.
- **Some errors are still the wrong type.** An unknown target rhythm in a plant modifier raises a
  KeyError where the rest of the package raises ValueError.
- **Spec identity has one wart.** `True` and `1` are different specs, although numbers now compare as
  floats. Adding extra sites to a widespread-excess plant also shifts the base rhythm's per-patch
  offsets and lags for its original patches.
- **The delayed network starts from silence.** The first 47 ms or so carry no delayed coupling.
- **The heart signal meets a looser chunking tolerance than the EEG layers** (1e-3 µV against 1e-4).
- **An artifact's recorded end can fall past the end of the recording.** Consumers clip it.

## 4. Output and consumers

- **Only a 19-channel head model ships.** A 31- or 37-channel export would let 10-10 labels be
  modelled; today a stream gives them amplifier noise only.
- **Responses to markers are not generated.** The stream carries marker schedules, but no evoked
  response follows them. Event-related practice needs this.
- **Whether the head perturbation defeats the "inverse crime" is unmeasured.** Cases are generated
  through a perturbed lead field so a source analysis cannot localise planted sources perfectly, but
  nobody has compared dipole-fit residuals on synthetic against real recordings.
- **Case files are whole seconds only**, because EDF records are.
- **Life-stage profiles are deliberately absent.** Clinical thresholds in the consuming application are
  adult-tuned, so child or older-adult profiles would mislead.

## 5. Build and platform

- **The fast test suite runs about 62-66 seconds** against a 60 second target, and the heaviest tests
  each render a whole case.
- **On macOS with numpy older than 2.3, which is the last numpy for Python 3.10, Accelerate raises
  hundreds of spurious matrix warnings** from this package's lines. The samples are identical. The
  README says how to filter them, and the package does not suppress them itself.
- **Reproducibility across numpy and scipy versions is checked only by CI.** The fingerprint holds on
  Python 3.10, 3.11 and 3.12 in CI, and on one developer machine. No older-numpy matrix runs.
- **`docs/PLAN.md` is a long historical record**, not a description of the package. Read
  `docs/DESIGN.md` instead; the plan is kept because it shows how each decision was reached.

## 6. Growing the package: extension and documentation

This package is expected to grow a lot: many more artifacts, and more planted patterns. The
extension points exist, but the on-ramp for someone who has not read the whole design does not.
These are the planned pieces of work, roughly in the order they pay off.

- **A guide for writing a plug-in.** `docs/DESIGN.md` §5.1 states the contract, but there is no page
  that walks from an empty file to a working artifact: what to subclass, how to draw a scalp pattern,
  what each truth field means, which remedies to declare, how to register it in this repo, and how to
  ship one from another package through the entry point. One complete worked example, an electrode
  pop in a few dozen lines, would carry most of it.
- **Contact-state coordination.** The continuous-artifact base and the first continuous acquisition
  artifacts now exist. A future `ContactTimeline` should coordinate impedance, loose-lead drift,
  pops and the recording application's impedance display rather than drawing each independently.
- **A public test kit.** The invariants every plug-in must meet are tested here but not offered to
  outside authors: identical output whatever the chunk size, a truth record that matches the rendered
  layer, amplitudes that do not depend on which channels were requested, and seeded reproducibility.
  A small `open_eeg_synth.testing` module with those checks would let an author prove a new plug-in in
  a few lines.
- **Planted patterns cannot be discovered from another package.** Artifacts load through an entry
  point; patterns register only in-process, so a case file naming a third-party pattern fails to load.
  Mirroring the artifact discovery would close that.
- **Guidance on when a new pattern needs new code.** Most patterns should be composed from the six
  primitives, as the first consumer does for all 25 of its patterns. The contributing notes should say
  so, and say what justifies a seventh primitive.
- **An API reference.** The README shows the common path. There is no listing of the public names with
  their arguments, so today the source is the reference.
