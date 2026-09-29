# Brag Plan: Model-Distillery — whiteboard lecture

## What is this app?
Model-Distillery is an end-to-end distillation pipeline: it has a big expensive
"teacher" model write a synthetic SFT textbook, cleans it (MinHash dedup,
quality gates, rejection sampling, 13-gram decontamination), plans and costs a
QLoRA training run, benchmarks the small "student" against the teacher per
skill, and computes the breakeven math. Impressive claim: fully offline,
byte-identical runs, 114 tests in ~4.5s — and a $60 one-time investment that
pays back in ~3 days at 100k requests/month.

## The angle
NOT a launch video. A patient senior engineer at a whiteboard teaching one
system to a smart junior who knows almost nothing about AI. The lecture
follows the repo's own spine: the surgeon's-hourly-rate-to-ask-directions
scenario → generate the textbook → dedup/filter/rejection → decontamination →
QLoRA in one sentence → the exam (student vs master per skill) → breakeven
math → the measured numbers (122 → 59 kept → 68.6% retention, byte-identical
runs, hand-computed KD fixture) → 30-second recap. Every technical keyword is
defined on screen in one plain sentence the moment it first appears.

## Hook (first 2-3 seconds)
A whiteboard with a hand-drawn surgeon's fee schedule: "World-class surgeon:
$4,000/hr — question: how do I get to the train station?" Narration: "You
would not pay a surgeon's hourly rate to ask for directions. Yet that is
exactly what most software does every day."

## Key moments (the middle)
- The funnel counts on the board: 122 candidates → 13 dedup + 8 repetition +
  42 rejection → 59 kept → 1 leak caught → 46/12 split. Numbers appear as the
  stages are taught.
- The dedup sketch: an answer with a fingerprint card, a near-twin detected,
  "Jaccard ≥ 0.8", clusters collapsing via union-find.
- The exam table: student 0.229 vs teacher 0.333 token-F1 — and the slices
  where the student BEATS the master (classification 0.270 vs 0.259,
  extraction 0.329 vs 0.303).
- The ledger: $645/month teacher bill vs $28.50 student bill; $60 investment;
  breakeven ~3 days, drawn as a crossing line on a cost curve.
- The KD fixture: the loss formula T²·KL + α·CE written on the board with
  "validated against a hand-computed example to 1e-9".

## Outro / punchline
The 30-second recap the viewer could repeat to a colleague, then the board
wipes to the repo name and "measurable, auditable, byte-identical."

## User flow worth showing
This is a pipeline/CLI project, not a GUI. The "flow" is the pipeline itself:
prompt pool (61 prompts) → teacher grid (122 candidates) → curation funnel
(59 kept) → split (46/12) → training plan (9 steps, 22,386 tokens) → bench
(68.6% retention) → cost report (breakeven 0.10 months). Real artifact numbers
from `artifacts/final/summary.json`, `bench_report.md`, `cost_report.md`,
`training_plan.md` appear as the whiteboard fills.

## Tone
- Preset: polished (nearest to the brief's "calm, precise, friendly" lecture)
- Creative direction (the --tone contract, verbatim): "A patient senior
  engineer at a whiteboard teaching ONE system to a smart junior who knows
  almost nothing about AI. Whiteboard explainer, NOT a product demo: no hype
  adjectives, no 'lightning-fast', no launch energy. Long-form lecture pacing
  (aim 5+ minutes; take the time each idea needs). Every technical keyword
  that appears must be defined on screen in one plain sentence the moment it
  first appears. Structure: (1) the real-world problem with a concrete
  everyday scenario, (2) the core idea explained with a simple analogy,
  (3) how the pieces work — one whiteboard sketch per concept, (4) the
  measured numbers from the repo's tests and what each number means in plain
  words, (5) a 30-second recap the viewer could repeat to a colleague.
  Narration on. Calm, precise, friendly. The viewer should finish able to
  explain the system to someone else."
- Interpretation: long holds, hand-drawn ink sketches, definition cards that
  stay put; transitions are quiet wipes, not slams; SFX restrained to soft
  pen/drop accents; music barely present under the voice.

## Format: landscape — 1920x1080
## Duration: ~360-420 seconds (voice-driven; scene durations flex to narration)

## Visual identity (whiteboard explainer, not from a website)
- Background: #F6F2E8 (warm whiteboard / paper)
- Ink (text): #23272E (dark marker)
- Accent blue: #1D4ED8 (headings, arrows, structure)
- Accent green: #15803D (kept/saved numbers)
- Accent red-orange: #C2410C (discards, leaks, warnings)
- Display font: "Segoe Print" (marker feel), fallback "Comic Sans MS", cursive
- Body font: "Segoe UI", Arial, sans-serif for dense definition sentences
- Strongest visual element: hand-drawn whiteboard sketches (SVG) + definition
  cards ("keyword — one plain sentence") pinned under each sketch

## Share copy (draft)
"Most questions don't need the surgeon — but the junior needs training.
Model-Distillery: the whole distillation pipeline on one whiteboard, from
textbook generation to breakeven math. 122 candidates → 59 kept → 68.6%
retention, byte-identical runs."

## Audio direction
- Role: narration-forward lecture; music is a barely-there bed
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3 (steady, clean)
- Music treatment: starts at 0s, volume 0.13 constant — the voice runs the
  entire video so the bed stays ducked at 0.12-0.15 throughout; fades out over
  the last 4s
- Music cue guidance: bundled preset available at
  <skill-dir>/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json;
  beat sync is intentionally NOT used for text reveals — lecture pacing and
  reading time win; natural timing chosen for readability throughout
- Audio-reactive treatment: none — a still whiteboard suits the tone; skipped
  deliberately for calm (documented here rather than forced)
- SFX posture: sparse; soft drop on definition cards, soft impact on section
  headers, none during dense narration
- Audio-coupled moments: definition card arrivals (soft drop), counter/settled
  numbers (single soft tick or drop), scene wipes (soft whoosh/impact at most)
- Restraint rule: the voice is the product; nothing may compete with it. No
  SFX stack on every card, no music swells over narration.

## Voiceover script

Voice: af_heart via `npx hyperframes tts`. One WAV per scene; each scene's
`data-duration` is set from its WAV length (voice sets the pace).

### Scene 1 — The problem (hook)
You would not pay a world-class surgeon's hourly rate to ask for directions to
the train station. Yet that is exactly what most software does every day. The
biggest AI models are brilliant, and shockingly expensive, and most questions
do not need the surgeon. But swap in a cheap junior, and quality collapses.
Unless the junior was trained by watching the surgeon work. That training
program is called distillation. The expensive model is the teacher. The small
model you own is the student. This is Model Distillery: a pipeline that runs
that program end to end, and measures every step of it.

### Scene 2 — The core idea
Here is the whole idea on one whiteboard. Train a junior chef by watching the
master cook. The master writes thousands of worked answers to real questions.
Those answers become a textbook. The apprentice studies the textbook, then
takes the same exam the master takes. And then we do the math: does the
cheaper apprentice keep enough of the master's skill to pay for the training?
That is the pipeline. Nine stages. Let's walk them.

### Scene 3 — Generate the textbook
Stage one: generate the textbook. We start with a prompt pool: sixty-one demo
prompts spread across six skills, like classification, coding, extraction, and
reasoning. That is the curriculum. The teacher answers each prompt through a
persona grid: different personas, different temperatures, different seeds, two
candidates per prompt. One hundred and twenty-two in total. Temperature is the
randomness dial. A seed is fixed randomness, so every run is reproducible.
Each candidate is a supervised fine-tuning example: prompt in, answer out,
written as chat JSONL, one record per line. And when reasoning matters, the
teacher shows its working. Chain of thought. In a textbook, the steps are the
lesson.

### Scene 4 — Remove the junk: dedup
Stage two: remove the junk. Near-identical answers slip in; the same winning
phrasing across seeds. Training on one lesson twenty times biases the student.
Exact hashing cannot catch these: they are almost identical, not identical. So
we fingerprint. MinHash gives every answer a signature, so similar texts get
similar fingerprints. Locality-sensitive hashing buckets the fingerprints so
the similar ones meet. Every candidate pair is then verified with Jaccard
similarity: shared pieces over total pieces, threshold zero point eight. And
union-find collapses each cluster of duplicates down to one canonical
survivor. In the demo run, thirteen candidates removed.

### Scene 5 — Quality gates: filter and rejection sampling
Stage three: quality gates. A rule filter first: too short, too long, repeated
lines, wrong language. Deterministic, and free. Then rejection sampling: for
each question, keep the best of the k candidates and discard the rest. Forty
two more gone. Everything removed is logged in a discard histogram: a receipt
for every deletion. And that log matters, because best-of-k optimizes the
judge, and Goodhart's law warns: a measure you optimize against stops
measuring. So the rules gate validity, and the judge only ranks. After both
gates: fifty-nine examples kept, out of one hundred and twenty-two.

### Scene 6 — No exam leaks: decontamination
Stage four, the exam-integrity gate. Decontamination. A distillation project
dies quietly when exam questions leak into the textbook: the student scores
well by memorizing, not learning. The screen is brutal and simple. Any
thirteen-token span shared with the eval set disqualifies the example.
Thirteen tokens is long enough that a match is never chance. It runs before
the split, and it caught one planted overlap in the demo run. Then a
stratified train-validation split: forty-six to study, twelve held back for
the exam, every skill represented in both.

### Scene 7 — Train the apprentice: QLoRA and the KD loss
Stage five: train the apprentice. QLoRA in one sentence: train a small adapter
on a compressed base model, so one GPU suffices. The base model is frozen at
four bits, and a LoRA adapter, a quarter of one percent of the parameters,
carries all the learning. The plan generator turns the dataset into steps,
memory, and GPU hours before you rent anything. And there is a second signal
while the student learns: the distillation loss. Soften the master's grades
with a temperature, and teach the student to match the whole distribution, the
soft labels, not just the right answer. That distance between beliefs is KL
divergence. And this loss is not hand-waved: it is validated against a
hand-computed example, checked to nine decimal places.

### Scene 8 — The exam: student vs master
Stage six: the exam. Student versus master, per skill. Exact match is the
brutal one: one if identical, else zero. Both score zero: open-ended answers
never match verbatim. So we use token F1, partial credit that punishes both
bluffing and omissions, and containment: how much of the reference the answer
covers. The slices tell the real story. The student beats the master on
classification and extraction, and lags on summarization. Overall: sixty-eight
point six percent retention. And the bench runs on a pipeline that is
deterministic, byte-identical across runs, covered by one hundred and fourteen
tests that pass offline in under five seconds.

### Scene 9 — The math that pays for it
Stage seven: does it pay? Cost per million tokens: three dollars in and
fifteen out for the teacher; fifteen cents in and sixty cents out for the
student. At one hundred thousand requests a month, the teacher bill is six
hundred forty-five dollars. The student: twenty-eight dollars and fifty cents.
Savings: six hundred sixteen a month. The one-time investment, generation,
training, and eval, is sixty dollars. Breakeven: about three days. And because
savings scale with volume while the investment stays fixed, more traffic only
shortens the payback. The only real question is whether sixty-eight point six
percent retention is enough for your workload.

### Scene 10 — Recap
The thirty-second recap. Repeat this to a colleague. A frontier model is a
surgeon's rate for directions. Distillation is the training program. The
teacher writes a textbook: a persona grid over a prompt pool. MinHash
fingerprints remove near-duplicates. Rules and rejection sampling keep only
the best answer per question. Thirteen-gram decontamination keeps the exam out
of the textbook. QLoRA trains a small adapter on a compressed base model: one
GPU. The exam grades the student against the master, per skill: sixty-eight
point six percent retention. And the ledger: six hundred sixteen dollars saved
a month; breakeven in three days. Measurable, auditable, byte-identical. That
is Model Distillery.

## Storyboard

### Scene 1 — The problem — ~45s (voice-driven)
Whiteboard. Hand-drawn fee schedule card: "World-class surgeon — $4,000/hr";
below, a small speech bubble: "How do I get to the train station?" Definition
cards pin in as named: **distillation** ("training a small, cheap model to
behave like a large, expensive one by learning from its outputs"),
**teacher model** ("the big, expensive model whose answers you copy"),
**student model** ("the small model you own, trained by imitating the
teacher"). Title "Model-Distillery" underlined at the end.
Sequential/interaction: yes — fee card, bubble, then three definition cards
arrive one by one, each holding long enough to read.
Audio intent: calm, conversational; the bed barely audible.
Audio-coupled idea: soft drop on each definition card.
Transition mood: soft wipe → Scene 2.

### Scene 2 — The core idea — ~28s
Whiteboard sketch: MASTER → (writes answers) → TEXTBOOK → STUDENT (studies)
→ EXAM (side-by-side). Underline "train a junior chef by watching the master
cook". A small pipeline map appears: 9 numbered stage dots along an arrow.
Sequential/interaction: sketch elements draw in one by one with the narration.
Audio intent: steady, teaching rhythm.
Audio-coupled idea: soft drop per pipeline-map stage dot.
Transition mood: soft wipe → Scene 3.

### Scene 3 — Generate the textbook — ~45s
Left: "prompt pool" jar sketch labeled 61 prompts / 6 skills. Middle: teacher
figure. Right: persona grid (persona × temperature × seed) fanning into 122
candidate sheets (k=2). Definition cards: **prompt pool**, **persona grid**,
**temperature**, **seed**, **SFT**, **chat JSONL**, **chain-of-thought**.
Counter ticks to 122.
Sequential/interaction: yes — definition cards arrive with their keyword;
counter counts 0→122.
Audio intent: bright, curious, unhurried.
Audio-coupled idea: counter ticks; card drops.
Transition mood: soft wipe → Scene 4.

### Scene 4 — Dedup — ~42s
Sketch: two near-identical answer sheets with fingerprint cards; LSH bucket;
"Jaccard ≥ 0.8" threshold note; duplicate cluster collapsing to one sheet.
Definition cards: **near-duplicate detection**, **MinHash**, **LSH**,
**Jaccard similarity**, **union-find**. Tally: "−13".
Sequential/interaction: cluster collapse animation; tally stamps −13.
Audio intent: focused, detective-ish but calm.
Audio-coupled idea: soft drop per definition card; single stamp on −13.
Transition mood: soft wipe → Scene 5.

### Scene 5 — Filter + rejection — ~40s
Sketch: funnel. Top: 122. Rule gate (length/repetition/language). Then
best-of-k: k sheets, one circled "keep best". Definition cards: **quality
filter**, **rejection sampling**, **discard histogram**, **Goodhart's law**
(the warning card gets red-orange ink). Tally: "−8 rules · −42 rejection →
59 kept".
Sequential/interaction: funnel fills; tallies stamp.
Audio intent: measured; the Goodhart line gets a beat of quiet.
Audio-coupled idea: stamp on each tally.
Transition mood: soft wipe → Scene 6.

### Scene 6 — Decontamination — ~38s
Sketch: "TEXTBOOK" book; an eval paper with a highlighted 13-token span
trying to sneak in; a red-orange X and "leak caught: 1". Definition cards:
**eval contamination**, **13-gram decontamination**, **stratified split**.
Split boxes: 46 study / 12 exam, six skill chips balanced on both sides.
Sequential/interaction: span highlight draws; X stamps; split boxes fill.
Audio intent: serious but reassuring.
Audio-coupled idea: stamp on "leak caught".
Transition mood: soft wipe → Scene 7.

### Scene 7 — QLoRA + KD loss — ~48s
Sketch: big frozen block "base model · 4-bit" with a small warm block
"adapter r=16 (~0.25% params)" on top; "one GPU" caption. Formula card:
KD loss = T² · KL(student ‖ teacher) + α · CE. Definition cards: **QLoRA**,
**LoRA adapter**, **NF4**, **double quantization**, **logit**, **KD loss**,
**temperature T**, **soft labels**, **KL divergence**. Note: "plan: 9 steps ·
22,386 tokens · GPU-hours before you rent". Stamp: "hand-computed fixture ✓
(to 1e-9)".
Sequential/interaction: formula writes in term by term.
Audio intent: the densest scene; extra patience.
Audio-coupled idea: soft drop per card; quiet tick per formula term.
Transition mood: soft wipe → Scene 8.

### Scene 8 — The exam — ~44s
Sketch: two desks "student" / "master" with graded papers. Table sketch:
overall token-F1 0.229 vs 0.333; then slices — classification 0.270 vs 0.259
(green: student wins), extraction 0.329 vs 0.303 (green), summarization
0.285 vs 0.515 (master). Big circle: "retention 68.6%". Definition cards:
**exact match**, **token-F1**, **containment**, **capability slice**,
**quality retention**. Footer stamp: "114 tests · byte-identical runs".
Sequential/interaction: table rows reveal one by one (held readable).
Audio intent: honest, even-toned; slight warmth on the wins.
Audio-coupled idea: soft tick per row.
Transition mood: soft wipe → Scene 9.

### Scene 9 — Breakeven — ~42s
Sketch: cost curve — fixed "investment $60" horizontal line; savings line
rising with volume; crossing point stamped "breakeven ~3 days". Side ledger:
teacher $645/mo vs student $28.50/mo at 100k requests; savings $616.50/mo.
Definition cards: **cost per 1M tokens**, **breakeven horizon**,
**GPU-hours** (if not shown in Scene 7), **deterministic pipeline** moves to
Scene 8 footer instead. Green ink on savings.
Sequential/interaction: lines draw; crossing point stamps.
Audio intent: the payoff; still calm, no hype.
Audio-coupled idea: stamp on breakeven.
Transition mood: soft wipe → Scene 10.

### Scene 10 — Recap — ~46s
Numbered recap list writes in (5 lines), each a one-line takeaway with its
number: 122→59 kept · 68.6% retention · $616.50/mo saved · breakeven ~3 days
· byte-identical. End card: "Model-Distillery — measurable, auditable,
byte-identical" + "GLOSSARY.md: every keyword, defined."
Sequential/interaction: recap lines appear one by one, all held to the end.
Audio intent: gentle close; music fades under final line.
Audio-coupled idea: soft drop per line.
Transition mood: fade out.

**Music mood for this video:** steady, clean, barely-there bed (polished)
**Audio summary:** the voice carries the lecture start to finish; vol-12 sits
at 0.13 underneath and fades over the last 4 seconds; a dozen soft drops and
stamps mark definition cards and tallies; nothing ever competes with the
narration.
