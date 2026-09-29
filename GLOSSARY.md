# Model-Distillery — Glossary

Every keyword this repo uses, defined in plain sentences. Read it before the
video, after the video, or instead of asking me twice. Each entry ends with
*why it matters here* — because a definition you can't connect to the code is
just trivia.

---

## 1. Teacher & data — writing the textbook

**Distillation.** Training a small, cheap model to behave like a large,
expensive one by learning from its outputs. It's how you keep most of the
quality while shedding most of the cost.

*Why it matters here:* this repo is the whole distillation loop — generate the
teacher's answers, clean them, train the student, then prove the trade was
worth it.

**Teacher model.** The big, expensive model whose answers you want to copy —
brilliant, but billed like a specialist. In this repo it's any
OpenAI-compatible endpoint (vLLM, Ollama, OpenAI).

*Why it matters here:* every training example the student ever sees was
written by the teacher; teacher quality is the ceiling on student quality.

**Student model.** The small model you actually own and serve — cheap to run,
trained by imitating the teacher. Also called the apprentice.

*Why it matters here:* the entire pipeline exists so the student scores close
enough to the teacher to be worth deploying.

**Prompt pool.** The set of questions the teacher will answer — here, 61 demo
prompts spread across six capabilities (classification, coding, extraction,
reasoning, safety-refusal, summarization). The curriculum, before it's a
textbook.

*Why it matters here:* if the prompts don't look like real traffic, the
student learns the wrong lessons; the pool is the prompt distribution you're
amortizing.

**Persona grid.** A generation plan that walks `persona × temperature × seed`:
the teacher answers each prompt wearing different personas, at different
sampling temperatures, with different random seeds. One question, many
distinct lessons.

*Why it matters here:* with k candidates per prompt (k=2 in the demo), the
grid gives the rejection sampler choices — and diversity is what stops the
textbook from being one answer photocopied.

**Chain-of-thought (CoT).** Asking the model to write out its reasoning steps
before the final answer, instead of answering in one hop. Think "show your
working" on a math exam.

*Why it matters here:* `TeacherRunner` has a CoT mode, because for reasoning
prompts the steps in the textbook are the lesson, not just the conclusion.

**SFT (supervised fine-tuning).** Training on plain (prompt → answer) pairs
where the answer is the label — ordinary supervised learning, applied to chat
behavior. The most common way a base model becomes an assistant.

*Why it matters here:* the pipeline's spine is *data* distillation: it
produces an SFT dataset, and everything upstream is quality control for it.

**Chat JSONL.** A text file with one JSON object per line, each a chat record
(messages, roles, metadata). JSONL keeps a dataset appendable, greppable, and
stream-friendly.

*Why it matters here:* every artifact the pipeline writes — candidates,
discards, train, val — is chat JSONL, so any stage can be audited with `less`
and `grep`.

**Offline mock teacher.** A stand-in teacher (`EchoMockClient`) that is
seeded and deterministic: same prompt + same seed → same text, no GPU, no
network. It simulates a weaker student via a `fidelity` knob.

*Why it matters here:* it makes the whole pipeline runnable and testable
anywhere — 114 tests in ~4.5 seconds — while real runs swap in a real
endpoint with one config change.

**Quality filter.** The first gate on generated answers: deterministic rules
(too short, too long, repeated lines, wrong language) plus an optional judge
score. Cheap checks first, expensive opinions only for survivors.

*Why it matters here:* it's why the demo run keeps 59 of 122 candidates; the
discard log tells you exactly which rule threw out what.

**Near-duplicate detection.** Finding texts that are *almost* identical —
same lesson, lightly reworded — which exact-hash dedup can't see. You compare
fingerprints of content, not whole strings.

*Why it matters here:* in rejection-sampling pipelines the same winning answer
recurs across seeds, and training on it ten times biases the student toward
one phrasing.

**MinHash.** A fingerprinting trick: shuffle word shingles through many hash
functions and keep the minimums — two texts with similar sets get similar
signatures. Estimating set similarity without storing the sets.

*Why it matters here:* it's how the repo dedups in O(n) expected time,
catching the planted duplicate in the demo data with exact, auditable
precision.

**LSH (locality-sensitive hashing).** Hashing items so *similar* items land
in the same bucket — the opposite of a normal hash, where similarity is
destroyed. Banded LSH slices MinHash signatures into bands and buckets on
band matches.

*Why it matters here:* with 128 permutations in 16 bands × 8 rows, J≈0.8
duplicates collide ~95% of the time per pass — candidate finding without the
O(n²) all-pairs comparison.

**Jaccard similarity.** For two sets: |shared| / |total distinct|. Two answers
sharing 8 of 10 distinct shingles have Jaccard 0.8.

*Why it matters here:* it's the threshold a human reviewer can actually argue
about (this repo dedups at 0.8), verified exactly after LSH proposes
candidates.

**Union-find.** A data structure that tracks which items belong to the same
group, with near-constant-time `find` (who's my group's root?) and `union`
(merge these groups). Also called disjoint-set.

*Why it matters here:* after LSH buckets near-duplicates, union-find
collapses each duplicate cluster to one canonical survivor — earliest in
input order wins.

**Rejection sampling (best-of-k).** Generate k candidate answers per prompt,
score them, keep the best, discard the rest. Quality bought with generation
tokens, not with training noise.

*Why it matters here:* it removed 42 more candidates after the rules did —
and the discard histogram of *why* is the data-quality dashboard.

**Judge hook (JudgeHook).** The pluggable scoring interface that ranks
surviving candidates. The default is a lexical heuristic; you can swap in an
LLM judge without touching the pipeline.

*Why it matters here:* rules gate *validity*, the judge only *ranks* —
keeping that separation is what keeps the dataset auditable.

**Discard histogram.** The count of removals by reason — `rule:repetition`,
`rule:length`, `rejection_sampling`, and friends — written alongside the
dataset. Not a silent filter: a receipt for every deletion.

*Why it matters here:* if `rule:repetition` spikes after a teacher upgrade,
you know within one run; the demo run shows 13 dedup + 8 repetition + 42
rejection + 1 contamination discards.

**Goodhart's law.** "When a measure becomes a target, it ceases to be a good
measure." Optimize hard against one score and you'll maximize the score, not
the thing it stood for.

*Why it matters here:* best-of-k optimizes the judge, so a malleable judge
gets gamed — the honest fix is validating any judge swap against human labels.

**13-gram decontamination.** A leak screen: any 13-token span shared between
a training example and an eval case disqualifies the example. 13 is long
enough that a match is almost never chance, short enough to catch partial
rewording.

*Why it matters here:* it runs on prompt+response against every eval case
*before* the split, and it caught 1 planted overlap in the demo run — that's
the exam-integrity tripwire.

**Eval contamination.** The failure where exam questions (or near-copies)
end up in training data, so the student scores well by memorizing, not
learning. The quiet killer of distillation projects.

*Why it matters here:* the repo treats it as a first-class gate, not an
afterthought — a contaminated bench number is worse than no number.

**Train/val split.** Carving the dataset into a training portion (the student
studies it) and a validation portion (held back, used only to check
generalization). 46/12 in the demo run.

*Why it matters here:* the split runs *last* — after dedup and
decontamination — so the val set never contains a near-duplicate of a train
item.

**Stratified split.** A split that preserves the mix of classes (here,
capability labels) in both portions, instead of shuffling blindly.

*Why it matters here:* with six capability slices, a random split could
starve a slice of validation examples; stratification keeps every skill
measured.

---

## 2. Training — the apprentice year

**QLoRA.** A training recipe: freeze a 4-bit-quantized base model and train a
tiny set of LoRA adapters on top, so fine-tuning fits on one GPU. In one
sentence: *train a small adapter on a compressed base model so a single GPU
suffices.*

*Why it matters here:* the plan estimates a 7B student at ~13.8 GB VRAM —
where full fine-tuning would want ~112 GB or more.

**LoRA adapter.** A small pair of low-rank matrices added alongside a frozen
weight matrix; you train only these (r=16 on the q/k/v/o projections here —
about 0.25% of the parameters). The base model never changes.

*Why it matters here:* backprop and optimizer states exist only for the
adapters, which is the whole memory trick; you merge them back in before
shipping.

**NF4 quantization.** Storing base weights in 4-bit NormalFloat — a codebook
built from the quantiles of the normal distribution, which is what trained
weights actually look like. At 4 bits it beats uniform INT4 on perplexity.

*Why it matters here:* it's the "compressed base" half of QLoRA and the
reason the memory model uses ~0.55 bytes per parameter.

**Double quantization.** Quantizing the quantization constants themselves:
the FP32 per-block scales become 8-bit, with a second-level scale. Saves
~0.4 bits per parameter — roughly 3 GB on a 70B — at negligible error.

*Why it matters here:* the training plan ships with `double_quant=True`
because it's nearly free memory; the demo plan (22,386 tokens, 9 steps) fits
even smaller because of it.

**Gradient accumulation.** Computing gradients on several small
micro-batches and applying the update only after adding them up — simulating
a large batch on hardware that can't hold one. Effective batch = micro-batch
× accumulation steps.

*Why it matters here:* the demo plan's effective batch of 16 is really 2 × 8;
that's how the recipe stays inside a single-GPU envelope.

**Cosine learning rate.** A schedule where the learning rate decays along a
cosine curve — brisk early strides, careful small steps near the end —
usually with a short linear warmup at the start.

*Why it matters here:* the plan pins lr 2e-4, cosine, 3% warmup, seed 42, so
the training recipe is reproducible, not vibes.

**Epoch.** One full pass over the training dataset. Three epochs means the
student reads the whole textbook three times.

*Why it matters here:* the 46-example training set × 3 epochs is what sets
the 9 optimizer steps and the token estimate in the plan.

**Logit.** The raw, pre-softmax score a model assigns to each candidate next
token — big scores mean "likely", but they're not yet probabilities.

*Why it matters here:* logits are what logit-KD compares; if you own both
models you can learn from the teacher's full next-token distribution, not
just its chosen words.

**KD loss (knowledge distillation).** A training objective that pulls the
student's whole probability distribution toward the teacher's, not just
toward the correct label: `T² · KL(student ‖ teacher) + α · CE`. Here it's a
numpy reference implementation with a torch mirror.

*Why it matters here:* the loss is validated against a hand-computed
2-class × 2-token example — arithmetic in the test docstring, checked to
1e-9 — so the math is provably the math.

**KL divergence.** A measure of how one probability distribution differs
from another; zero when they're identical. In KD, it's the "distance" between
student and teacher beliefs, per token.

*Why it matters here:* it's the exact quantity the hand-computed fixture
computes — KL₁ + KL₂ over two tokens, scaled by T².

**Temperature T.** A dial applied before softmax: divide logits by T to
sharpen (T<1) or soften (T>1) a distribution. In KD it softens the teacher's
grades so the *relative* shape of its uncertainty survives.

*Why it matters here:* the fixture runs at T=2.0, and the loss multiplies by
T² to keep gradients comparable across temperatures.

**Soft labels.** The teacher's full probability distribution over tokens —
"70% answer A, 20% answer B, 10% anything else" — as opposed to the hard
label ("it's A"). The dark matter of KD: knowledge in the probabilities.

*Why it matters here:* it's the "softening the master's grades so the student
learns judgment" idea — the student learns *how sure* to be, not just what to
say.

**Seed.** The fixed number that initializes all randomness. Same seed, same
inputs → same outputs.

*Why it matters here:* the demo run is byte-identical across runs — seed 42
from generation through split — which is what makes the pipeline auditable.

---

## 3. Quantization & serving — shipping the graduate

**Quantization.** Storing a model's weights in fewer bits (16-bit floats →
4-bit codes) to shrink memory and speed up inference, trading a little
precision. The compression step between training and serving.

*Why it matters here:* the student is served 4-bit; that's where the $0.15/1M
input-token price in the cost report comes from.

**GGUF.** The llama.cpp model file format — weights plus metadata in one
file — used to run models on CPU or edge devices. The repo emits the
convert-and-quantize command runbook for it.

*Why it matters here:* GGUF is the path to Ollama and laptops; it's how the
distilled student leaves the datacenter.

**AWQ.** Activation-aware weight quantization: 4-bit weights chosen using
calibration data so the channels that matter stay accurate. The repo builds
the autoawq command for it.

*Why it matters here:* the prompt pool itself is the recommended calibration
set — the student is quantized on the same distribution it was taught on.

**vLLM.** A high-throughput GPU inference server (OpenAI-compatible API,
continuous batching, prefix caching). The repo generates the `vllm serve`
command and YAML descriptor.

*Why it matters here:* it's the recommended GPU home for the student —
`--enable-prefix-caching` is free latency for chat traffic with a static
system prompt.

**Ollama.** A one-command local model runner (pull a GGUF, get an OpenAI-style
API on localhost). The repo writes the Ollama Modelfile for the student.

*Why it matters here:* it's the zero-friction way to run the distilled model
on your own machine — and doubles as a teacher endpoint for real runs.

**Merge-then-quantize.** The rule that LoRA adapters are merged into the base
weights (`merge_and_unload`) *before* GGUF/AWQ conversion — quantizing an
unmerged adapter is unsupported.

*Why it matters here:* it's an ordering gotcha the serving docs call out so
the graduate doesn't arrive with its medal still in the envelope.

---

## 4. Eval & cost — the exam and the ledger

**Exact match (EM).** The strictest metric: 1 if the prediction equals the
reference exactly, else 0. Nothing partial counts.

*Why it matters here:* both models score 0.0 on it in the demo bench —
honest evidence that open-ended answers rarely match verbatim, and that the
other metrics are the ones doing the work.

**Token-F1.** Overlap between prediction and reference at the token level:
the harmonic mean of precision (how much of the prediction is right) and
recall (how much of the answer it found). Partial credit that punishes both
bluffing and omissions.

*Why it matters here:* it's the headline metric — student 0.229 vs teacher
0.333 in the demo, which is the 68.6% retention the whole cost argument
hangs on.

**Containment.** The fraction of the reference's tokens that appear anywhere
in the prediction, regardless of order or extras. Recall with a looser
definition of "found it".

*Why it matters here:* student 0.323 vs teacher 0.500 — it shows the student
covers less of the answer even when phrasing drifts.

**Capability slice.** The bench computed separately per skill area
(classification, coding, extraction, reasoning, safety-refusal,
summarization) instead of one blended score. A report card per subject.

*Why it matters here:* it exposes that the demo student *beats* the teacher
on classification (0.270 vs 0.259) and extraction (0.329 vs 0.303) while
lagging on summarization — a per-skill trade, not a uniform drop.

**Quality retention.** Student metric ÷ teacher metric, as a percentage —
how much of the master's performance the apprentice kept. The decision hinge:
cost savings mean nothing if retention is unacceptable for the workload.

*Why it matters here:* 68.6% token-F1 retention is the number every other
number in the cost report defers to.

**Cost per 1M tokens.** The list price for processing (or generating) a
million tokens, usually split input/output. The unit that makes models
comparable.

*Why it matters here:* $3/$15 teacher vs $0.15/$0.60 student is a 20-25x
gap per token — the entire business case, stated in two numbers.

**Breakeven horizon.** How long the monthly savings take to repay the
one-time distillation investment (data generation + training + eval).
Savings scale with volume; the investment is fixed.

*Why it matters here:* at demo pricing and 100k requests/month, a $60
investment breaks even in ~3 days — and the volume table shows payback
shrinking as traffic grows.

**GPU-hours.** One GPU running for one hour — the unit training time is
rented and billed in. The plan converts dataset size into steps, seconds per
step, and GPU-hours before you spend anything.

*Why it matters here:* the demo plan estimates 0.01 GPU-hours for a 7B
student; you get a cost ceiling *before* renting the GPU.

**Deterministic pipeline.** Every stage — seeded generation, rule filters,
dedup, split — is a pure function of its inputs, so the full run reproduces
byte-identically. Not "usually the same": provably, file-for-file the same.

*Why it matters here:* byte-identical runs are what let flow-conservation
tests assert `candidates = dedup + discards + kept` and make every number in
the summary defensible.
