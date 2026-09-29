# Hyperframes Composition Brief: Model-Distillery — whiteboard lecture

## Objective
Create a long-form whiteboard explainer lecture video (NOT a launch video) for
Model-Distillery, teaching the distillation pipeline end to end with narration
on.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: ~360-420 seconds (voice-driven; TUTOR_BRIEF overrides the 15-25s
  default — this is a 5+ minute lecture; scene durations flex to narration WAV
  lengths)

## Source Material
- Project root: D:\aria\Projects\Model-Distillery
- Primary files read: README.md, artifacts/final/summary.json,
  artifacts/final/bench_report.md, artifacts/final/cost_report.md,
  artifacts/final/training_plan.md, src/distillery/train/kd.py (test fixture),
  tests/test_kd.py
- Product name: Model-Distillery
- Tagline / strongest claim: "122 candidates → 59 kept → 68.6% retention,
  byte-identical runs" and "$60 invested, breakeven in ~3 days at 100k
  requests/month"
- Key visual moment to recreate: the whiteboard itself — hand-drawn sketches
  of the pipeline stages with real artifact numbers filling in as the lecture
  progresses
- Copy that must appear verbatim (numbers from artifacts):
  - 122 candidates / 59 kept / −13 dedup / −8 repetition / −42 rejection
  - 46 / 12 train/val split · leak caught: 1
  - token-F1: student 0.229 vs teacher 0.333 · retention 68.6%
  - slices: classification 0.270 vs 0.259 · extraction 0.329 vs 0.303 ·
    summarization 0.285 vs 0.515
  - $645/mo vs $28.50/mo · savings $616.50 · investment $60 · breakeven ~3 days
  - 114 tests · byte-identical runs · hand-computed KD fixture (to 1e-9)
  - KD loss = T² · KL(student ‖ teacher) + α · CE

## Creative Direction
- Tone preset: polished (nearest preset; freeform direction governs)
- Creative direction: "A patient senior engineer at a whiteboard teaching ONE
  system to a smart junior" — whiteboard explainer, no hype, lecture pacing,
  every keyword defined on screen at first use
- Interpretation: long holds; hand-drawn ink sketches; definition cards
  (keyword + one plain sentence) pinned under each sketch; quiet wipes
  between scenes; calm narration carries the video
- Angle: the surgeon's-hourly-rate scenario → junior-chef analogy → nine
  stages walked on the board → measured numbers → 30-second recap
- Hook: fee schedule card "$4,000/hr — question: how do I get to the train
  station?"
- Outro / punchline: numbered recap → "Model-Distillery — measurable,
  auditable, byte-identical."
- Avoid:
  - Hype adjectives, launch energy, generic SaaS language
  - Abstract filler visuals; every sketch is one of the nine stages
  - Any text the viewer can't finish reading before it exits

## Visual Identity (whiteboard explainer)
- Background: #F6F2E8 (warm whiteboard) with faint #E7E0CF grid lines
- Ink: #23272E; Accent blue #1D4ED8 (structure/headings); Accent green
  #15803D (kept/saved); Accent red-orange #C2410C (discards/leaks)
- Display font: "Segoe Print" (marker feel; system font), fallback
  "Comic Sans MS", cursive
- Body font: "Segoe UI", Arial, sans-serif for dense definition sentences
- Visual references from the project: pipeline stages (README mermaid),
  discard histogram, bench slices table, cost curve, KD loss formula

## Storyboard
Use the storyboard + voiceover script in `brag-output/brag-plan.md` as the
creative contract. Ten scenes:
1. The problem (hook) — surgeon fee card → distillation/teacher/student
   definition cards
2. The core idea — MASTER → TEXTBOOK → STUDENT → EXAM sketch + 9-stage map
3. Generate the textbook — prompt pool (61) → persona grid → 122 counter;
   SFT/chat JSONL/CoT cards
4. Dedup — fingerprints, LSH buckets, Jaccard ≥ 0.8, union-find; −13
5. Filter + rejection — funnel; discard histogram; Goodhart warning; 59 kept
6. Decontamination — 13-gram leak screen; caught: 1; stratified 46/12
7. QLoRA + KD loss — frozen 4-bit base + adapter; formula card; hand-computed
   fixture stamp
8. The exam — bench table + slices (student wins 2) + 68.6% retention circle
9. Breakeven — cost curve crossing + ledger $645 → $28.50, $616.50 saved
10. Recap — numbered takeaways + end card

## Audio
- Audio role: narration-forward lecture; music is a barely-there bed
- Audio arc: voice runs the entire video; music sits at 0.13 underneath and
  fades over the final ~4s
- Music: assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3
- Music treatment: data-volume 0.13 constant (voice is always present, so the
  bed stays ducked); fade out over the last 4 seconds
- Music cue guidance: bundled preset exists at
  brag skill assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json;
  beat sync is intentionally NOT used for text reveals — lecture pacing and
  reading time win; natural timing chosen for readability throughout
- Audio-reactive treatment: none — deliberately skipped for calm; a still
  whiteboard suits the tone (documented decision, not an extraction failure)
- Audio-coupled moments:
  - definition card arrivals — soft drop (drop_001/drop_002, volume ≤ 0.4)
  - tally stamps (−13, −8, −42, 59 kept, leak caught, breakeven) — soft
    impact (impactSoft_medium_000, volume ≤ 0.5)
  - scene header arrival — one soft impact at most
- SFX selection guidance: sparse (roughly one cue per 20-30s); never over
  narration emphasis; reuse a small coherent palette
- SFX analysis guidance: brag skill assets/sfx/sfx-analysis.md; low
  high-frequency-risk picks only
- Exact SFX choice: Hyperframes should adjust timestamps/density to the
  implemented animation within the sparse posture above
- Voiceover: af_heart via `npx hyperframes tts`, one WAV per scene in
  assets/voiceover/ (vo-scene1.wav … vo-scene10.wav), each wired on its own
  track at its scene start, data-volume 1
- Audio files: music + SFX already copied into composition/assets/

## Hyperframes Instructions
Load the Hyperframes domain skills — hyperframes-core, hyperframes-animation,
hyperframes-creative, hyperframes-keyframes, hyperframes-cli. /brag is its own
workflow: do not enter the hyperframes entry-point intent interview and do not
route into its generic promo / launch-video workflow. Prefer native
Hyperframes conventions over anything in /brag.

Requirements:
- Standalone composition (top-level index.html, no template wrapper); one
  paused GSAP timeline registered on window.__timelines
- Ten scene clips with data-start/data-duration driven by the measured
  narration WAV lengths; root data-duration = total
- Show real numbers from the repo artifacts (list above) — this satisfies
  "show the thing" for a pipeline/CLI project
- Keep all text readable: definition sentences hold ≥ 0.3s per word; the
  lecture format gives generous holds
- WCAG contrast: dark ink on warm white passes; check must pass 0 errors
- No CSS transform + GSAP transform conflicts; no tweening .clip elements;
  every <audio> has a unique id; no crossorigin on media
- Run `npx hyperframes check` before render — it is brag's single gate
