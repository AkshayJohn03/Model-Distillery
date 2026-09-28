# Security Policy — Model-Distillery

Model-Distillery generates training data from teacher models and trains
students on it. Threat model and defenses:

| Threat | Defense |
|---|---|
| Training-data poisoning via the teacher | Ordered rule gate + JudgeHook scoring + rejection sampling; every kept item carries its source config and quality score so provenance is auditable |
| Eval contamination (student "cheats" the exam) | 13-gram decontamination against the eval corpus with overlap evidence; planted-overlap tests prove the screen fires |
| Dataset drift/silent corruption | Deterministic pipeline (byte-identical summaries across runs); balance reports; versioned JSONL with sha256 manifests |
| GPU/serve surface | Serve configs are templates; the OpenAI-compatible shim binds locally by default; quantize tooling is documented, never auto-executed |
| Secrets | `DISTILLERY_TEACHER_*` env-only; the offline path requires no keys |

Report issues marked `security`.