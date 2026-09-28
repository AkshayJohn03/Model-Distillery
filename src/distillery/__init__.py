"""Model-Distillery: end-to-end model distillation pipeline.

Synthetic SFT data generation -> quality filtering -> QLoRA training (with a
logit-KD reference) -> quantization -> teacher-vs-student evaluation -> serving
configs. Heavy ML dependencies are optional and lazily imported; the whole
pipeline runs offline on a deterministic mock teacher.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
