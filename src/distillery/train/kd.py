"""Logit-based knowledge distillation loss (reference implementation).

    loss = T^2 * KL( softmax(student/T) || softmax(teacher/T) ) + alpha * CE(student, labels)

Direction note: as specified this is the *reverse* KL -- KL(p_s || p_t) =
sum p_s log(p_s / p_t) -- which is **mode-seeking**: the student concentrates on
where the teacher is confident and may under-cover the teacher's tails. The
classic Hinton formulation is the *forward* KL, KL(p_t || p_s), which is
mass-covering (student spreads mass over all teacher modes). Swap the two
softmax arguments to get the forward form; the ``T^2`` factor compensates the
gradient shrinkage of softmax at temperature T so the distillation term keeps a
usable magnitude relative to CE.

The numpy implementation is the canonical reference (works everywhere, tested
against a hand-computed example); the torch twin mirrors it for autograd.
"""

from __future__ import annotations

import numpy as np

from distillery.errors import DependencyMissingError


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def kd_loss_numpy(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> float:
    """Distillation loss on logits; shapes ``(V,)``, ``(N, V)`` or ``(B, S, V)``.

    KD and CE terms are means over the active positions. Positions whose label
    equals ``ignore_index`` are excluded from both terms. Returns a float.

    Hand-computed reference (see ``tests/test_kd.py``), T=2, alpha=0.5,
    labels=[1, 1], both terms averaged over the 2 active positions::

        student [[2ln3, 0], [0, 2ln4]]  -> p_s = [[3/4, 1/4], [1/5, 4/5]]
        teacher [[2ln2, 0], [0, 2ln3]]  -> p_t = [[2/3, 1/3], [1/4, 3/4]]
        KL1 = 0.75 ln(9/8) + 0.25 ln(3/4)
        KL2 = 0.2 ln(4/5) + 0.8 ln(16/15)
        KD  = T^2 * (KL1 + KL2) / 2
        CE  = (-ln p_s1[1] - ln p_s2[1]) / 2 = (ln 4 + ln(5/4)) / 2
        loss = KD + 0.5 * CE
    """
    student = np.asarray(student_logits, dtype=np.float64)
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    if student.shape != teacher.shape:
        raise ValueError(f"shape mismatch: student {student.shape} vs teacher {teacher.shape}")
    if student.ndim == 1:
        student = student.reshape(1, -1)
        teacher = teacher.reshape(1, -1)
    if student.ndim == 3:
        student = student.reshape(-1, student.shape[-1])
        teacher = teacher.reshape(-1, teacher.shape[-1])
    if student.ndim != 2:
        raise ValueError(f"expected 1D/2D/3D logits, got ndim={np.asarray(student_logits).ndim}")

    if labels is not None:
        flat_labels = np.asarray(labels).reshape(-1)
        if flat_labels.shape[0] != student.shape[0]:
            raise ValueError("labels must match the token dimension of the logits")
        mask = flat_labels != ignore_index
        if not mask.any():
            return 0.0
    else:
        flat_labels = None
        mask = np.ones(student.shape[0], dtype=bool)

    log_p = _log_softmax(student / temperature)
    log_q = _log_softmax(teacher / temperature)
    p = np.exp(log_p)
    kl = (p * (log_p - log_q)).sum(axis=-1)
    kd_term = temperature**2 * kl[mask].mean()

    ce_term = 0.0
    if flat_labels is not None:
        rows = np.arange(student.shape[0])[mask]
        ce_term = -log_p[rows, flat_labels[mask]].mean()

    return float(kd_term + alpha * ce_term)


def kd_loss_torch(
    student_logits: object,  # torch.Tensor when the torch extra is installed
    teacher_logits: object,
    labels: object | None = None,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> object:
    """Autograd-friendly torch mirror of :func:`kd_loss_numpy` (scalar tensor)."""
    try:
        import torch
    except ImportError as exc:
        raise DependencyMissingError("logit-KD torch path", "'model-distillery[torch]'") from exc

    student = torch.as_tensor(student_logits, dtype=torch.float64)
    teacher = torch.as_tensor(teacher_logits, dtype=torch.float64)
    if student.dim() == 1:
        student = student.reshape(1, -1)
        teacher = teacher.reshape(1, -1)
    if student.dim() == 3:
        student = student.reshape(-1, student.shape[-1])
        teacher = teacher.reshape(-1, teacher.shape[-1])

    tensor_labels = None if labels is None else torch.as_tensor(labels).reshape(-1)
    if tensor_labels is not None:
        mask = tensor_labels != ignore_index
        if not mask.any():
            return torch.tensor(0.0, dtype=torch.float64)
    else:
        mask = torch.ones(student.shape[0], dtype=torch.bool)

    log_p = torch.log_softmax(student / temperature, dim=-1)
    log_q = torch.log_softmax(teacher / temperature, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    kd_term = temperature**2 * kl[mask].mean()

    ce_term = torch.tensor(0.0, dtype=torch.float64)
    if tensor_labels is not None:
        rows = torch.arange(student.shape[0])[mask]
        ce_term = -log_p[rows, tensor_labels[mask]].mean()

    return kd_term + alpha * ce_term
