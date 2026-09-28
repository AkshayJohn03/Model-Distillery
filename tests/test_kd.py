"""KD loss: exact hand-computed values (see distillery/train/kd.py docstring),
masking, shapes, and the guarded torch path."""

import importlib.util
import math

import numpy as np
import pytest

from distillery.errors import DependencyMissingError
from distillery.train.kd import kd_loss_numpy, kd_loss_torch

T = 2.0

STUDENT = np.array([[2 * math.log(3), 0.0], [0.0, 2 * math.log(4)]])
TEACHER = np.array([[2 * math.log(2), 0.0], [0.0, 2 * math.log(3)]])
LABELS = np.array([1, 1])

# Hand-derived arithmetic (independent of the implementation):
#   token 1: p_s = [3/4, 1/4], p_t = [2/3, 1/3]
#   token 2: p_s = [1/5, 4/5], p_t = [1/4, 3/4]
KL1 = 0.75 * math.log((3 / 4) / (2 / 3)) + 0.25 * math.log((1 / 4) / (1 / 3))
KL2 = 0.2 * math.log((1 / 5) / (1 / 4)) + 0.8 * math.log((4 / 5) / (3 / 4))
KD = T**2 * (KL1 + KL2) / 2  # KD term is a MEAN over active positions
CE = (math.log(4) + math.log(5 / 4)) / 2  # labels [1, 1]: -ln(1/4), -ln(4/5)


def test_exact_hand_computed_value() -> None:
    loss = kd_loss_numpy(STUDENT, TEACHER, LABELS, temperature=T, alpha=0.5)
    expected = KD + 0.5 * CE
    assert loss == pytest.approx(expected, abs=1e-9)


def test_alpha_zero_is_kd_only() -> None:
    loss = kd_loss_numpy(STUDENT, TEACHER, LABELS, temperature=T, alpha=0.0)
    assert loss == pytest.approx(KD, abs=1e-9)


def test_alpha_one_adds_full_cross_entropy() -> None:
    loss = kd_loss_numpy(STUDENT, TEACHER, LABELS, temperature=T, alpha=1.0)
    assert loss == pytest.approx(KD + CE, abs=1e-9)


def test_ignore_index_masks_both_terms() -> None:
    loss = kd_loss_numpy(
        STUDENT, TEACHER, np.array([1, -100]), temperature=T, alpha=0.5
    )
    expected = T**2 * KL1 + 0.5 * math.log(4)  # only token 1 active
    assert loss == pytest.approx(expected, abs=1e-9)


def test_all_labels_ignored_returns_zero() -> None:
    loss = kd_loss_numpy(STUDENT, TEACHER, np.array([-100, -100]))
    assert loss == 0.0


def test_identical_logits_give_zero_kd_term() -> None:
    loss = kd_loss_numpy(STUDENT, STUDENT, LABELS, temperature=T, alpha=0.0)
    assert loss == pytest.approx(0.0, abs=1e-12)


def test_labels_none_is_pure_kd_over_all_positions() -> None:
    loss = kd_loss_numpy(STUDENT, TEACHER, None, temperature=T, alpha=0.0)
    assert loss == pytest.approx(KD, abs=1e-9)


def test_3d_shape_matches_flattened_2d() -> None:
    student = STUDENT.reshape(1, 2, 2)
    teacher = TEACHER.reshape(1, 2, 2)
    labels = LABELS.reshape(1, 2)
    assert kd_loss_numpy(student, teacher, labels, temperature=T, alpha=0.5) == pytest.approx(
        kd_loss_numpy(STUDENT, TEACHER, LABELS, temperature=T, alpha=0.5), abs=1e-12
    )


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        kd_loss_numpy(STUDENT, TEACHER[:1], LABELS)


def test_label_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="labels"):
        kd_loss_numpy(STUDENT, TEACHER, np.array([1]))


def test_torch_path_guarded() -> None:
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(DependencyMissingError, match="torch"):
            kd_loss_torch(STUDENT, TEACHER, LABELS)
    else:
        torch_loss = float(kd_loss_torch(STUDENT, TEACHER, LABELS, temperature=T, alpha=0.5))
        assert torch_loss == pytest.approx(KD + 0.5 * CE, abs=1e-9)
