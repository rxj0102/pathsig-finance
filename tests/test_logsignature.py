"""Log-signature checks. Without signatory the old code returned the plain signature
under the log-signature's name, and the augmentation fallback was off by a factor of 2."""
import math

import numpy as np
import pytest

from pathsig.augmentations import logsig_augmentation
from pathsig.signatures import _tensor_mul, compute_logsignature, compute_signature, lyndon_words, tensor_log_levels


@pytest.mark.parametrize("d,depth,n", [(2, 2, 3), (2, 4, 8), (3, 3, 14), (4, 3, 30)])
def test_dimension_is_the_number_of_lyndon_words(d, depth, n):
    assert len(lyndon_words(d, depth)) == n                       # Witt's formula
    path = np.cumsum(np.random.default_rng(0).normal(size=(20, d)), axis=0)
    assert compute_logsignature(path, depth).shape == (n,)


def test_levels_one_and_two():
    path = np.cumsum(np.random.default_rng(1).normal(size=(100, 2)), axis=0)
    ls, sig = compute_logsignature(path, 2), compute_signature(path, 2)
    assert np.allclose(ls[:2], sig[:2])                            # increments
    s2 = sig[2:].reshape(2, 2)
    x, y = path[:, 0] - path[0, 0], path[:, 1] - path[0, 1]
    levy = 0.5 * np.sum(x[:-1] * np.diff(y) - y[:-1] * np.diff(x))
    assert ls[2] == pytest.approx(0.5 * (s2[0, 1] - s2[1, 0])) and ls[2] == pytest.approx(levy)


def test_exponential_of_the_log_recovers_the_signature():
    d, depth = 3, 4
    path = np.cumsum(np.random.default_rng(2).normal(size=(40, d)), axis=0)
    sig = compute_signature(path, depth)
    log = tensor_log_levels(sig, d, depth)
    out, power = [np.zeros_like(x) for x in log], log
    for k in range(1, depth + 1):
        out = [o + p / math.factorial(k) for o, p in zip(out, power)]
        power = _tensor_mul(power, log, depth)
    assert np.allclose(np.concatenate([o.ravel() for o in out]), sig, atol=1e-10)


def test_straight_line_has_only_its_increment():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    ls = compute_logsignature(line, 3)
    assert np.allclose(ls[:3], [1, 2, 3]) and np.allclose(ls[3:], 0, atol=1e-12)


def test_augmentation_matches_and_works_beyond_depth_two():
    path = np.cumsum(np.random.default_rng(3).normal(size=(30, 2)), axis=0)
    assert np.allclose(logsig_augmentation(path, 2), compute_logsignature(path, 2))
    assert logsig_augmentation(path, 3).shape == (5,)
