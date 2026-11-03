import numpy as np

from sensorfm.config import DataConfig
from sensorfm.data.synthetic import (
    fit_normalizer,
    generate_synthetic,
    normalize,
)


def test_synthetic_shapes():
    cfg = DataConfig(n_subjects=3, days_per_subject=2, window_minutes=1440, patch_time=20)
    batch = generate_synthetic(cfg)

    assert batch.x.shape == (6, 1440, 34)
    assert batch.observed.shape == (6, 1440, 34)
    assert batch.cyclic.shape == (6, 72, 8)
    assert batch.subject_id.shape == (6,)
    assert batch.labels["stress_level"].shape == (3,)
    assert bool(batch.observed.any())
    assert not bool(np.isinf(batch.x[batch.observed]).any())


def test_normalizer_removes_missing():
    cfg = DataConfig(n_subjects=2, days_per_subject=1, window_minutes=1440, patch_time=20)
    batch = generate_synthetic(cfg)
    mean, std = fit_normalizer(batch.x, batch.observed)
    z = normalize(batch.x, batch.observed, mean, std)

    assert np.isfinite(z).all()
    assert np.all(np.abs(z) <= 5.0)
    assert np.all(z[~batch.observed] == 0.0)
