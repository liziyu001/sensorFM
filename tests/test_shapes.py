import torch

from sensorfm.config import AimConfig, ModelConfig
from sensorfm.model.sensorfm import SensorFM


def test_forward_shapes_and_finite():
    model_cfg = ModelConfig(
        n_channels=34,
        window_minutes=1440,
        patch_time=20,
        d_model=64,
        enc_layers=2,
        enc_heads=1,
        d_dec=48,
        dec_layers=1,
        dec_heads=1,
    )
    model = SensorFM(model_cfg, AimConfig())
    x = torch.randn(2, 1440, 34)
    missing = torch.zeros(2, 1440, 34, dtype=torch.bool)
    cyclic = torch.randn(2, 72, 8)

    pred, loss_mask = model(x, missing, cyclic)

    assert pred.shape == (2, 72, 34, 20)
    assert loss_mask.shape == (2, 72, 34)
    assert torch.isfinite(pred).all()
