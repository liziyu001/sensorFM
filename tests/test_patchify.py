import torch

from sensorfm.model.patchify import patchify, patchify_mask, unpatchify


def test_patchify_shapes():
    x = torch.randn(2, 1440, 34)
    tokens = patchify(x, 20)
    assert tokens.shape == (2, 72, 34, 20)


def test_patchify_roundtrip():
    x = torch.randn(2, 1440, 34)
    tokens = patchify(x, 20)
    restored = unpatchify(tokens, 1440, 34)
    assert torch.equal(restored, x)


def test_patchify_mask_any():
    missing = torch.zeros(1, 1440, 34, dtype=torch.bool)
    missing[0, 0, 0] = True
    tokens = patchify_mask(missing, 20, rule="any")
    assert tokens.shape == (1, 72, 34)
    assert bool(tokens[0, 0, 0])
    assert int(tokens.sum()) == 1


def test_patchify_mask_all():
    missing = torch.zeros(1, 1440, 34, dtype=torch.bool)
    missing[0, 0, 0] = True
    tokens = patchify_mask(missing, 20, rule="all")
    assert not bool(tokens[0, 0, 0])
