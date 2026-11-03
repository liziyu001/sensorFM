import torch

from sensorfm.model.aim import build_artificial_mask, combine_masks, ensure_visible


def test_mask_invariants():
    generator = torch.Generator().manual_seed(0)
    inherited = torch.rand(4, 72, 34, generator=generator) < 0.2
    artificial = build_artificial_mask(4, 72, 34, generator=generator)

    full, loss_mask = combine_masks(inherited, artificial)

    assert not bool((loss_mask & inherited).any())
    assert torch.equal(loss_mask | inherited, full)


def test_ensure_visible():
    full = torch.ones(2, 72, 34, dtype=torch.bool)
    fixed = ensure_visible(full, torch.Generator().manual_seed(0))
    assert bool((~fixed).any(dim=(1, 2)).all())


def test_artificial_modes_are_nonempty():
    generator = torch.Generator().manual_seed(1)
    for _ in range(8):
        art = build_artificial_mask(3, 72, 34, generator=generator)
        assert bool(art.any())
