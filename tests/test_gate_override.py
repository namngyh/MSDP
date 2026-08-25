"""The online tier steers the frozen network through its gate, nothing else."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from msdp.models import MSDP

HORIZONS = (5, 20, 60)


def _model():
    torch.manual_seed(0)
    model = MSDP(n_features=8, horizons=HORIZONS, hidden_dim=16, latent_dim=12, n_blocks=1)
    model.eval()
    return model


def _batch(model):
    torch.manual_seed(1)
    return torch.randn(2, 252, 8)


def test_overriding_with_the_networks_own_weights_changes_nothing():
    """Identity gate: proves the override enters the same place the gate does."""
    model = _model()
    x = _batch(model)
    with torch.no_grad():
        base = model(x)
        same = model(x, gate_override=base["gate_weights"])
    for key in ("return_quantiles", "mdd_quantiles", "volatility", "direction_prob"):
        assert torch.allclose(base[key], same[key], atol=1e-6), key


def test_a_different_override_actually_moves_the_forecast():
    model = _model()
    x = _batch(model)
    with torch.no_grad():
        base = model(x)
        peaked = torch.zeros_like(base["gate_weights"])
        peaked[..., 0] = 1.0
        moved = model(x, gate_override=peaked)
    assert not torch.allclose(base["return_quantiles"], moved["return_quantiles"], atol=1e-6)


def test_the_output_reports_both_the_prior_and_the_weights_actually_used():
    model = _model()
    x = _batch(model)
    with torch.no_grad():
        peaked = torch.zeros(2, len(HORIZONS), len(model.experts))
        peaked[..., 1] = 1.0
        out = model(x, gate_override=peaked)
    assert torch.allclose(out["gate_weights"], peaked)
    assert not torch.allclose(out["gate_prior"], peaked)
    assert torch.allclose(out["gate_prior"].sum(-1), torch.ones(2, len(HORIZONS)), atol=1e-6)


def test_an_override_of_the_wrong_shape_is_refused():
    model = _model()
    x = _batch(model)
    with pytest.raises(ValueError, match="gate_override"):
        model(x, gate_override=torch.ones(2, len(HORIZONS), len(model.experts) + 1))


def test_a_hedge_posterior_can_drive_the_network():
    """End to end: evidence collected online reaches the published forecast."""
    from msdp.online.hedge import HedgeGateState

    model = _model()
    x = _batch(model)
    with torch.no_grad():
        base = model(x)
    n_experts = len(model.experts)
    state = HedgeGateState.initial(HORIZONS, n_experts, eta=1.0)
    for _ in range(30):
        losses = np.ones(n_experts)
        losses[0] = 0.0
        state.update(0, losses)

    prior = base["gate_prior"][0].numpy()
    posterior = state.posterior_matrix(prior)
    assert posterior[0].argmax() == 0
    assert np.allclose(posterior[1], prior[1]), "horizon chưa có bằng chứng phải giữ nguyên prior"

    override = base["gate_weights"].clone()
    override[0] = torch.tensor(posterior, dtype=override.dtype)
    with torch.no_grad():
        steered = model(x, gate_override=override)
    assert not torch.allclose(base["return_quantiles"][0], steered["return_quantiles"][0], atol=1e-6)
