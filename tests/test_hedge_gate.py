"""Bayesian model combination over the experts, updated one session at a time.

The network's gate supplies a prior over experts. Each time a horizon matures we
learn which expert was actually right, and multiply that evidence in. No weight
of the network is touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from msdp.online.hedge import HedgeGateState, normalized_losses

HORIZONS = [5, 20, 60]
N_EXPERTS = 4


def _state(eta=0.5):
    return HedgeGateState.initial(horizons=HORIZONS, n_experts=N_EXPERTS, eta=eta)


def test_without_evidence_the_posterior_is_the_network_gate(  ):
    state = _state()
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(state.posterior(prior, horizon_index=0), prior)


def test_posterior_is_always_a_probability_simplex():
    state = _state()
    rng = np.random.default_rng(0)
    for _ in range(50):
        state.update(0, rng.random(N_EXPERTS))
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    posterior = state.posterior(prior, horizon_index=0)
    assert posterior.min() >= 0
    assert posterior.sum() == pytest.approx(1.0)


def test_a_consistently_worse_expert_loses_weight():
    state = _state()
    prior = np.full(N_EXPERTS, 0.25)
    before = state.posterior(prior, horizon_index=0)[3]
    for _ in range(20):
        state.update(0, np.array([0.1, 0.1, 0.1, 1.0]))
    after = state.posterior(prior, horizon_index=0)[3]
    assert after < before
    assert state.posterior(prior, horizon_index=0).argmax() != 3


def test_evidence_for_one_horizon_does_not_leak_into_another():
    state = _state()
    prior = np.full(N_EXPERTS, 0.25)
    for _ in range(20):
        state.update(0, np.array([0.0, 1.0, 1.0, 1.0]))
    assert np.allclose(state.posterior(prior, horizon_index=1), prior)
    assert not np.allclose(state.posterior(prior, horizon_index=0), prior)


def test_equal_losses_leave_the_posterior_untouched():
    state = _state()
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    for _ in range(10):
        state.update(0, np.full(N_EXPERTS, 0.7))
    assert np.allclose(state.posterior(prior, horizon_index=0), prior)


def test_a_zero_learning_rate_disables_the_update():
    state = _state(eta=0.0)
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    for _ in range(10):
        state.update(0, np.array([0.0, 1.0, 1.0, 1.0]))
    assert np.allclose(state.posterior(prior, horizon_index=0), prior)


def test_combination_tracks_the_best_expert_better_than_the_static_gate():
    """The point of Hedge: beat the prior when the prior backs the wrong expert."""
    rng = np.random.default_rng(7)
    truth = rng.normal(0, 1, 400)
    # Expert 0 is accurate; the network's gate wrongly favours expert 3.
    predictions = np.stack(
        [
            truth + rng.normal(0, 0.10, 400),
            truth + rng.normal(0, 0.80, 400),
            truth + rng.normal(0, 0.90, 400),
            truth + rng.normal(0, 1.20, 400),
        ],
        axis=1,
    )
    prior = np.array([0.10, 0.10, 0.10, 0.70])

    state = _state(eta=1.0)
    hedge_error, prior_error = [], []
    for t in range(len(truth)):
        posterior = state.posterior(prior, horizon_index=0)
        hedge_error.append((predictions[t] @ posterior - truth[t]) ** 2)
        prior_error.append((predictions[t] @ prior - truth[t]) ** 2)
        state.update(0, np.abs(predictions[t] - truth[t]))

    assert np.mean(hedge_error[200:]) < np.mean(prior_error[200:])


def test_state_round_trips_through_a_plain_dict():
    state = _state()
    for _ in range(5):
        state.update(1, np.array([0.2, 0.9, 0.4, 0.6]))
    restored = HedgeGateState.from_dict(state.to_dict())
    prior = np.full(N_EXPERTS, 0.25)
    assert restored.eta == state.eta
    assert restored.horizons == state.horizons
    assert np.allclose(
        restored.posterior(prior, horizon_index=1), state.posterior(prior, horizon_index=1)
    )


def test_normalised_losses_are_scale_free_and_bounded():
    """Hedge's guarantee assumes bounded losses; returns are not."""
    small = normalized_losses(np.array([0.001, 0.002, 0.003, 0.004]))
    large = normalized_losses(np.array([10.0, 20.0, 30.0, 40.0]))
    assert np.allclose(small, large)
    assert small.min() == 0.0 and small.max() == 1.0


def test_normalised_losses_of_identical_inputs_are_all_zero():
    assert np.allclose(normalized_losses(np.full(4, 3.3)), np.zeros(4))


def test_a_non_finite_loss_is_refused_rather_than_poisoning_the_weights():
    state = _state()
    with pytest.raises(ValueError, match="hữu hạn"):
        state.update(0, np.array([0.1, np.nan, 0.3, 0.4]))
