"""Per-session bookkeeping: a forecast only teaches the gate once it matures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msdp.online.state import OnlineState, PendingForecast
from msdp.online.session import mature_pending, realized_return_percent

HORIZONS = [5, 20, 60]
N_EXPERTS = 4


@pytest.fixture
def closes():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=120)
    return pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)


def _state():
    return OnlineState.initial(
        horizons=HORIZONS, n_experts=N_EXPERTS, eta=1.0, source_run_metadata={"run_id": "test"}
    )


def _pending(closes, position, horizon, predictions):
    return PendingForecast(
        origin_date=str(closes.index[position].date()),
        horizon=horizon,
        horizon_index=HORIZONS.index(horizon),
        expert_predictions=list(predictions),
        lower=-5.0,
        upper=5.0,
    )


def test_realized_return_is_the_percent_log_return_over_exactly_h_sessions(closes):
    origin, horizon = 40, 5
    expected = 100.0 * np.log(closes.iloc[origin + horizon] / closes.iloc[origin])
    assert realized_return_percent(closes, str(closes.index[origin].date()), horizon) == pytest.approx(expected)


def test_a_forecast_does_not_mature_before_its_horizon(closes):
    state = _state()
    state.pending.append(_pending(closes, 40, 5, [0.1, 0.2, 0.3, 0.4]))
    matured = mature_pending(state, closes.iloc[: 40 + 4 + 1])
    assert matured == []
    assert len(state.pending) == 1


def test_a_forecast_matures_exactly_on_its_horizon(closes):
    state = _state()
    state.pending.append(_pending(closes, 40, 5, [0.1, 0.2, 0.3, 0.4]))
    matured = mature_pending(state, closes.iloc[: 40 + 5 + 1])
    assert len(matured) == 1
    assert state.pending == []
    assert matured[0]["horizon"] == 5


def test_maturing_moves_the_gate_toward_the_expert_that_was_right(closes):
    state = _state()
    prior = np.full(N_EXPERTS, 0.25)
    origin = 40
    truth = realized_return_percent(closes, str(closes.index[origin].date()), 5)
    # Expert 2 nails it; the rest are far off.
    predictions = [truth + 5.0, truth - 5.0, truth, truth + 4.0]
    for step in range(12):
        state.pending.append(_pending(closes, origin + step, 5, predictions))
    mature_pending(state, closes)
    assert state.hedge.posterior(prior, 0).argmax() == 2


def test_only_matured_horizons_receive_evidence(closes):
    state = _state()
    prior = np.full(N_EXPERTS, 0.25)
    state.pending.append(_pending(closes, 40, 5, [0.0, 9.0, 9.0, 9.0]))
    mature_pending(state, closes)
    assert not np.allclose(state.hedge.posterior(prior, 0), prior)
    assert np.allclose(state.hedge.posterior(prior, 1), prior)
    assert np.allclose(state.hedge.posterior(prior, 2), prior)


def test_a_pending_forecast_never_scores_itself(closes):
    """Walk forward and assert nothing unmatured has already been folded in."""
    state = _state()
    for position in range(30, 80):
        visible = closes.iloc[: position + 1]
        mature_pending(state, visible)
        for item in state.pending:
            origin = visible.index.get_loc(pd.Timestamp(item.origin_date))
            assert origin + item.horizon > len(visible) - 1
        state.pending.append(_pending(closes, position, 5, [0.1, 0.2, 0.3, 0.4]))
    assert len(state.pending) == 5


def test_a_forecast_whose_origin_left_the_series_is_dropped_not_scored(closes):
    state = _state()
    state.pending.append(
        PendingForecast(
            origin_date="1999-01-04", horizon=5, horizon_index=0,
            expert_predictions=[0.1] * N_EXPERTS, lower=-1.0, upper=1.0,
        )
    )
    matured = mature_pending(state, closes)
    assert matured == []
    assert state.pending == []


def test_state_round_trips_through_a_plain_dict(closes):
    state = _state()
    state.pending.append(_pending(closes, 40, 5, [0.1, 0.2, 0.3, 0.4]))
    mature_pending(state, closes)
    restored = OnlineState.from_dict(state.to_dict())
    prior = np.full(N_EXPERTS, 0.25)
    assert restored.as_of_date == state.as_of_date
    assert np.allclose(restored.hedge.posterior(prior, 0), state.hedge.posterior(prior, 0))
    assert restored.source_run_metadata["run_id"] == "test"
