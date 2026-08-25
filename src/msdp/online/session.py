"""Advance the MSDP online tier by the sessions that have become observable.

Nothing here trains. A published forecast sits in `state.pending` until its
horizon has actually elapsed in the price series; only then is it scored, and
only then does the gate combination learn from it. That rule is what keeps the
online weights free of look-ahead.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .state import OnlineState, PendingForecast


def realized_return_percent(closes: pd.Series, origin_date: str, horizon: int) -> float | None:
    """Log return over exactly `horizon` sessions, in percent.

    Percent because the network's return quantiles are in percent - see
    `inference.predict_latest_ensemble`, which projects the index with
    `exp(q / 100)`. Returns None while the target session has not happened yet.
    """
    index = pd.DatetimeIndex(closes.index)
    stamp = pd.Timestamp(origin_date)
    if stamp not in index:
        return None
    origin = int(index.get_loc(stamp))
    target = origin + int(horizon)
    if target > len(closes) - 1:
        return None
    values = np.asarray(closes, dtype=float)
    return float(100.0 * np.log(values[target] / values[origin]))


def mature_pending(state: OnlineState, closes: pd.Series) -> list[dict[str, Any]]:
    """Score every pending forecast whose horizon has elapsed, and learn from it.

    A pending forecast whose origin is no longer in the series is dropped rather
    than scored: the price history it was made against has changed, so any loss
    computed from it would be meaningless.
    """
    index = pd.DatetimeIndex(closes.index)
    remaining: list[PendingForecast] = []
    matured: list[dict[str, Any]] = []

    for item in state.pending:
        stamp = pd.Timestamp(item.origin_date)
        if stamp not in index:
            continue
        realized = realized_return_percent(closes, item.origin_date, item.horizon)
        if realized is None:
            remaining.append(item)
            continue
        predictions = np.asarray(item.expert_predictions, dtype=float)
        losses = np.abs(predictions - realized)
        state.hedge.update(item.horizon_index, losses)
        covered = bool(item.lower <= realized <= item.upper)
        record = {
            "origin_date": item.origin_date,
            "horizon": item.horizon,
            "realized_return_percent": realized,
            "expert_losses": losses.tolist(),
            "best_expert": int(losses.argmin()),
            "covered": covered,
        }
        state.coverage_log.append(
            {"origin_date": item.origin_date, "horizon": item.horizon, "covered": covered}
        )
        matured.append(record)

    state.pending = remaining
    return matured


def record_forecast(
    state: OnlineState,
    origin_date: str,
    horizon: int,
    horizon_index: int,
    expert_predictions,
    lower: float,
    upper: float,
) -> None:
    """Queue a published forecast so a later session can score it."""
    state.pending.append(
        PendingForecast(
            origin_date=str(origin_date),
            horizon=int(horizon),
            horizon_index=int(horizon_index),
            expert_predictions=[float(v) for v in np.ravel(expert_predictions)],
            lower=float(lower),
            upper=float(upper),
        )
    )


def empirical_coverage(state: OnlineState, horizon: int, window: int = 250) -> float | None:
    """Realised coverage of the matured forecasts for one horizon."""
    hits = [
        bool(row["covered"])
        for row in state.coverage_log
        if int(row["horizon"]) == int(horizon)
    ][-int(window):]
    return float(np.mean(hits)) if hits else None
