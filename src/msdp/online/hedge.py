"""Bayesian model combination over the experts (multiplicative weights / Hedge).

The trained gate in `models/gate.py` produces a softmax over experts for each
horizon. That is a *prior*: it was learned in batch and is frozen between
retrains. This module accumulates the evidence that arrives afterwards - which
expert actually predicted well - and multiplies it in:

    posterior_k  ∝  prior_k · exp(-eta · Σ_t loss_{k,t})

which is Bayes' rule with the gate as prior and the exponentiated cumulative
loss as likelihood, and is exactly the Hedge / multiplicative-weights update.
No network weight is modified; a batch retrain replaces the prior and resets
the evidence.

Hedge's regret bound assumes losses in [0, 1]. Return errors are unbounded and
their scale drifts with volatility, so each round's losses are rescaled to
[0, 1] across experts before being applied - see `normalized_losses`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-12


def normalized_losses(losses) -> np.ndarray:
    """Rescale one round's per-expert losses onto [0, 1].

    Only the *ranking and relative spread* within a round carry information
    about which expert did better; the absolute scale is a property of the
    market that day. Rescaling per round therefore makes the update
    scale-free, and makes a round where every expert did equally well a no-op
    rather than a uniform down-weighting.
    """
    losses = np.asarray(losses, dtype=float)
    if not np.isfinite(losses).all():
        raise ValueError(f"Loss của expert phải hữu hạn, nhận được {losses}")
    smallest, largest = float(losses.min()), float(losses.max())
    if largest - smallest <= EPS:
        return np.zeros_like(losses)
    return (losses - smallest) / (largest - smallest)


@dataclass
class HedgeGateState:
    """Accumulated per-horizon evidence about each expert."""

    horizons: list[int]
    log_weights: np.ndarray  # (n_horizons, n_experts), renormalised each update
    eta: float = 0.5
    rounds: list[int] | None = None

    @classmethod
    def initial(cls, horizons, n_experts: int, eta: float = 0.5) -> "HedgeGateState":
        horizons = [int(h) for h in horizons]
        return cls(
            horizons=horizons,
            log_weights=np.zeros((len(horizons), int(n_experts)), dtype=float),
            eta=float(eta),
            rounds=[0] * len(horizons),
        )

    @property
    def n_experts(self) -> int:
        return int(self.log_weights.shape[1])

    def horizon_index(self, horizon: int) -> int:
        return self.horizons.index(int(horizon))

    def update(self, horizon_index: int, losses) -> np.ndarray:
        """Fold one matured round of per-expert losses into the evidence."""
        losses = normalized_losses(losses)
        if losses.shape != (self.n_experts,):
            raise ValueError(
                f"Cần {self.n_experts} loss cho horizon index {horizon_index}, nhận {losses.shape}"
            )
        row = self.log_weights[horizon_index] - float(self.eta) * losses
        # Renormalise in log space so the weights cannot underflow over years.
        self.log_weights[horizon_index] = row - _logsumexp(row)
        if self.rounds is not None:
            self.rounds[horizon_index] += 1
        return self.log_weights[horizon_index]

    def posterior(self, gate_prior, horizon_index: int) -> np.ndarray:
        """Combine the frozen gate prior with the accumulated evidence."""
        prior = np.asarray(gate_prior, dtype=float)
        if prior.shape != (self.n_experts,):
            raise ValueError(f"Gate prior phải có {self.n_experts} phần tử, nhận {prior.shape}")
        log_posterior = np.log(np.clip(prior, EPS, None)) + self.log_weights[horizon_index]
        log_posterior -= _logsumexp(log_posterior)
        return np.exp(log_posterior)

    def posterior_matrix(self, gate_priors) -> np.ndarray:
        """`(n_horizons, n_experts)` posterior for a whole gate output."""
        priors = np.asarray(gate_priors, dtype=float)
        return np.stack(
            [self.posterior(priors[j], j) for j in range(len(self.horizons))], axis=0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizons": list(self.horizons),
            "log_weights": self.log_weights.tolist(),
            "eta": float(self.eta),
            "rounds": list(self.rounds or []),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HedgeGateState":
        return cls(
            horizons=[int(h) for h in payload["horizons"]],
            log_weights=np.asarray(payload["log_weights"], dtype=float),
            eta=float(payload["eta"]),
            rounds=[int(r) for r in payload.get("rounds", [])] or None,
        )


def _logsumexp(values: np.ndarray) -> float:
    largest = float(np.max(values))
    return largest + float(np.log(np.sum(np.exp(values - largest))))
