"""State the MSDP online tier carries from one session to the next.

Only the gate combination and the conformal calibration learn online. The
network - experts, context, heads - is frozen between batch retrains, which for
MSDP are far apart because one training run costs roughly 6000 seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hedge import HedgeGateState

SCHEMA_VERSION = 1


@dataclass
class PendingForecast:
    """A published forecast whose horizon has not elapsed yet.

    `expert_predictions` are the per-expert auxiliary medians (`aux_return_median`)
    the network produced at the origin, in percent. They are what lets the gate
    learn *which* expert was right rather than only that the blend was wrong.
    """

    origin_date: str
    horizon: int
    horizon_index: int
    expert_predictions: list[float]
    lower: float
    upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_date": self.origin_date,
            "horizon": int(self.horizon),
            "horizon_index": int(self.horizon_index),
            "expert_predictions": [float(v) for v in self.expert_predictions],
            "lower": float(self.lower),
            "upper": float(self.upper),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PendingForecast":
        return cls(
            origin_date=str(payload["origin_date"]),
            horizon=int(payload["horizon"]),
            horizon_index=int(payload["horizon_index"]),
            expert_predictions=[float(v) for v in payload["expert_predictions"]],
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
        )


@dataclass
class OnlineState:
    schema_version: int
    as_of_date: str | None
    hedge: HedgeGateState
    pending: list[PendingForecast] = field(default_factory=list)
    coverage_log: list[dict[str, Any]] = field(default_factory=list)
    session_log: list[dict[str, Any]] = field(default_factory=list)
    source_run_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        horizons,
        n_experts: int,
        eta: float = 0.5,
        source_run_metadata: dict[str, Any] | None = None,
        as_of_date: str | None = None,
    ) -> "OnlineState":
        return cls(
            schema_version=SCHEMA_VERSION,
            as_of_date=as_of_date,
            hedge=HedgeGateState.initial(horizons, n_experts, eta),
            source_run_metadata=dict(source_run_metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "as_of_date": self.as_of_date,
            "hedge": self.hedge.to_dict(),
            "pending": [item.to_dict() for item in self.pending],
            "coverage_log": list(self.coverage_log),
            "session_log": list(self.session_log),
            "source_run_metadata": dict(self.source_run_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OnlineState":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Online state dùng schema {payload.get('schema_version')}, code hiện tại là "
                f"{SCHEMA_VERSION}; chạy lại batch rồi init-online-state."
            )
        return cls(
            schema_version=SCHEMA_VERSION,
            as_of_date=payload.get("as_of_date"),
            hedge=HedgeGateState.from_dict(payload["hedge"]),
            pending=[PendingForecast.from_dict(item) for item in payload.get("pending", [])],
            coverage_log=list(payload.get("coverage_log", [])),
            session_log=list(payload.get("session_log", [])),
            source_run_metadata=dict(payload.get("source_run_metadata", {})),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of_date": self.as_of_date,
            "horizons": list(self.hedge.horizons),
            "n_experts": self.hedge.n_experts,
            "eta": self.hedge.eta,
            "hedge_rounds": list(self.hedge.rounds or []),
            "pending": len(self.pending),
            "sessions_applied": len(self.session_log),
            "source_run_metadata": self.source_run_metadata,
        }
