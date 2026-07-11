"""The seven evaluation arms (plan §6). Each is a configuration of the layer;
the base model is identical across arms (matched-model identification)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmConfig:
    name: str
    gating: str        # "table" | "confmap" | "uniform" | "always_hedged"
    three_state: bool  # NOT_KNOWING -> DECLINE (the floor path)
    store: bool
    acknowledge: bool
    provenance: bool   # informational: confmap arms ignore provenance in expression


ARMS: dict[str, ArmConfig] = {
    "full":           ArmConfig("full",           "table",         True,  True,  True,  True),
    "uniform":        ArmConfig("uniform",        "uniform",       False, True,  True,  True),
    "always_hedged":  ArmConfig("always_hedged",  "always_hedged", False, True,  True,  True),
    "threshold_only": ArmConfig("threshold_only", "confmap",       False, True,  True,  True),
    "store_no_ack":   ArmConfig("store_no_ack",   "table",         True,  True,  False, True),
    "no_provenance":  ArmConfig("no_provenance",  "confmap",       True,  True,  True,  False),
    "stateless":      ArmConfig("stateless",      "table",         True,  False, False, True),
}

ARM_ORDER = list(ARMS)  # arm seeds derive from this ordering (PROTOCOL)
