"""Frozen protocol constants for the final Salvage-B dissociation gate."""

from __future__ import annotations


CONDITIONS = (
    "current_all",
    "wrong_all",
    "svd_r97",
    "svd_r170",
)

ENDPOINT_CONDITIONS = ("current_all", "wrong_all")
PROJECTED_CONDITIONS = ("svd_r97", "svd_r170")
RANK_BY_CONDITION = {
    "current_all": 3072,
    "wrong_all": 0,
    "svd_r97": 97,
    "svd_r170": 170,
}

PRIMARY_CONDITION = "svd_r170"
SECONDARY_CONDITION = "svd_r97"
WORLD_DRAWS_PER_SAMPLE = 4
WORLD_METRIC_DIRECTION = "lower_is_better"

SPECIAL_FAILURE_CLASSIFICATIONS = (
    "SHARED-INTERFACE-NOT-AVAILABLE",
    "WORLD-METRIC-NOT-VALIDATABLE",
    "WORLD-ENDPOINT-UNINFORMATIVE",
)
SCIENTIFIC_CLASSIFICATIONS = ("STRONG", "MODERATE", "WEAK")
FINAL_CLASSIFICATIONS = SCIENTIFIC_CLASSIFICATIONS + SPECIAL_FAILURE_CLASSIFICATIONS

# Pre-registered point-estimate thresholds from the Salvage-B protocol.
STRONG_ACTION_CURRENT_GAP_MAX = 0.05
STRONG_ACTION_RECOVERY_MIN = 0.95
STRONG_WORLD_RECOVERY_MAX = 0.75
MODERATE_ACTION_CURRENT_GAP_MAX = 0.10
MODERATE_WORLD_RECOVERY_MAX = 0.85
WEAK_WORLD_RECOVERY_MIN = 0.90

# This value only labels the protocol's qualitative "broadly track" clause;
# all outcomes not satisfying STRONG or MODERATE conservatively become WEAK.
BROAD_TRACKING_GAP_MAX = 0.10

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
