from __future__ import annotations

import unittest

from experiments.asre_diagnosis.common import (
    ROUND4C_PROTOCOL,
    SALVAGE_A_PROTOCOL,
    build_salvage_a_conditions,
    resolve_condition,
)
from experiments.asre_diagnosis.salvage_a.preflight import (
    ROUND4C_ANALYSIS_COMMIT,
    ROUND4C_ARTIFACT_TYPE,
    _validate_round4c_authorization,
)


class SalvageAConditionTest(unittest.TestCase):
    def test_round4c_aggregate_authorizes_salvage(self) -> None:
        frozen = {
            "artifact_type": ROUND4C_ARTIFACT_TYPE,
            "protocol": ROUND4C_PROTOCOL,
            "status": "complete",
            "classification": {"classification": "WEAK"},
            "stop_rule_applied": True,
            "later_stage_launched": False,
            "git_commit_hash": ROUND4C_ANALYSIS_COMMIT,
        }
        _validate_round4c_authorization(frozen)

        drifted = dict(frozen, artifact_type="asre_round4c_summary")
        with self.assertRaisesRegex(ValueError, "artifact_type"):
            _validate_round4c_authorization(drifted)

    def test_frozen_eight_condition_matrix(self) -> None:
        conditions = build_salvage_a_conditions(30)
        self.assertEqual(
            [condition.name for condition in conditions],
            [
                "current_all",
                "wrong_all",
                "svd_r36",
                "actionaware_r36",
                "random_r36",
                "svd_r97",
                "actionaware_r97",
                "random_r97",
            ],
        )
        for condition in conditions:
            self.assertEqual(condition.disabled_video_layers, tuple(range(15)))
        for condition in conditions[1:]:
            self.assertEqual(condition.replacement_video_layers, tuple(range(15, 30)))

    def test_resolver_accepts_actionaware_and_rejects_drift(self) -> None:
        config = {
            "enabled": True,
            "protocol": SALVAGE_A_PROTOCOL,
            "mode": "replace_video_kv",
            "condition_index": 3,
            "enabled_video_retrieval_layers": list(range(15, 30)),
            "disabled_video_layers": list(range(15)),
            "replacement_video_layers": list(range(15, 30)),
            "subspace_basis_kind": "actionaware",
            "subspace_rank": 36,
        }
        condition = resolve_condition(config, 30)
        self.assertEqual(condition.name, "actionaware_r36")
        drifted = dict(config, subspace_rank=97)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            resolve_condition(drifted, 30)

    def test_protocol_is_locked_to_thirty_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 30"):
            build_salvage_a_conditions(29)


if __name__ == "__main__":
    unittest.main()
