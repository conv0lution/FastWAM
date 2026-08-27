from __future__ import annotations

import unittest

from experiments.asre_diagnosis.common import (
    ROUND3A_PROTOCOL,
    ROUND3B_PROTOCOL,
    build_conditions,
    build_round2_conditions,
    build_round3a_conditions,
    build_round3b_conditions,
    enabled_to_disabled_layers,
    resolve_condition,
)


class ConditionsTest(unittest.TestCase):
    def test_thirty_layer_conditions_match_protocol(self) -> None:
        conditions = build_conditions(30)
        self.assertEqual(
            [condition.name for condition in conditions],
            [
                "baseline",
                "drop_00_04",
                "drop_05_09",
                "drop_10_14",
                "drop_15_19",
                "drop_20_24",
                "drop_25_29",
                "drop_all",
            ],
        )
        self.assertEqual(conditions[-1].disabled_video_layers, tuple(range(30)))

    def test_non_thirty_layer_groups_are_contiguous_and_cover_every_layer(self) -> None:
        conditions = build_conditions(31)
        group_layers = [layer for condition in conditions[1:7] for layer in condition.disabled_video_layers]
        self.assertEqual(group_layers, list(range(31)))
        group_sizes = [len(condition.disabled_video_layers) for condition in conditions[1:7]]
        self.assertLessEqual(max(group_sizes) - min(group_sizes), 1)

    def test_condition_index_resolves_against_dynamic_layer_count(self) -> None:
        selected = resolve_condition(
            {
                "enabled": True,
                "mode": "drop_video_kv",
                "condition_index": 3,
                "disabled_video_layers": [],
            },
            num_layers=31,
        )
        self.assertEqual(selected, build_conditions(31)[3])

    def test_disabled_mode_rejects_nonempty_intervention(self) -> None:
        with self.assertRaises(ValueError):
            resolve_condition(
                {
                    "enabled": False,
                    "mode": "drop_video_kv",
                    "disabled_video_layers": [0],
                },
                num_layers=30,
            )

    def test_round2_conditions_match_pre_registered_keep_schedules(self) -> None:
        conditions = build_round2_conditions(30)
        self.assertEqual(
            [condition.name for condition in conditions],
            [
                "baseline_round2",
                "keep_15_29",
                "keep_20_29",
                "keep_25_29",
                "keep_00_14",
                "keep_00_19",
                "keep_15_19",
                "keep_15_19_25_29",
            ],
        )
        self.assertEqual(conditions[1].disabled_video_layers, tuple(range(15)))
        self.assertEqual(
            conditions[-1].enabled_video_retrieval_layers(30),
            tuple(range(15, 20)) + tuple(range(25, 30)),
        )

    def test_round2_rejects_wrong_named_schedule(self) -> None:
        with self.assertRaises(ValueError):
            resolve_condition(
                {
                    "enabled": True,
                    "mode": "drop_video_kv",
                    "protocol": "round2_keep_schedules",
                    "condition_name": "keep_15_29",
                    "enabled_video_retrieval_layers": list(range(20, 30)),
                    "disabled_video_layers": [],
                },
                num_layers=30,
            )

    def test_round3a_conditions_are_exact_missing_factorial_cells(self) -> None:
        conditions = build_round3a_conditions(30)
        self.assertEqual(
            [condition.name for condition in conditions],
            ["keep_none_late", "keep_20_24", "keep_15_24"],
        )
        self.assertEqual(
            [condition.enabled_video_retrieval_layers(30) for condition in conditions],
            [(), tuple(range(20, 25)), tuple(range(15, 25))],
        )
        self.assertEqual(conditions[0].disabled_video_layers, tuple(range(30)))
        self.assertEqual(
            conditions[1].disabled_video_layers,
            tuple(range(20)) + tuple(range(25, 30)),
        )
        self.assertEqual(
            conditions[2].disabled_video_layers,
            tuple(range(15)) + tuple(range(25, 30)),
        )

    def test_round3a_empty_keep_schedule_is_not_treated_as_null(self) -> None:
        selected = resolve_condition(
            {
                "enabled": True,
                "mode": "drop_video_kv",
                "protocol": ROUND3A_PROTOCOL,
                "condition_name": "keep_none_late",
                "enabled_video_retrieval_layers": [],
                "disabled_video_layers": list(range(30)),
            },
            num_layers=30,
        )
        self.assertEqual(selected, build_round3a_conditions(30)[0])

    def test_round3a_rejects_mismatched_named_schedule(self) -> None:
        with self.assertRaises(ValueError):
            resolve_condition(
                {
                    "enabled": True,
                    "mode": "drop_video_kv",
                    "protocol": ROUND3A_PROTOCOL,
                    "condition_name": "keep_20_24",
                    "enabled_video_retrieval_layers": list(range(15, 25)),
                    "disabled_video_layers": [],
                },
                num_layers=30,
            )

    def test_enabled_schedule_compiles_to_exact_complement(self) -> None:
        self.assertEqual(
            enabled_to_disabled_layers([1, 3], 5),
            (0, 2, 4),
        )

    def test_round3b_conditions_freeze_exact_cache_sources(self) -> None:
        correct, wrong, no_video = build_round3b_conditions(30)
        self.assertEqual(
            [condition.name for condition in (correct, wrong, no_video)],
            ["late_current_correct", "late_wrong_scene", "late_no_video"],
        )
        self.assertEqual(correct.disabled_video_layers, tuple(range(15)))
        self.assertEqual(correct.replacement_video_layers, ())
        self.assertEqual(wrong.disabled_video_layers, tuple(range(15)))
        self.assertEqual(wrong.replacement_video_layers, tuple(range(15, 30)))
        self.assertEqual(no_video.disabled_video_layers, tuple(range(30)))
        self.assertEqual(no_video.replacement_video_layers, ())

    def test_round3b_wrong_scene_resolves_only_exact_frozen_schedule(self) -> None:
        selected = resolve_condition(
            {
                "enabled": True,
                "mode": "replace_video_kv",
                "protocol": ROUND3B_PROTOCOL,
                "condition_name": "late_wrong_scene",
                "enabled_video_retrieval_layers": list(range(15, 30)),
                "disabled_video_layers": list(range(15)),
                "replacement_video_layers": list(range(15, 30)),
            },
            num_layers=30,
        )
        self.assertEqual(selected, build_round3b_conditions(30)[1])

    def test_round3b_rejects_drop_mode_or_disabled_replacement_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires.*replace_video_kv"):
            resolve_condition(
                {
                    "enabled": True,
                    "mode": "drop_video_kv",
                    "protocol": ROUND3B_PROTOCOL,
                    "condition_name": "late_current_correct",
                    "enabled_video_retrieval_layers": list(range(15, 30)),
                    "disabled_video_layers": list(range(15)),
                },
                num_layers=30,
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            resolve_condition(
                {
                    "enabled": True,
                    "mode": "replace_video_kv",
                    "protocol": ROUND3B_PROTOCOL,
                    "condition_name": "late_wrong_scene",
                    "enabled_video_retrieval_layers": list(range(15, 30)),
                    "disabled_video_layers": list(range(15)),
                    "replacement_video_layers": [14, *range(15, 30)],
                },
                num_layers=30,
            )


if __name__ == "__main__":
    unittest.main()
