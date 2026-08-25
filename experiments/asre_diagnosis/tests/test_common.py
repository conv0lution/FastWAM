from __future__ import annotations

import unittest

from experiments.asre_diagnosis.common import build_conditions, resolve_condition


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


if __name__ == "__main__":
    unittest.main()
