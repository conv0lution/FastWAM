from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import G0_PROTOCOL
from experiments.asre_diagnosis.g0.definitions import CONDITIONS, runtime_for
from experiments.asre_diagnosis.g0.launch_four_gpu import (
    Provenance,
    _condition_command,
    _select_gpus,
)


def test_g0_condition_command_uses_normal_suite_text_and_no_ddp(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_text("x", encoding="utf-8")
    donor_root = tmp_path / "donors"
    donor_root.mkdir()
    provenance = Provenance(
        checkpoint=artifact,
        checkpoint_sha256="a" * 64,
        dataset_stats=artifact,
        dataset_stats_sha256="b" * 64,
        donor_mapping=artifact,
        donor_mapping_sha256="c" * 64,
        donor_manifest=artifact,
        donor_manifest_sha256="d" * 64,
        donor_root=donor_root,
        preflight_report=artifact,
        preflight_report_sha256="e" * 64,
        machinery_report=artifact,
        machinery_report_sha256="f" * 64,
    )
    condition = CONDITIONS[3]
    command = _condition_command(
        python=artifact,
        condition_index=3,
        condition=condition,
        output=tmp_path / "out",
        runtime=runtime_for("libero_object", "smoke"),
        provenance=provenance,
    )
    joined = " ".join(command)
    assert f"ASRE_DIAGNOSIS.protocol={G0_PROTOCOL}" in command
    assert "model.load_text_encoder=true" in command
    assert "EVALUATION.prompt_context_cache_path=null" in command
    assert "EVALUATION.task_suite_name=libero_object" in command
    assert "ASRE_DIAGNOSIS.replacement_video_layers=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]" in command
    assert "torchrun" not in joined
    assert "WORLD_SIZE" not in joined


def test_physical_gpu_mapping_requires_four_dedicated_cards() -> None:
    inventory = [
        {"index": index, "memory_free_mib_at_launch": 24000, "name": "A5000"}
        for index in range(8)
    ]
    selected = _select_gpus(inventory, [4, 5, 6, 7])
    assert [row["index"] for row in selected] == [4, 5, 6, 7]
    inventory[4]["memory_free_mib_at_launch"] = 16000
    with pytest.raises(RuntimeError, match="22000 MiB"):
        _select_gpus(inventory, [4, 5, 6, 7])
