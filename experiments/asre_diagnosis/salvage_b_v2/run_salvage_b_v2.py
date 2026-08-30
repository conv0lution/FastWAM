"""Fail-closed staged driver for the corrected native-node experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run(name: str, command: list[str], *, root: Path, env=None) -> None:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    print(f"[SalvageB-v2] Starting {name}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"Salvage-B-v2 stage {name} exited {process.returncode}; inspect {log}.")
    print(f"[SalvageB-v2] Completed {name}", flush=True)


def _parallel(name: str, commands: list[tuple[list[str], dict[str, str] | None]], *, root: Path) -> None:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    print(f"[SalvageB-v2] Starting {name}", flush=True)
    for index, (command, env) in enumerate(commands):
        log = logs / f"{name}.{index}.log"
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, handle, log))
    failures = []
    for process, handle, log in running:
        code = process.wait()
        handle.close()
        if code:
            failures.append((code, log))
    if failures:
        raise RuntimeError(f"Salvage-B-v2 parallel stage {name} failed: {failures}")
    print(f"[SalvageB-v2] Completed {name}", flush=True)


def _launcher_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    return env


def _publish_stop(root: Path, classification: str, reason: str, python: str) -> None:
    _run(
        "publish_stop",
        [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.publish_stop", "--output-root", str(root), "--classification", classification, "--reason", reason],
        root=root,
    )


def run(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    python = str(args.python.resolve())
    preflight = root / "preflight_report.json"
    _run(
        "preflight",
        [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.preflight", "--output-root", str(root), "--source-root", str(args.source_root.resolve())],
        root=root,
    )
    _run(
        "unit_tests",
        [python, "-m", "pytest", "-q", "experiments/asre_diagnosis/tests/test_salvage_b_v2_native_shared_node.py"],
        root=root,
    )
    _run(
        "full_asre_tests",
        [python, "-m", "pytest", "-q", "experiments/asre_diagnosis/tests"],
        root=root,
    )
    machinery_env = dict(os.environ)
    machinery_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
    try:
        _run(
            "machinery",
            [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.machinery", "--preflight", str(preflight), "--output-root", str(root)],
            root=root,
            env=machinery_env,
        )
    except RuntimeError:
        report_path = root / "machinery_report.json"
        classification = "NATIVE-CLAMP-IDENTITY-FAILED"
        reason = "Native shared-node machinery did not pass."
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("native_clamp_identity_passed") is True:
                classification = "SHARED-NODE-REACH-FAILED"
                if report.get("donor_observation_only_passed") is not True:
                    reason = (
                        "Donor provenance gate failed: Wrong must vary donor RGB only "
                        "under current prompt/proprio/noise/scheduler inputs."
                    )
                elif report.get("basis_coordinate_gate_passed") is not True:
                    reason = (
                        "Round-4B basis-coordinate gate failed against native-stock "
                        "prefix K/V; stop before outcomes and fit a native-stock basis "
                        "on the frozen calibration split."
                    )
        _publish_stop(root, classification, reason, python)
        return

    action_base = [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.launch_action", "--preflight", str(preflight), "--output-root", str(root), "--python", python, "--launch-stagger-seconds", str(args.launch_stagger_seconds)]
    world_base = [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.launch_world", "--preflight", str(preflight), "--output-root", str(root), "--python", python, "--launch-stagger-seconds", str(args.launch_stagger_seconds)]
    if len(args.gpu_ids) >= 4:
        _run(
            "action_smoke",
            action_base + ["--mode", "smoke", "--conditions", "current", "wrong", "svd_r97", "svd_r170", "--gpu-ids", *map(str, args.gpu_ids[:4])],
            root=root,
            env=_launcher_env(),
        )
    else:
        _run(
            "action_smoke_endpoints",
            action_base + ["--mode", "smoke", "--conditions", "current", "wrong", "--gpu-ids", *map(str, args.gpu_ids[:2])],
            root=root,
            env=_launcher_env(),
        )
        _run(
            "action_smoke_projected",
            action_base + ["--mode", "smoke", "--conditions", "svd_r97", "svd_r170", "--gpu-ids", *map(str, args.gpu_ids[:2])],
            root=root,
            env=_launcher_env(),
        )
    if len(args.gpu_ids) >= 4:
        _parallel(
            "endpoints",
            [
                (action_base + ["--mode", "full", "--conditions", "current", "wrong", "--gpu-ids", *map(str, args.gpu_ids[:2])], _launcher_env()),
                (world_base + ["--conditions", "current", "wrong", "--gpu-ids", *map(str, args.gpu_ids[2:4])], _launcher_env()),
            ],
            root=root,
        )
    else:
        _run("action_endpoints", action_base + ["--mode", "full", "--conditions", "current", "wrong", "--gpu-ids", *map(str, args.gpu_ids[:2])], root=root, env=_launcher_env())
        _run("world_endpoints", world_base + ["--conditions", "current", "wrong", "--gpu-ids", *map(str, args.gpu_ids[:2])], root=root, env=_launcher_env())
    _run("endpoint_gate", [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.endpoint_gate", "--output-root", str(root)], root=root)
    endpoint = json.loads((root / "endpoint_gate.json").read_text(encoding="utf-8"))
    if endpoint.get("passed") is not True:
        classification = str(endpoint["classification"])
        _publish_stop(root, classification, "The registered pre-projection endpoint gate failed.", python)
        return
    if len(args.gpu_ids) >= 4:
        _parallel(
            "projected",
            [
                (action_base + ["--mode", "full", "--conditions", "svd_r97", "svd_r170", "--gpu-ids", *map(str, args.gpu_ids[:2])], _launcher_env()),
                (world_base + ["--conditions", "svd_r97", "svd_r170", "--gpu-ids", *map(str, args.gpu_ids[2:4])], _launcher_env()),
            ],
            root=root,
        )
    else:
        _run("action_projected", action_base + ["--mode", "full", "--conditions", "svd_r97", "svd_r170", "--gpu-ids", *map(str, args.gpu_ids[:2])], root=root, env=_launcher_env())
        _run("world_projected", world_base + ["--conditions", "svd_r97", "svd_r170", "--gpu-ids", *map(str, args.gpu_ids[:2])], root=root, env=_launcher_env())
    _run("aggregate", [python, "-m", "experiments.asre_diagnosis.salvage_b_v2.aggregate", "--output-root", str(root)], root=root)
    print(f"Salvage B v2 completed. Final report: {root / 'aggregate/result_summary_for_gpt.md'}", flush=True)
    print("No later ASRE experiment was launched.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--launch-stagger-seconds", type=float, default=70.0)
    args = parser.parse_args()
    if len(args.gpu_ids) < 2 or len(args.gpu_ids) > 4 or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("Salvage-B v2 requires two to four distinct physical GPU IDs.")
    run(args)


if __name__ == "__main__":
    main()
