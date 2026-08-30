"""Launch one isolated GPU per native-world condition."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .definitions import CONDITIONS


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def launch(args: argparse.Namespace) -> None:
    if any(condition not in CONDITIONS for condition in args.conditions):
        raise ValueError("Unknown native-world condition.")
    if len(args.gpu_ids) < len(args.conditions):
        raise ValueError("World launch requires one GPU per condition.")
    logs = args.output_root.resolve() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for slot, condition in enumerate(args.conditions):
        completion = args.output_root / "world" / condition / "completion.json"
        if completion.is_file():
            continue
        command = [
            str(args.python.resolve()),
            "-m",
            "experiments.asre_diagnosis.salvage_b_v2.world_worker",
            "--preflight",
            str(args.preflight.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--condition",
            condition,
        ]
        if args.max_samples is not None:
            command.extend(("--max-samples", str(args.max_samples)))
        log = logs / f"world.{condition}.log"
        handle = log.open("w", encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[slot])
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            env.pop(key, None)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((condition, process, handle, log))
        if args.launch_stagger_seconds and slot + 1 < len(args.conditions):
            time.sleep(args.launch_stagger_seconds)
    failure = None
    while children:
        for child in list(children):
            condition, process, handle, log = child
            code = process.poll()
            if code is None:
                continue
            children.remove(child)
            handle.close()
            if code:
                failure = f"Native world condition {condition} exited {code}; inspect {log}."
                for _, other, other_handle, _ in children:
                    if other.poll() is None:
                        os.killpg(other.pid, signal.SIGTERM)
                    other_handle.close()
                children.clear()
                break
        if children:
            time.sleep(2)
    if failure:
        raise RuntimeError(failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
