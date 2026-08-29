"""Narrow provenance bridge for the reviewed post-fit machinery fix."""

from __future__ import annotations

import subprocess
from pathlib import Path


REVIEWED_ARTIFACT_SOURCE_COMMIT = "d9ed000741e4ad8b40aa1a3d91c594d8e0bd313c"
REVIEWED_BRIDGE_ALLOWED_PATHS = frozenset(
    {
        "experiments/asre_diagnosis/salvage_a/machinery_tests.py",
        "experiments/asre_diagnosis/salvage_a/basis.py",
        "experiments/asre_diagnosis/salvage_a/reviewed_resume.py",
        "experiments/asre_diagnosis/salvage_a/run_salvage_a.py",
        "experiments/asre_diagnosis/tests/test_salvage_a_basis.py",
        "experiments/asre_diagnosis/tests/test_salvage_a_runner.py",
    }
)


def validate_reviewed_artifact_source(
    *, source_commit: str, current_commit: str, project_root: Path
) -> bool:
    """Return whether the exact reviewed pre-machinery bridge is in use."""

    if source_commit == current_commit:
        return False
    if source_commit != REVIEWED_ARTIFACT_SOURCE_COMMIT:
        raise RuntimeError(
            "Refusing artifacts from an unreviewed source commit: "
            f"{source_commit} != {current_commit}."
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, current_commit],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Reviewed Salvage-A artifact source is not an ancestor of HEAD.")
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", f"{source_commit}..{current_commit}"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()
    )
    unexpected = sorted(changed - REVIEWED_BRIDGE_ALLOWED_PATHS)
    if unexpected:
        raise RuntimeError(
            "Reviewed Salvage-A resume bridge contains out-of-scope changes: "
            f"{unexpected}."
        )
    return True
