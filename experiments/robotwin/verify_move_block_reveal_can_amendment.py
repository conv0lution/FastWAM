#!/usr/bin/env python3
"""Strictly verify the frozen task plus one sealed post-freeze amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_file(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed checksum line {line_number}: {line!r}")
        expected, raw_path = parts
        raw_path = raw_path.lstrip("*")
        entries.append((expected, Path(raw_path)))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--freeze-name", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    amendment_path = args.amendment.resolve()
    amendment = json.loads(amendment_path.read_text())
    if amendment.get("freeze_name") != args.freeze_name:
        raise RuntimeError("Amendment freeze_name does not match requested freeze")
    if amendment.get("classification") != "POST_FREEZE_ENGINEERING_ONLY":
        raise RuntimeError("Amendment is not classified as engineering-only")

    checksum_file = run_root / "metadata" / f"{args.freeze_name}.sha256"
    expected_seal = amendment["original_freeze_checksum_file_sha256"]
    actual_seal = sha256(checksum_file)
    if actual_seal != expected_seal:
        raise RuntimeError(
            f"Original freeze checksum file changed: {actual_seal} != {expected_seal}"
        )

    allowed: dict[str, dict[str, str]] = {}
    for record in amendment["allowlisted_file_changes"]:
        relative = str(record["path"])
        if relative in allowed:
            raise RuntimeError(f"Duplicate amendment allowlist path: {relative}")
        allowed[relative] = record

    seen_allowed: set[str] = set()
    mismatches: set[str] = set()
    for frozen_sha, path in parse_checksum_file(checksum_file):
        if not path.is_file():
            raise RuntimeError(f"Frozen file is missing: {path}")
        actual = sha256(path)
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative = ""
        if relative in allowed:
            record = allowed[relative]
            seen_allowed.add(relative)
            if frozen_sha != record["frozen_sha256"]:
                raise RuntimeError(
                    f"Frozen SHA disagrees with amendment for {relative}: "
                    f"{frozen_sha} != {record['frozen_sha256']}"
                )
            if actual != record["amended_sha256"]:
                raise RuntimeError(
                    f"Amended file changed for {relative}: "
                    f"{actual} != {record['amended_sha256']}"
                )
            if actual != frozen_sha:
                mismatches.add(relative)
        elif actual != frozen_sha:
            raise RuntimeError(
                f"Non-allowlisted frozen file changed: {path}: {actual} != {frozen_sha}"
            )

    if seen_allowed != set(allowed):
        missing = sorted(set(allowed) - seen_allowed)
        raise RuntimeError(f"Amendment paths absent from original freeze: {missing}")
    if mismatches != set(allowed):
        raise RuntimeError(
            "Freeze mismatch set is not exactly the amendment allowlist: "
            f"mismatches={sorted(mismatches)} allowlist={sorted(allowed)}"
        )

    amendment_seal = amendment_path.with_suffix(".sha256")
    for expected, path in parse_checksum_file(amendment_seal):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Post-freeze amendment artifact failed verification: {path}")

    print(
        "POST_FREEZE_AMENDMENT_OK "
        f"freeze={args.freeze_name} allowlist={','.join(sorted(allowed))}"
    )


if __name__ == "__main__":
    main()
