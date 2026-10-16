#!/usr/bin/env python3
"""Reject generated deliverables in the Git index; ignored local files are untouched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

SOURCE_DIRS = {"nucleus_rl", "scripts", "tests", "baselines", "protocols", "configs"}
ARTIFACT_DIRS = {"reports", "runs", "data", "checkpoints", "results", "outputs", "artifacts", "local-notes", "references"}
SOURCE_SUFFIXES = {".py", ".sh", ".json", ".toml", ".yaml", ".yml", ".ipynb"}
ROOT_FILES = {".gitignore", ".gitattributes", "pyproject.toml", "setup.cfg", "setup.py", "Makefile", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"}
EXECUTION_KEYS = {"execution", "ExecuteTime", "executionInfo", "outputId", "base_uri"}
RESULT_HEADING = re.compile(r"^\s*#{1,6}\s+(?:(?:baseline|experiment|training|evaluation|benchmark)\s+)?results?\b", re.I | re.M)


def path_error(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return "invalid repository path"
    if path.parts[0] == "protocols" and re.fullmatch(r".*_grpo_v.*\.json", path.name):
        return "generated dataset protocol belongs under ignored data or runs"
    if any(part in ARTIFACT_DIRS for part in path.parts[:-1]):
        return "generated artifacts and research notes belong outside Git"
    if path.name.lower() == "readme.md" and (len(path.parts) == 1 or path.parts[0] in SOURCE_DIRS):
        return None
    if name in ROOT_FILES or (len(path.parts) == 1 and re.fullmatch(r"requirements(?:-[\w-]+)?\.txt", name)):
        return None
    if path.parts[:2] == (".github", "workflows") and path.suffix in {".yml", ".yaml"}:
        return None
    if path.parts[0] in SOURCE_DIRS and path.suffix in SOURCE_SUFFIXES:
        return None
    return "only source, configuration, tests, licenses, and setup READMEs are allowed"


def has_execution_metadata(value) -> bool:
    if isinstance(value, dict):
        return any(k in EXECUTION_KEYS or has_execution_metadata(v) for k, v in value.items())
    if isinstance(value, list):
        return any(has_execution_metadata(v) for v in value)
    return False


def content_errors(name: str, raw: bytes) -> list[str]:
    error = path_error(name)
    if error:
        return [error]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ["binary content is not source code"]
    if "\0" in text:
        return ["binary content is not source code"]
    if not name.endswith(".ipynb"):
        return []
    try:
        notebook = json.loads(text)
        if not isinstance(notebook, dict):
            raise ValueError("notebook must be an object")
        cells = notebook["cells"]
        if not isinstance(cells, list):
            raise ValueError("cells must be a list")
    except (ValueError, KeyError, TypeError):
        return ["invalid notebook JSON"]
    errors = []
    if has_execution_metadata(notebook.get("metadata", {})):
        errors.append("notebook has execution metadata")
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {i}: invalid cell")
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            errors.append(f"cell {i}: clear saved outputs and execution count")
        if cell.get("attachments"):
            errors.append(f"cell {i}: embedded attachments belong outside Git")
        if has_execution_metadata(cell.get("metadata", {})):
            errors.append(f"cell {i}: remove execution metadata")
        source = cell.get("source", [])
        if not isinstance(source, (str, list)) or (isinstance(source, list) and any(not isinstance(s, str) for s in source)):
            errors.append(f"cell {i}: invalid cell source")
            continue
        if cell.get("cell_type") == "markdown" and RESULT_HEADING.search("".join(source)):
            errors.append(f"cell {i}: move the results section to local notes")
    return errors


def check_index(root: Path) -> list[str]:
    """Read indexed blob bytes, so unstaged edits cannot hide a staged output."""
    records = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    errors = []
    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        name = raw_name.decode("utf-8", errors="surrogateescape")
        if stage != "0" or mode not in {"100644", "100755"}:
            errors.append(f"{name}: unresolved merge, symlink, or submodule is not an allowed source file")
            continue
        raw = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
        errors.extend(f"{name}: {reason}" for reason in content_errors(name, raw))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check_index(args.root)
    if errors:
        print("Repository content check failed:\n" + "\n".join(f"- {e}" for e in errors))
        return 1
    print("Git index contains source/configuration/tests only; notebook outputs are empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
