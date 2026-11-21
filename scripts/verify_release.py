"""Verify a frozen release using local metadata and the Python standard library."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


DEFAULT_MANIFEST = "releases/v0.3.0/release.json"
DEFAULT_CONFIG = "config/hkust-deepseek-flash-code-json.json"
EXPECTED_VERSION = "0.3.0"


class ReleaseVerificationError(ValueError):
    """The recorded release does not match its local source or result registry."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseVerificationError(message)


def _relative_file(root: Path, value: str | Path) -> Path:
    value = str(value)
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(bool(value) and "\x00" not in value and "\\" not in value
             and not path.is_absolute() and not windows.drive and not windows.root
             and ".." not in path.parts and value == path.as_posix()
             and value != ".", "File paths must be normalized project-relative paths.")
    candidate = root / value
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseVerificationError("A referenced file is missing or escapes the project root.") from exc
    _require(resolved.is_file(), "A referenced path is not a regular file.")
    return resolved


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("Cannot read a required JSON object.") from exc
    _require(isinstance(value, dict), "Required JSON content must be an object.")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReleaseVerificationError("Cannot read a release file for hashing.") from exc
    return digest.hexdigest()


def _sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _verify_files(root: Path, recorded: object, expected: set[str], label: str) -> dict[str, str]:
    _require(isinstance(recorded, dict) and bool(recorded), f"{label} must contain a file-hash mapping.")
    actual = {}
    for relative, expected_hash in recorded.items():
        _require(isinstance(relative, str), "File-map keys must be strings.")
        file = _relative_file(root, relative)
        _require(_sha256(expected_hash), f"{label} contains an invalid SHA-256 value.")
        actual[relative] = _hash(file)
        _require(actual[relative] == expected_hash, f"{label} hash mismatch: {relative}")
    _require(set(recorded) == expected, f"{label} does not match the complete on-disk file inventory.")
    return actual


def _literal_version(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ReleaseVerificationError("Cannot parse the package version declaration.") from exc
    declarations = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            declarations.append(node.value)
    _require(len(declarations) == 1 and isinstance(declarations[0], ast.Constant)
             and isinstance(declarations[0].value, str), "Package __version__ must have one literal string declaration.")
    return declarations[0].value


def _verify_results(root: Path, source_sha256: str, version: str) -> tuple[int, int]:
    registry = _read_json(_relative_file(root, "docs/results_registry.json"))
    current = registry.get("current_release")
    _require(isinstance(current, dict) and current.get("version") == "v" + version,
             "Results registry current version differs from the release.")
    _require(current.get("source_sha256") in (None, source_sha256),
             "Results registry assigns a different source to the current release.")
    historical = registry.get("historical_results")
    _require(isinstance(historical, list), "Results registry needs a historical-results list.")
    for row in historical:
        _require(isinstance(row, dict) and row.get("is_current_release_result") is False,
                 "Historical results must not be attributed to the current release.")
        _require(_sha256(row.get("source_sha256")) and row["source_sha256"] != source_sha256,
                 "Historical results must retain their historical source fingerprint.")
    rows = current.get("results")
    _require(isinstance(rows, list) and bool(rows), "Results registry needs current-release task entries.")
    pending = 0
    for row in rows:
        _require(isinstance(row, dict), "Current-release result entries must be objects.")
        if current.get("results_status") == "pending_not_run":
            _require(row.get("status") == "pending_not_run", "Pending release contains a non-pending task result.")
        if row.get("status") == "pending_not_run":
            pending += 1
            _require("score" in row and row["score"] is None
                     and row.get("correct") is None and row.get("rows") is None,
                     "Pending result entries must not contain scores or result counts.")
        elif row.get("score") is not None:
            _require(current.get("source_sha256") == source_sha256
                     and row.get("source_sha256") == source_sha256,
                     "Recorded current-release scores need their current source fingerprint.")
    return len(historical), pending


def verify_release(root: str | Path, manifest_path: str | Path = DEFAULT_MANIFEST) -> dict:
    """Return a compact verified summary, or raise without changing any files.

    The project root may be absolute. All manifest and artifact paths must be
    relative to that root. Package code is parsed, never imported or executed.
    """
    root = Path(root).resolve()
    _require(root.is_dir(), "Project root must be an existing directory.")
    manifest = _read_json(_relative_file(root, manifest_path))
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
             "Unsupported release manifest schema_version.")
    _require(manifest.get("version") == EXPECTED_VERSION, "Release version must be 0.3.0.")
    _require(_sha256(manifest.get("source_sha256")), "Release source_sha256 is invalid.")
    source_paths = {path.relative_to(root).as_posix() for path in (root / "superagent").glob("*.py")}
    source_paths.add("requirements.txt")
    _require("superagent/__init__.py" in source_paths, "Release package is missing __init__.py.")
    source_files = _verify_files(root, manifest.get("source_files"), source_paths, "source_files")
    aggregate = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    _require(aggregate == manifest["source_sha256"], "Aggregate source fingerprint differs from the manifest.")
    config_paths = {path.relative_to(root).as_posix() for path in (root / "config").glob("*.json")}
    configs = _verify_files(root, manifest.get("config_files"), config_paths, "config_files")
    _require(manifest.get("default_config") == DEFAULT_CONFIG and DEFAULT_CONFIG in configs,
             "Release default_config must select the recorded Flash code-json configuration.")
    package_version = _literal_version(_relative_file(root, "superagent/__init__.py"))
    _require(package_version == manifest["version"], "Package version differs from the manifest.")
    historical, pending = _verify_results(root, aggregate, package_version)
    return {"verified": True, "version": package_version, "source_sha256": aggregate,
            "source_files": len(source_files), "config_files": len(configs),
            "historical_results": historical, "pending_results": pending}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify local v0.3.0 release files and result attribution; no API calls.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Project-relative manifest path")
    args = parser.parse_args(argv)
    try:
        result = verify_release(args.project_root, args.manifest)
    except ReleaseVerificationError as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
