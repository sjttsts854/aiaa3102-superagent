"""Release verification uses tiny local fixtures, never models or API calls."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("release_verifier", Path(__file__).parents[1] / "scripts/verify_release.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_manifest(root):
    def mapping(paths):
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    sources = mapping(sorted((root / "superagent").glob("*.py")) + [root / "requirements.txt"])
    manifest = {"schema_version": 1, "version": "0.3.0", "source_files": sources,
                "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                "config_files": mapping(sorted((root / "config").glob("*.json"))),
                "default_config": verifier.DEFAULT_CONFIG,
                "additional_metadata": {"tests_passed": True}}
    write_json(root / verifier.DEFAULT_MANIFEST, manifest)
    return manifest


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "superagent").mkdir(parents=True)
    (root / "superagent/__init__.py").write_text('__version__ = "0.3.0"\nraise RuntimeError("Do not import package code")\n', encoding="utf-8")
    (root / "superagent/worker.py").write_text("# worker fixture\n", encoding="utf-8")
    (root / "requirements.txt").write_text("example==1\n", encoding="utf-8")
    write_json(root / verifier.DEFAULT_CONFIG, {"model": "Flash fixture"})
    write_json(root / "config/other.json", {"model": "Other fixture"})
    write_json(root / "docs/results_registry.json", {
        "current_release": {"version": "v0.3.0", "source_sha256": None, "results_status": "pending_not_run",
                            "results": [{"task": "coupon", "status": "pending_not_run", "score": None,
                                         "correct": None, "rows": None}]},
        "historical_results": [{"source_sha256": "a" * 64, "is_current_release_result": False,
                                "score": .75}]})
    make_manifest(root)
    return root


def test_release_verifies_without_importing_package(project):
    result = verifier.verify_release(project, verifier.DEFAULT_MANIFEST)
    assert result["verified"] is True
    assert result["version"] == "0.3.0"
    assert result["source_files"] == 3 and result["config_files"] == 2
    assert result["historical_results"] == result["pending_results"] == 1


def test_runtime_tampering_is_rejected(project):
    (project / "superagent/worker.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(verifier.ReleaseVerificationError, match="source_files hash mismatch"):
        verifier.verify_release(project)


@pytest.mark.parametrize("relative", ["superagent/new_runtime.py", "config/new.json"])
def test_new_file_cannot_be_omitted_from_manifest(project, relative):
    (project / relative).write_text("{}\n", encoding="utf-8")
    with pytest.raises(verifier.ReleaseVerificationError, match="complete on-disk"):
        verifier.verify_release(project)


def test_configuration_tampering_is_rejected(project):
    write_json(project / verifier.DEFAULT_CONFIG, {"model": "changed"})
    with pytest.raises(verifier.ReleaseVerificationError, match="config_files hash mismatch"):
        verifier.verify_release(project)


def test_aggregate_fingerprint_is_recomputed(project):
    manifest = make_manifest(project)
    manifest["source_sha256"] = "b" * 64
    write_json(project / verifier.DEFAULT_MANIFEST, manifest)
    with pytest.raises(verifier.ReleaseVerificationError, match="Aggregate"):
        verifier.verify_release(project)


def test_package_version_must_match_even_with_matching_file_hashes(project):
    (project / "superagent/__init__.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8")
    make_manifest(project)
    with pytest.raises(verifier.ReleaseVerificationError, match="Package version"):
        verifier.verify_release(project)


@pytest.mark.parametrize("bad_path", ["../outside.py", "superagent/../outside.py", "/outside.py",
    "C:/outside.py", "C:outside.py", "C:\\outside.py", "\\\\server\\outside.py"])
def test_manifest_rejects_absolute_and_parent_paths(project, bad_path):
    manifest = make_manifest(project)
    manifest["source_files"][bad_path] = "b" * 64
    write_json(project / verifier.DEFAULT_MANIFEST, manifest)
    with pytest.raises(verifier.ReleaseVerificationError, match="project-relative"):
        verifier.verify_release(project)


def test_manifest_itself_cannot_escape_root(project):
    with pytest.raises(verifier.ReleaseVerificationError, match="project-relative"):
        verifier.verify_release(project, "../release.json")


def test_symlink_cannot_escape_root(project, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("# outside fixture\n", encoding="utf-8")
    linked = project / "superagent/link.py"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("This environment does not permit creating symlinks")
    make_manifest(project)
    with pytest.raises(verifier.ReleaseVerificationError, match="escapes"):
        verifier.verify_release(project)


@pytest.mark.parametrize("change", ["flag", "hash", "pending_score", "registry_version"])
def test_historical_results_cannot_masquerade_as_current(project, change):
    path = project / "docs/results_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((project / verifier.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    if change == "flag":
        registry["historical_results"][0]["is_current_release_result"] = True
    elif change == "hash":
        registry["historical_results"][0]["source_sha256"] = manifest["source_sha256"]
    elif change == "pending_score":
        registry["current_release"]["results"][0]["score"] = .75
    else:
        registry["current_release"]["version"] = "v0.2.0"
    write_json(path, registry)
    with pytest.raises(verifier.ReleaseVerificationError):
        verifier.verify_release(project)


def test_cli_returns_small_json_and_nonzero_on_failure(project, capsys):
    args = ["--project-root", str(project), "--manifest", verifier.DEFAULT_MANIFEST]
    assert verifier.main(args) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    (project / "superagent/worker.py").write_text("# tampered", encoding="utf-8")
    assert verifier.main(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is False
    assert str(project) not in json.dumps(result)
