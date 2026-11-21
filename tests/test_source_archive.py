"""Synthetic source/archive files only; never load models or credentials."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "manage_source_archive.py"
SPEC = importlib.util.spec_from_file_location("source_archive_tool", SCRIPT)
archive_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive_tool)


@pytest.fixture
def sample(tmp_path):
    project = tmp_path / "project"
    root = project / "runs" / "old" / "source_snapshot"
    (root / "superagent").mkdir(parents=True)
    (root / "superagent" / "__init__.py").write_text('VERSION = "old"\n', encoding="utf-8")
    (root / "requirements.txt").write_text("fake-package==1.0\n", encoding="utf-8")
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (root.parent / "manifest.json").write_text(json.dumps({"files": files, "source_sha256": fingerprint}), encoding="utf-8")
    roots = project / "roots.json"
    roots.write_text(json.dumps(["runs/old/source_snapshot"]), encoding="utf-8")
    return project, root, roots


def create(sample):
    project, _, roots = sample
    result = archive_tool.archive_sources(project, "archives/index.json", "archives/source.zip", roots)
    return result, project / "archives" / "index.json", project / "archives" / "source.zip"


def reindex(index, archive):
    document = json.loads(index.read_text())
    document["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    document["index_sha256"] = archive_tool._index_hash(document)
    index.write_text(json.dumps(document), encoding="utf-8")


def test_roundtrip_preserves_source_and_manifest_then_restores_exact_bytes(sample):
    project, root, _ = sample
    original = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest_bytes = (root.parent / "manifest.json").read_bytes()
    result, index, _ = create(sample)
    assert result["status"] == "archived" and result["source_directories_deleted"] is False
    assert root.is_dir()
    assert archive_tool.verify_archive(project, index)["files"] == 2
    document = json.loads(index.read_text())
    assert document["roots"][0]["manifests"][0]["source_sha256"]
    shutil.rmtree(root)  # Only this disposable test fixture is removed.
    assert archive_tool.verify_archive(project, index)["status"] == "verified"
    restored = archive_tool.restore_source(project, index, "runs/old/source_snapshot")
    assert restored["executed_source"] is False
    assert {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()} == original
    assert (root.parent / "manifest.json").read_bytes() == manifest_bytes


def test_existing_archive_index_or_restore_target_cannot_be_overwritten(sample):
    project, _, _ = sample
    _, index, archive = create(sample)
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="must not exist"):
        create(sample)
    with pytest.raises(ValueError, match="already exists"):
        archive_tool.restore_source(project, index, "runs/old/source_snapshot")
    assert archive.read_bytes() == original


@pytest.mark.parametrize("root", ["../outside/source_snapshot", "/runs/source_snapshot", "runs/old/../source_snapshot",
                                     "runs/old/source", "superagent/source_snapshot", "runs\\old\\source_snapshot"])
def test_invalid_source_roots_rejected_before_archive_creation(sample, root):
    project, _, roots = sample
    roots.write_text(json.dumps([root]), encoding="utf-8")
    with pytest.raises(ValueError):
        create(sample)
    assert not (project / "archives" / "source.zip").exists()


def test_archive_destination_outside_project_is_rejected(sample):
    project, _, roots = sample
    with pytest.raises(ValueError, match="inside project"):
        archive_tool.archive_sources(project, "archives/index.json", project.parent / "outside.zip", roots)


def test_source_different_from_manifest_is_rejected(sample):
    _, root, _ = sample
    (root / "requirements.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its manifest"):
        create(sample)


def test_index_tampering_is_rejected(sample):
    project, _, _ = sample
    _, index, _ = create(sample)
    document = json.loads(index.read_text())
    document["roots"][0]["files"]["requirements.txt"]["size"] += 1
    index.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Index checksum"):
        archive_tool.verify_archive(project, index)


def test_zip_tampering_is_rejected_without_creating_restore_target(sample):
    project, root, _ = sample
    _, index, archive = create(sample)
    shutil.rmtree(root)
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Archive checksum"):
        archive_tool.restore_source(project, index, "runs/old/source_snapshot")
    assert not root.exists()


@pytest.mark.parametrize("name", ["../escape.py", "runs/old/source_snapshot/extra.py", "runs/old/source_snapshot/requirements.txt"])
def test_zip_traversal_unknown_and_duplicate_members_rejected_even_with_updated_checksums(sample, name):
    project, _, _ = sample
    _, index, archive = create(sample)
    if name.endswith("requirements.txt"):
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(archive, "a") as zipped:
                zipped.writestr(name, "extra")
    else:
        with zipfile.ZipFile(archive, "a") as zipped:
            zipped.writestr(name, "extra")
    reindex(index, archive)
    with pytest.raises(ValueError, match="traversal|Unknown ZIP|Duplicate ZIP"):
        archive_tool.verify_archive(project, index)


def test_changed_member_content_rejected_with_updated_archive_checksum(sample):
    project, _, _ = sample
    _, index, archive = create(sample)
    with zipfile.ZipFile(archive) as zipped:
        files = {item.filename: zipped.read(item) for item in zipped.infolist()}
    name = "runs/old/source_snapshot/requirements.txt"
    files[name] = b"x" * len(files[name])
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in files.items():
            zipped.writestr(name, data)
    reindex(index, archive)
    with pytest.raises(ValueError, match="member checksum"):
        archive_tool.verify_archive(project, index)


def test_changed_original_manifest_rejected(sample):
    project, root, _ = sample
    _, index, _ = create(sample)
    with (root.parent / "manifest.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Original manifest checksum"):
        archive_tool.verify_archive(project, index)


def test_zip_reparse_member_rejected(sample):
    project, _, _ = sample
    _, index, archive = create(sample)
    with zipfile.ZipFile(archive) as zipped:
        files = [(item, zipped.read(item)) for item in zipped.infolist()]
    files[0][0].external_attr |= 0x400
    with zipfile.ZipFile(archive, "w") as zipped:
        for info, data in files:
            zipped.writestr(info, data)
    reindex(index, archive)
    with pytest.raises(ValueError, match="ZIP links"):
        archive_tool.verify_archive(project, index)


def test_symlink_source_is_rejected(sample):
    _, root, _ = sample
    link = root / "link.py"
    try:
        link.symlink_to(root / "requirements.txt")
    except OSError:
        pytest.skip("Creating symlinks is not permitted in this test environment")
    with pytest.raises(ValueError, match="Symlink/reparse"):
        create(sample)


def test_reparse_attribute_is_rejected_without_following_path(monkeypatch, tmp_path):
    marker = tmp_path / "junction"
    original = Path.lstat
    def lstat(path):
        if path == marker:
            return type("Info", (), {"st_mode": stat.S_IFDIR, "st_file_attributes": 0x400})()
        return original(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="Symlink/reparse"):
        archive_tool._reject_links(marker)


def test_cli_archive_and_verify(sample):
    project, _, roots = sample
    common = ["--project-root", str(project), "--index", "archives/index.json"]
    created = subprocess.run([sys.executable, str(SCRIPT), "archive", *common,
                              "--archive", "archives/source.zip", "--roots-file", str(roots)], capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    verified = subprocess.run([sys.executable, str(SCRIPT), "verify", *common], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["status"] == "verified"
