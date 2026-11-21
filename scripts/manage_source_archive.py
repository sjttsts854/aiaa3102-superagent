"""Archive historical source trees without deleting or executing anything.

The index checksum detects accidental modification; it is not a signature.
Historical ledgers and manifests stay at their original paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


EXPLICIT_ROOTS = {
    "runs/final_release_round3/source",
    "runs/three_tier_20261121_001/source",
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_hash(document: dict) -> str:
    payload = {k: v for k, v in document.items() if k != "index_sha256"}
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or any(character in value for character in '\\:\x00<>"|?*'):
        raise ValueError("Expected a safe project-relative POSIX path")
    parts = value.split("/")
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in parts):
        raise ValueError("Path traversal or ambiguous path is forbidden")
    # Avoid Windows device names, including names with extensions.
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)]}
    if any(part.split(".")[0].upper() in reserved for part in parts):
        raise ValueError("Reserved Windows path is forbidden")
    if PurePosixPath(value).is_absolute():
        raise ValueError("Absolute archive paths are forbidden")
    return value


def _reject_links(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Symlink/reparse paths are forbidden")


def _project(value: str | Path) -> Path:
    path = Path(os.path.abspath(value))
    for ancestor in reversed([path, *path.parents]):
        if ancestor.exists() or ancestor.is_symlink():
            _reject_links(ancestor)
    if not path.is_dir():
        raise ValueError("Project root must be an existing directory")
    return path


def _inside(project: Path, relative: str) -> Path:
    path = project.joinpath(*_relative(relative).split("/"))
    for ancestor in reversed([path, *path.parents]):
        if ancestor == project or project in ancestor.parents:
            if ancestor.exists() or ancestor.is_symlink():
                _reject_links(ancestor)
    if not path.resolve().is_relative_to(project.resolve()):
        raise ValueError("Path escapes project root")
    return path


def _argument_path(project: Path, value: str | Path) -> tuple[Path, str]:
    raw = Path(value)
    path = Path(os.path.abspath(raw if raw.is_absolute() else project / raw))
    try:
        relative = path.relative_to(project).as_posix()
    except ValueError:
        raise ValueError("Archive/index path must remain inside project root") from None
    return _inside(project, relative), relative


def _source_root(value: str) -> str:
    value = _relative(value)
    parts = value.split("/")
    if not (len(parts) >= 2 and parts[0] == "runs" and
            (parts[-1] == "source_snapshot" or value in EXPLICIT_ROOTS)):
        raise ValueError("Only approved historical source roots below runs may be archived")
    return value


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _manifest_records(project: Path, source: Path, files: dict) -> list[dict]:
    candidates = [source.parent / "manifest.json"]
    if source.relative_to(project).as_posix() == "runs/final_release_round3/source":
        candidates.append(source.parent / "freeze.json")
    if source.relative_to(project).as_posix() == "runs/three_tier_20261121_001/source":
        candidates.append(source.parent / "campaign" / "planner_01" / "manifest.json")
    records = []
    for candidate in candidates:
        candidate = _inside(project, candidate.relative_to(project).as_posix())
        if not candidate.exists():
            continue
        raw = candidate.read_bytes()
        manifest = json.loads(raw.decode("utf-8-sig"))
        expected = manifest.get("files")
        fingerprint = manifest.get("source_sha256")
        if not isinstance(expected, dict) or not expected or not isinstance(fingerprint, str):
            raise ValueError("Source manifest lacks file/source fingerprints")
        if _digest(json.dumps(expected, sort_keys=True).encode()) != fingerprint:
            raise ValueError("Source manifest aggregate fingerprint differs")
        for relative, digest in expected.items():
            _relative(relative)
            if files.get(relative, {}).get("sha256") != digest:
                raise ValueError("Source file differs from its manifest")
        records.append({"path": candidate.relative_to(project).as_posix(),
                        "sha256": _digest(raw), "source_sha256": fingerprint,
                        "files": expected})
    return records


def _members(document: dict) -> dict:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported archive index")
    if document.get("index_sha256") != _index_hash(document):
        raise ValueError("Index checksum differs")
    roots = document.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("Archive index needs source roots")
    members, seen_roots, seen_members = {}, set(), set()
    for root in roots:
        source = _source_root(root["source_root"])
        normalized = source.casefold()
        if normalized in seen_roots or any(normalized.startswith(old + "/") or old.startswith(normalized + "/") for old in seen_roots):
            raise ValueError("Duplicate or overlapping source roots")
        seen_roots.add(normalized)
        files = root.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Each source root needs files")
        for relative, record in files.items():
            member = source + "/" + _relative(relative)
            if member.casefold() in seen_members:
                raise ValueError("Duplicate archive member")
            if (not isinstance(record, dict) or type(record.get("size")) is not int or record["size"] < 0
                    or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64):
                raise ValueError("Invalid archived file metadata")
            seen_members.add(member.casefold())
            members[member] = record
    return members


def _verify_document(project: Path, document: dict) -> dict:
    expected = _members(document)
    archive = _inside(project, document["archive"])
    if _file_hash(archive) != document.get("archive_sha256"):
        raise ValueError("Archive checksum differs")
    for root in document["roots"]:
        _inside(project, root["source_root"])
        for manifest in root.get("manifests", []):
            original = _inside(project, manifest["path"])
            if _file_hash(original) != manifest["sha256"]:
                raise ValueError("Original manifest checksum differs")
            expected_source = manifest["files"]
            if _digest(json.dumps(expected_source, sort_keys=True).encode()) != manifest["source_sha256"]:
                raise ValueError("Manifest aggregate fingerprint differs")
            if any(root["files"].get(name, {}).get("sha256") != digest for name, digest in expected_source.items()):
                raise ValueError("Archive index differs from manifest file fingerprints")
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        for member in zipped.infolist():
            name = _relative(member.filename)
            if name.casefold() in seen:
                raise ValueError("Duplicate ZIP member")
            seen.add(name.casefold())
            mode = member.external_attr >> 16
            if (member.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                    or member.external_attr & 0x400 or member.flag_bits & 1):
                raise ValueError("ZIP links, special entries and encrypted members are forbidden")
            if name not in expected:
                raise ValueError("Unknown ZIP member")
            record = expected[name]
            if member.file_size != record["size"]:
                raise ValueError("ZIP member size differs")
            digest, size = hashlib.sha256(), 0
            with zipped.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if size != record["size"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("ZIP member checksum differs")
        if seen != {name.casefold() for name in expected}:
            raise ValueError("ZIP is missing indexed members")
    return {"status": "verified", "roots": len(document["roots"]), "files": len(expected),
            "archive_sha256": document["archive_sha256"], "index_sha256": document["index_sha256"]}


def archive_sources(project_root, index_path, archive_path, roots_file) -> dict:
    project = _project(project_root)
    index, index_relative = _argument_path(project, index_path)
    archive, archive_relative = _argument_path(project, archive_path)
    if index == archive or index.exists() or archive.exists():
        raise ValueError("Archive and index destinations must be distinct and must not exist")
    roots = _read_json(Path(roots_file))
    if not isinstance(roots, list) or not roots:
        raise ValueError("Roots file must contain a nonempty JSON array")
    records = []
    for value in roots:
        relative = _source_root(value)
        source = _inside(project, relative)
        if not source.is_dir():
            raise ValueError("Source root must be an existing directory")
        if any(name == relative or name.startswith(relative + "/") for name in (index_relative, archive_relative)):
            raise ValueError("Archive/index cannot be written inside an archived source tree")
        files = {}
        for folder, directories, names in os.walk(source, followlinks=False):
            for name in directories + names:
                path = Path(folder) / name
                _reject_links(path)
            for name in sorted(names):
                path = Path(folder) / name
                if not stat.S_ISREG(path.stat().st_mode):
                    raise ValueError("Only regular source files may be archived")
                member = _relative(path.relative_to(source).as_posix())
                files[member] = {"size": path.stat().st_size, "sha256": _file_hash(path)}
        records.append({"source_root": relative, "files": files,
                        "manifests": _manifest_records(project, source, files)})
    document = {"schema_version": 1, "archive": archive_relative, "archive_sha256": "",
                "roots": records, "deletes_source": False}
    document["index_sha256"] = _index_hash(document)
    members = _members(document)  # Validate duplicates and overlaps before writing.
    archive.parent.mkdir(parents=True, exist_ok=True)
    index.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, mode="x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(members):
            zipped.write(_inside(project, name), arcname=name)
    document["archive_sha256"] = _file_hash(archive)
    document["index_sha256"] = _index_hash(document)
    result = _verify_document(project, document)
    with index.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=True, sort_keys=True, indent=2)
        stream.write("\n")
    # Include the persisted index in the final check, not only its in-memory copy.
    verify_archive(project, index)
    return {**result, "status": "archived", "index": index_relative, "archive": archive_relative,
            "source_directories_deleted": False}


def verify_archive(project_root, index_path) -> dict:
    project = _project(project_root)
    index, _ = _argument_path(project, index_path)
    return _verify_document(project, _read_json(index))


def restore_source(project_root, index_path, source_root) -> dict:
    project = _project(project_root)
    index, _ = _argument_path(project, index_path)
    document = _read_json(index)
    _verify_document(project, document)
    relative = _source_root(source_root)
    records = [root for root in document["roots"] if root["source_root"] == relative]
    if len(records) != 1:
        raise ValueError("Requested source root is not in the index")
    target = _inside(project, relative)
    if target.exists():
        raise ValueError("Restore target already exists; refusing to overwrite")
    # Verify all ZIP entries before creating anything. Buffer this root so no
    # archive change can inject unverified bytes into the restored source tree.
    contents = {}
    with zipfile.ZipFile(_inside(project, document["archive"])) as zipped:
        for name, record in records[0]["files"].items():
            data = zipped.read(relative + "/" + name)
            if len(data) != record["size"] or _digest(data) != record["sha256"]:
                raise ValueError("Archive changed before restoration")
            contents[name] = data
    _inside(project, relative)
    target.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        path = _inside(project, relative + "/" + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return {"status": "restored", "source_root": relative, "files": len(contents), "executed_source": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("archive", "verify", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--project-root", required=True, type=Path)
        command.add_argument("--index", required=True, type=Path)
        if name == "archive":
            command.add_argument("--archive", required=True, type=Path)
            command.add_argument("--roots-file", required=True, type=Path)
        if name == "restore":
            command.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "archive":
            result = archive_sources(args.project_root, args.index, args.archive, args.roots_file)
        elif args.command == "verify":
            result = verify_archive(args.project_root, args.index)
        else:
            result = restore_source(args.project_root, args.index, args.source_root)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"Archive operation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
