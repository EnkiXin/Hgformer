"""Prepare the full RecBole-CDR Douban data used by Hgformer.

The original RecBole S3 object currently returns HTTP 403. A byte-identical
2024 Wayback snapshot remains available, so the archive and every extracted
interaction file are checked against pinned SHA256 digests. This is the large
RecBole-CDR release, not the much smaller CoPD Douban subset.

Filtering and train/validation/test splitting are intentionally left to the
shared RecBole configuration. Source user tokens are preserved across Book,
Movie, and Music for later cross-domain experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import urllib.request
import zipfile
from typing import Optional

try:
    from slrec_experiments.dataset_registry import (
        DATASETS,
        DOUBAN_ORIGINAL_URL,
        DOUBAN_PINNED_URL,
        RECBOLE_CDR_HOMEPAGE,
    )
except ModuleNotFoundError:  # Direct ``python slrec_experiments/<file>.py``.
    from dataset_registry import (  # type: ignore
        DATASETS,
        DOUBAN_ORIGINAL_URL,
        DOUBAN_PINNED_URL,
        RECBOLE_CDR_HOMEPAGE,
    )


_DOUBAN_RECORDS = tuple(record for record in DATASETS if record.dataset.startswith("Douban"))
ARCHIVE_URL = DOUBAN_PINNED_URL
ARCHIVE_BYTES = _DOUBAN_RECORDS[0].download_bytes
ARCHIVE_SHA256 = str(_DOUBAN_RECORDS[0].download_sha256)
DOMAINS = {
    record.dataset: {
        "member": record.raw_filename,
        "bytes": record.atomic_bytes,
        "sha256": record.atomic_sha256,
        "role": record.roles[0],
        "release": record.release,
    }
    for record in _DOUBAN_RECORDS
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(path: pathlib.Path, expected_bytes: int, expected_sha256: str) -> None:
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected size for {path}: {actual_bytes} (expected {expected_bytes})"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Unexpected SHA256 for {path}: {actual_sha256} "
            f"(expected {expected_sha256})"
        )


def _obtain_archive(
    output_root: pathlib.Path, archive: Optional[pathlib.Path]
) -> pathlib.Path:
    if archive is not None:
        archive = archive.expanduser().resolve()
        _validate(archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
        return archive

    cache_dir = output_root / ".source"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "Douban.zip"
    if destination.exists():
        _validate(destination, ARCHIVE_BYTES, ARCHIVE_SHA256)
        return destination

    partial = destination.with_suffix(".zip.part")
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "slrec-experiment/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        _validate(partial, ARCHIVE_BYTES, ARCHIVE_SHA256)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _extract_domain(
    archive: zipfile.ZipFile,
    dataset_name: str,
    metadata: dict[str, int | str],
    output_root: pathlib.Path,
    force: bool,
) -> pathlib.Path:
    member_name = str(metadata["member"])
    destination_dir = output_root / dataset_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{dataset_name}.inter"
    partial = destination.with_suffix(".inter.part")

    if destination.exists() and not force:
        try:
            _validate(destination, int(metadata["bytes"]), str(metadata["sha256"]))
        except ValueError as exc:
            raise ValueError(
                f"{destination} is not the full RecBole-CDR file; "
                "use --force to replace it"
            ) from exc
        print(f"Validated existing file: {destination}")
        return destination

    try:
        member = archive.getinfo(member_name)
        if member.file_size != metadata["bytes"]:
            raise ValueError(
                f"Unexpected uncompressed size for {member_name}: {member.file_size}"
            )
        with archive.open(member, "r") as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        _validate(partial, int(metadata["bytes"]), str(metadata["sha256"]))
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print(f"Prepared {destination}: sha256={metadata['sha256']}")
    return destination


def _pending_domains(
    selected: list[tuple[str, dict[str, int | str]]],
    output_root: pathlib.Path,
    force: bool,
) -> list[tuple[str, dict[str, int | str]]]:
    """Validate complete existing members before fetching the shared archive."""

    if force:
        return selected
    pending: list[tuple[str, dict[str, int | str]]] = []
    for dataset_name, metadata in selected:
        destination = output_root / dataset_name / f"{dataset_name}.inter"
        if not destination.exists():
            pending.append((dataset_name, metadata))
            continue
        try:
            _validate(destination, int(metadata["bytes"]), str(metadata["sha256"]))
        except ValueError as exc:
            raise ValueError(
                f"{destination} is not the full RecBole-CDR file; "
                "use --force to replace it"
            ) from exc
        print(f"Validated existing file: {destination}")
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"RecBole-CDR release page: {RECBOLE_CDR_HOMEPAGE}\n"
            f"Original object (currently HTTP 403): {DOUBAN_ORIGINAL_URL}\n"
            f"Pinned byte-identical snapshot: {ARCHIVE_URL}\n"
            "Use --list-sources to print archive/member sizes and SHA256 values.\n"
            "Do not substitute CoPD or RecBole-GNN Social-Datasets Douban files."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("dataset"),
        help="Directory that will contain RecBole dataset subdirectories.",
    )
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        default=None,
        help="Use an already-downloaded Douban.zip instead of fetching Wayback.",
    )
    parser.add_argument(
        "--domain",
        choices=["all", *DOMAINS],
        default="all",
        help="Prepare one domain or all three domains.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print the pinned full-release archive/member registry and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file that is not the pinned full release.",
    )
    args = parser.parse_args()

    if args.list_sources:
        print(
            f"Douban.zip\tbytes={ARCHIVE_BYTES}\tsha256={ARCHIVE_SHA256}\t"
            f"url={ARCHIVE_URL}"
        )
        for dataset_name, metadata in DOMAINS.items():
            print(
                f"{dataset_name}\trole={metadata['role']}\t"
                f"member={metadata['member']}\tbytes={metadata['bytes']}\t"
                f"sha256={metadata['sha256']}"
            )
        return

    selected = list(
        DOMAINS.items()
        if args.domain == "all"
        else [(args.domain, DOMAINS[args.domain])]
    )
    selected = _pending_domains(selected, args.output_root, args.force)
    if not selected:
        return
    archive_path = _obtain_archive(args.output_root, args.archive)
    with zipfile.ZipFile(archive_path) as archive:
        for dataset_name, metadata in selected:
            _extract_domain(archive, dataset_name, metadata, args.output_root, args.force)


if __name__ == "__main__":
    main()
