#!/usr/bin/env python3
"""Validate or prepare the MovieLens-100K smoke-test interaction file.

The repository already vendors the exact converted ``ml-100k.inter`` used by
``RecFormer_smoke.yaml``.  On a stripped deployment this script can rebuild it
from GroupLens' stable ``ml-100k.zip`` archive.  It does not apply the smoke
configuration's rating>=3 and iterative 5-core filters; RecBole applies those
at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import urllib.error
import urllib.request
import zipfile
from typing import Optional

try:
    from slrec_experiments.dataset_registry import DATASET_BY_SLUG
except ModuleNotFoundError:  # Direct ``python slrec_experiments/<file>.py``.
    from dataset_registry import DATASET_BY_SLUG  # type: ignore


SOURCE = DATASET_BY_SLUG["ml-100k"]
ARCHIVE_URL = SOURCE.download_url
ARCHIVE_BYTES = SOURCE.download_bytes
ARCHIVE_SHA256 = str(SOURCE.download_sha256)
RAW_MEMBER = SOURCE.raw_filename
RAW_BYTES = SOURCE.raw_bytes
RAW_SHA256 = str(SOURCE.raw_sha256)
RAW_ROWS = int(SOURCE.raw_rows or 0)
ATOMIC_BYTES = SOURCE.atomic_bytes
ATOMIC_SHA256 = str(SOURCE.atomic_sha256)
ATOMIC_HEADER = b"user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


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
    destination = cache_dir / SOURCE.download_filename
    if destination.exists():
        _validate(destination, ARCHIVE_BYTES, ARCHIVE_SHA256)
        return destination

    partial = destination.with_suffix(".zip.part")
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"Accept-Encoding": "identity", "User-Agent": "slrec-experiment/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(
            "wb"
        ) as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != ARCHIVE_BYTES:
                raise ValueError(
                    f"Unexpected Content-Length for {ARCHIVE_URL}: {content_length} "
                    f"(expected {ARCHIVE_BYTES})"
                )
            shutil.copyfileobj(response, output, length=1024 * 1024)
        _validate(partial, ARCHIVE_BYTES, ARCHIVE_SHA256)
        partial.replace(destination)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not fetch the official GroupLens archive. Download "
            f"{ARCHIVE_URL} in a browser and retry with --archive PATH."
        ) from exc
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def prepare(
    output_root: pathlib.Path,
    archive: Optional[pathlib.Path] = None,
    force: bool = False,
) -> pathlib.Path:
    destination_dir = output_root / SOURCE.dataset
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{SOURCE.dataset}.inter"
    partial = destination.with_suffix(".inter.part")

    if destination.exists() and not force:
        _validate(destination, ATOMIC_BYTES, ATOMIC_SHA256)
        print(f"Validated existing bundled smoke file: {destination}")
        return destination

    archive_path = _obtain_archive(output_root, archive)
    raw_digest = hashlib.sha256()
    raw_newlines = 0
    raw_final_byte = b""
    try:
        with zipfile.ZipFile(archive_path) as source_archive:
            member = source_archive.getinfo(RAW_MEMBER)
            if member.file_size != RAW_BYTES:
                raise ValueError(
                    f"Unexpected uncompressed size for {RAW_MEMBER}: {member.file_size} "
                    f"(expected {RAW_BYTES})"
                )
            with source_archive.open(member, "r") as source, partial.open("wb") as output:
                output.write(ATOMIC_HEADER)
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    raw_digest.update(block)
                    raw_newlines += block.count(b"\n")
                    raw_final_byte = block[-1:]
                    output.write(block)

        raw_lines = raw_newlines + (
            1 if raw_final_byte and raw_final_byte != b"\n" else 0
        )
        if raw_lines != RAW_ROWS:
            raise ValueError(
                f"Unexpected row count for {RAW_MEMBER}: {raw_lines} "
                f"(expected {RAW_ROWS})"
            )
        if raw_digest.hexdigest() != RAW_SHA256:
            raise ValueError(
                f"Unexpected SHA256 for {RAW_MEMBER}: {raw_digest.hexdigest()} "
                f"(expected {RAW_SHA256})"
            )
        _validate(partial, ATOMIC_BYTES, ATOMIC_SHA256)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print(f"Prepared {destination}: {RAW_ROWS:,} raw ratings")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Official README: {SOURCE.homepage_url}\n"
            f"Official archive: {ARCHIVE_URL}\n"
            f"Archive: bytes={ARCHIVE_BYTES}, sha256={ARCHIVE_SHA256}\n"
            f"Raw member: {RAW_MEMBER}, bytes={RAW_BYTES}, sha256={RAW_SHA256}\n"
            "This dataset is only an integration smoke test in Hgformer; it is "
            "not one of the six paper datasets."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("recbole/dataset_example"),
        help=(
            "Directory containing the ml-100k subdirectory "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        help="Use an already-downloaded official ml-100k.zip archive.",
    )
    parser.add_argument(
        "--list-source",
        action="store_true",
        help="Print the pinned archive/member/atomic metadata and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing converted interaction file.",
    )
    args = parser.parse_args()

    if args.list_source:
        print(
            f"archive={SOURCE.download_filename}\tbytes={ARCHIVE_BYTES}\t"
            f"sha256={ARCHIVE_SHA256}\turl={ARCHIVE_URL}"
        )
        print(
            f"member={RAW_MEMBER}\trows={RAW_ROWS}\tbytes={RAW_BYTES}\t"
            f"sha256={RAW_SHA256}"
        )
        print(
            f"atomic={SOURCE.atomic_relative_path}\tbytes={ATOMIC_BYTES}\t"
            f"sha256={ATOMIC_SHA256}"
        )
        return

    prepare(args.output_root, args.archive, args.force)


if __name__ == "__main__":
    main()
