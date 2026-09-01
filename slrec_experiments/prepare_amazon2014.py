"""Prepare the Amazon Reviews 2014 ratings used by Hgformer.

The public source is McAuley's ratings-only collection.  Rows are
``user,item,rating,timestamp`` without a header.  This script only converts
them to RecBole's atomic TSV format; rating and iterative k-core filtering are
left to the experiment configuration so every model receives the same graph.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import urllib.request
from typing import Iterable

try:
    from slrec_experiments.dataset_registry import (
        AMAZON_2014_HOMEPAGE,
        AMAZON_2014_SOURCE_ROOT,
        DATASETS,
    )
except ModuleNotFoundError:  # Direct ``python slrec_experiments/<file>.py``.
    from dataset_registry import (  # type: ignore
        AMAZON_2014_HOMEPAGE,
        AMAZON_2014_SOURCE_ROOT,
        DATASETS,
    )


SOURCE_ROOT = AMAZON_2014_SOURCE_ROOT
DOMAINS = {
    record.dataset: {
        "filename": record.raw_filename,
        "url": record.download_url,
        "raw_rows": record.raw_rows,
        "raw_bytes": record.raw_bytes,
        "raw_sha256": record.raw_sha256,
        "atomic_bytes": record.atomic_bytes,
        "atomic_sha256": record.atomic_sha256,
        "role": record.roles[0],
        "release": record.release,
    }
    for record in DATASETS
    if record.download_url.startswith(AMAZON_2014_SOURCE_ROOT)
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_atomic(path: pathlib.Path, metadata: dict[str, object]) -> None:
    expected_bytes = int(metadata["atomic_bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected size for {path}: {path.stat().st_size} "
            f"(expected {expected_bytes})"
        )
    expected_sha256 = metadata.get("atomic_sha256")
    if expected_sha256 is not None:
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Unexpected SHA256 for {path}: {actual_sha256} "
                f"(expected {expected_sha256})"
            )


def _source_lines(response: Iterable[bytes], digest: "hashlib._Hash") -> Iterable[str]:
    for raw_line in response:
        digest.update(raw_line)
        yield raw_line.decode("utf-8")


def prepare_domain(
    dataset_name: str,
    metadata: dict[str, object],
    output_root: pathlib.Path,
    force: bool,
) -> pathlib.Path:
    destination_dir = output_root / dataset_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{dataset_name}.inter"
    partial = destination.with_suffix(".inter.part")

    if destination.exists() and not force:
        _validate_atomic(destination, metadata)
        print(f"Validated existing file (use --force to replace): {destination}")
        return destination

    filename = str(metadata["filename"])
    url = str(metadata["url"])
    request = urllib.request.Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "slrec-experiment/1.0"},
    )
    row_count = 0
    raw_digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(
            "w", encoding="utf-8", newline=""
        ) as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != int(metadata["raw_bytes"]):
                raise ValueError(
                    f"Unexpected Content-Length for {url}: {content_length} "
                    f"(expected {metadata['raw_bytes']})"
                )

            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
            )
            reader = csv.reader(_source_lines(response, raw_digest))
            for row_count, row in enumerate(reader, start=1):
                if len(row) != 4:
                    raise ValueError(f"Unexpected source row {row_count} in {url}: {row!r}")
                writer.writerow(row)

        if row_count != int(metadata["raw_rows"]):
            raise ValueError(
                f"Unexpected row count for {url}: {row_count} "
                f"(expected {metadata['raw_rows']})"
            )
        expected_raw_sha256 = metadata.get("raw_sha256")
        if expected_raw_sha256 is not None and raw_digest.hexdigest() != expected_raw_sha256:
            raise ValueError(
                f"Unexpected raw SHA256 for {url}: {raw_digest.hexdigest()} "
                f"(expected {expected_raw_sha256})"
            )
        _validate_atomic(partial, metadata)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print(f"Prepared {destination}: {row_count:,} interactions")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Official release page: {AMAZON_2014_HOMEPAGE}\n"
            f"Direct-file root: {SOURCE_ROOT}\n"
            "These are the 2014 ratings-only CSVs, not Amazon Reviews 2018.\n"
            "Use --list-sources to print exact filenames, sizes, roles, and URLs."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("dataset"),
        help="Directory that will contain RecBole dataset subdirectories.",
    )
    parser.add_argument(
        "--domain",
        choices=["all", *DOMAINS],
        default="all",
        help="Prepare one Amazon domain or all four registered domains.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print the pinned 2014 source registry and exit without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing atomic interaction file.",
    )
    args = parser.parse_args()

    if args.list_sources:
        for dataset_name, metadata in DOMAINS.items():
            digest = metadata["raw_sha256"] or "not-pinned"
            print(
                f"{dataset_name}\trole={metadata['role']}\t"
                f"file={metadata['filename']}\trows={metadata['raw_rows']}\t"
                f"bytes={metadata['raw_bytes']}\tsha256={digest}\turl={metadata['url']}"
            )
        return

    selected = DOMAINS.items() if args.domain == "all" else [(args.domain, DOMAINS[args.domain])]
    for dataset_name, metadata in selected:
        prepare_domain(dataset_name, metadata, args.output_root, args.force)


if __name__ == "__main__":
    main()
