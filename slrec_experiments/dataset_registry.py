#!/usr/bin/env python3
"""Pinned dataset provenance for the Hgformer experiment surface.

This module is intentionally data-only.  Download/conversion scripts and
experiment runners can import it without importing RecBole or PyTorch.  The
registry covers the six datasets reported in the Hgformer paper, the Amazon
Toy negative control, and the bundled MovieLens-100K smoke test.  It does not
claim that these are every dataset supported by upstream RecBole-GNN.

Run ``python slrec_experiments/dataset_registry.py`` for a compact source
listing, or pass ``--format json`` for machine-readable metadata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


PAPER_ROLE = "hgformer-paper"
NEGATIVE_CONTROL_ROLE = "negative-control"
SMOKE_ROLE = "smoke-only"
ROLES = (PAPER_ROLE, NEGATIVE_CONTROL_ROLE, SMOKE_ROLE)

# Dataset labels still present in archival YAMLs but not accepted by any of the
# maintained paper/negative-control/smoke runners.  They deliberately have no
# guessed URL or checksum.  See DATASETS.md before attempting to revive one.
LEGACY_UNPINNED_CONFIG_DATASETS = {
    "Alibaba-iFashion": "archival pre-paper configuration; no pinned local source",
    "Amazon_movie": "stale singular alias; maintained Amazon Movies uses Amazon_movies",
    "HGCFAmazonBook": "archival HGCF-format alias; not the paper Amazon_book source",
    "HGCFYELP": "archival HGCF-format alias; no pinned local source",
    "netflix": "archival mislabeled RecFormer_hgcf_cd configuration; not a paper dataset",
}

AMAZON_2014_HOMEPAGE = "https://jmcauley.ucsd.edu/data/amazon/index_2014.html"
AMAZON_2014_SOURCE_ROOT = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
)
DOUBAN_ORIGINAL_URL = (
    "https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip"
)
DOUBAN_PINNED_URL = (
    "https://web.archive.org/web/20240401023103id_/"
    "https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip"
)
RECBOLE_CDR_HOMEPAGE = "https://github.com/RUCAIBox/RecBole-CDR"
MOVIELENS_100K_HOMEPAGE = (
    "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt"
)
MOVIELENS_100K_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
)


@dataclass(frozen=True)
class FilteredStats:
    """Expected graph after the configured rating and iterative-core filter.

    ``token_users`` and ``token_items`` count source IDs.  RecBole's
    ``user_num`` and ``item_num`` include reserved ID zero and are therefore
    one larger.  ``paper_*`` preserves the printed Hgformer table separately
    when it differs from the verified graph.
    """

    rating_min: int
    iterative_k_core: int
    token_users: int
    token_items: int
    interactions: int
    paper_users: int | None = None
    paper_items: int | None = None
    paper_interactions: int | None = None
    note: str = ""

    @property
    def framework_users(self) -> int:
        return self.token_users + 1

    @property
    def framework_items(self) -> int:
        return self.token_items + 1


@dataclass(frozen=True)
class DatasetRecord:
    slug: str
    dataset: str
    display_name: str
    roles: tuple[str, ...]
    release: str
    homepage_url: str
    download_url: str
    original_url: str | None
    download_filename: str
    download_bytes: int
    download_sha256: str | None
    raw_filename: str
    raw_bytes: int
    raw_sha256: str | None
    raw_rows: int | None
    atomic_relative_path: str
    atomic_bytes: int
    atomic_sha256: str | None
    recformer_config: str
    protocol_overlays: tuple[str, ...]
    filtered: FilteredStats
    source_note: str = ""


def _amazon(
    *,
    slug: str,
    dataset: str,
    display_name: str,
    role: str,
    filename: str,
    raw_bytes: int,
    raw_rows: int,
    raw_sha256: str | None,
    atomic_sha256: str | None,
    config: str,
    overlays: tuple[str, ...],
    filtered: FilteredStats,
) -> DatasetRecord:
    return DatasetRecord(
        slug=slug,
        dataset=dataset,
        display_name=display_name,
        roles=(role,),
        release=(
            "McAuley Amazon Reviews 2014 ratings-only "
            "(reviews through July 2014; not Amazon Reviews 2018)"
        ),
        homepage_url=AMAZON_2014_HOMEPAGE,
        download_url=f"{AMAZON_2014_SOURCE_ROOT}/{filename}",
        original_url=None,
        download_filename=filename,
        download_bytes=raw_bytes,
        download_sha256=raw_sha256,
        raw_filename=filename,
        raw_bytes=raw_bytes,
        raw_sha256=raw_sha256,
        raw_rows=raw_rows,
        atomic_relative_path=f"{dataset}/{dataset}.inter",
        # Conversion replaces three commas with tabs (same byte width) and
        # prepends the canonical 57-byte RecBole header.
        atomic_bytes=raw_bytes + 57,
        atomic_sha256=atomic_sha256,
        recformer_config=config,
        protocol_overlays=overlays,
        filtered=filtered,
        source_note="Four columns without a header: user,item,rating,timestamp.",
    )


DATASETS: tuple[DatasetRecord, ...] = (
    _amazon(
        slug="amazon-cd",
        dataset="Amazon_cd",
        display_name="Amazon CD",
        role=PAPER_ROLE,
        filename="ratings_CDs_and_Vinyl.csv",
        raw_bytes=152_336_022,
        raw_rows=3_749_004,
        # Reconstructed byte-for-byte from the pinned atomic file by removing
        # its header and reversing tab-to-comma conversion; size and row count
        # match the official object.
        raw_sha256="74b503152bc8f92f389a1f841ec00361f903772e5ebb2b664ce678c023dbbceb",
        atomic_sha256="7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4",
        config="RecFormer_cd.yaml",
        overlays=(),
        filtered=FilteredStats(
            3,
            5,
            66_316,
            58_868,
            952_547,
            66_317,
            58_869,
            952_547,
        ),
    ),
    _amazon(
        slug="amazon-movies",
        dataset="Amazon_movies",
        display_name="Amazon Movies",
        role=PAPER_ROLE,
        filename="ratings_Movies_and_TV.csv",
        raw_bytes=187_517_953,
        raw_rows=4_607_047,
        raw_sha256=None,
        atomic_sha256="bfe4801cc0e3382191d9b7c62869dc6eda870068030371b956d81701db9fa403",
        config="RecFormer_movie.yaml",
        overlays=(),
        filtered=FilteredStats(
            3,
            10,
            26_968,
            18_563,
            762_957,
            26_969,
            18_564,
            762_957,
        ),
    ),
    _amazon(
        slug="amazon-book",
        dataset="Amazon_book",
        display_name="Amazon Book",
        role=PAPER_ROLE,
        filename="ratings_Books.csv",
        raw_bytes=916_259_348,
        raw_rows=22_507_155,
        raw_sha256=None,
        atomic_sha256=None,
        config="RecFormer_book.yaml",
        overlays=("PaperProtocol_amazon_book_8core.yaml",),
        filtered=FilteredStats(
            3,
            8,
            211_169,
            163_788,
            5_069_747,
            211_170,
            163_789,
            5_069_747,
            "The released RecFormer_book.yaml says 5-core; the paper pipeline "
            "must append PaperProtocol_amazon_book_8core.yaml.",
        ),
    ),
    DatasetRecord(
        slug="douban-book",
        dataset="DoubanBook",
        display_name="Douban Book",
        roles=(PAPER_ROLE,),
        release="RecBole-CDR v0.1.0 full Douban bundle",
        homepage_url=RECBOLE_CDR_HOMEPAGE,
        download_url=DOUBAN_PINNED_URL,
        original_url=DOUBAN_ORIGINAL_URL,
        download_filename="Douban.zip",
        download_bytes=35_496_374,
        download_sha256="f6883d0ac6745876b92eae22977d3694993c52a83d1f039a833c083ed812551a",
        raw_filename="Douban/DoubanBook/DoubanBook.inter",
        raw_bytes=30_034_862,
        raw_sha256="760b815632e68fdec8a975cd5ead72e5b8cb03aa8b12f7979824d87ffd629e9e",
        raw_rows=None,
        atomic_relative_path="DoubanBook/DoubanBook.inter",
        atomic_bytes=30_034_862,
        atomic_sha256="760b815632e68fdec8a975cd5ead72e5b8cb03aa8b12f7979824d87ffd629e9e",
        recformer_config="RecFormer_doubanbook.yaml",
        protocol_overlays=(),
        filtered=FilteredStats(
            3,
            5,
            18_085,
            33_067,
            809_248,
            18_086,
            33_068,
            809_248,
        ),
        source_note="Already a RecBole atomic interaction file inside the archive.",
    ),
    DatasetRecord(
        slug="douban-movie",
        dataset="DoubanMovie",
        display_name="Douban Movie",
        roles=(PAPER_ROLE,),
        release="RecBole-CDR v0.1.0 full Douban bundle",
        homepage_url=RECBOLE_CDR_HOMEPAGE,
        download_url=DOUBAN_PINNED_URL,
        original_url=DOUBAN_ORIGINAL_URL,
        download_filename="Douban.zip",
        download_bytes=35_496_374,
        download_sha256="f6883d0ac6745876b92eae22977d3694993c52a83d1f039a833c083ed812551a",
        raw_filename="Douban/DoubanMovie/DoubanMovie.inter",
        raw_bytes=77_249_371,
        raw_sha256="febc5c978f8e7f1765bfd97a6095246d2af20e182877d09e92515076fa36db24",
        raw_rows=None,
        atomic_relative_path="DoubanMovie/DoubanMovie.inter",
        atomic_bytes=77_249_371,
        atomic_sha256="febc5c978f8e7f1765bfd97a6095246d2af20e182877d09e92515076fa36db24",
        recformer_config="RecFormer_doubanmovie.yaml",
        protocol_overlays=(),
        filtered=FilteredStats(
            3,
            5,
            22_040,
            25_801,
            2_552_305,
            22_041,
            25_802,
            2_553_305,
            "The paper interaction count is +1,000; the pinned official file "
            "and RecBole-CDR references yield 2,552,305.",
        ),
        source_note="Already a RecBole atomic interaction file inside the archive.",
    ),
    DatasetRecord(
        slug="douban-music",
        dataset="DoubanMusic",
        display_name="Douban Music",
        roles=(PAPER_ROLE,),
        release="RecBole-CDR v0.1.0 full Douban bundle",
        homepage_url=RECBOLE_CDR_HOMEPAGE,
        download_url=DOUBAN_PINNED_URL,
        original_url=DOUBAN_ORIGINAL_URL,
        download_filename="Douban.zip",
        download_bytes=35_496_374,
        download_sha256="f6883d0ac6745876b92eae22977d3694993c52a83d1f039a833c083ed812551a",
        raw_filename="Douban/DoubanMusic/DoubanMusic.inter",
        raw_bytes=37_977_505,
        raw_sha256="f29cb8a2321cedbc84320eb1d97a00e73105f52e9def225b27a3a7d89b6a88ef",
        raw_rows=None,
        atomic_relative_path="DoubanMusic/DoubanMusic.inter",
        atomic_bytes=37_977_505,
        atomic_sha256="f29cb8a2321cedbc84320eb1d97a00e73105f52e9def225b27a3a7d89b6a88ef",
        recformer_config="RecFormer_doubanmusic.yaml",
        protocol_overlays=(),
        filtered=FilteredStats(
            3,
            5,
            15_995,
            39_748,
            1_116_984,
            15_996,
            39_749,
            1_116_984,
        ),
        source_note="Already a RecBole atomic interaction file inside the archive.",
    ),
    _amazon(
        slug="amazon-toy",
        dataset="Amazon_toy",
        display_name="Amazon Toy",
        role=NEGATIVE_CONTROL_ROLE,
        filename="ratings_Toys_and_Games.csv",
        raw_bytes=91_781_923,
        raw_rows=2_252_771,
        raw_sha256=None,
        atomic_sha256=None,
        config="RecFormer_toy.yaml",
        overlays=(),
        filtered=FilteredStats(3, 5, 15_528, 9_696, 133_837),
    ),
    DatasetRecord(
        slug="ml-100k",
        dataset="ml-100k",
        display_name="MovieLens 100K",
        roles=(SMOKE_ROLE,),
        release=(
            "MovieLens 100K stable benchmark (ratings collected "
            "1997-09-19 through 1998-04-22)"
        ),
        homepage_url=MOVIELENS_100K_HOMEPAGE,
        download_url=MOVIELENS_100K_URL,
        original_url=None,
        download_filename="ml-100k.zip",
        download_bytes=4_924_029,
        download_sha256="50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229",
        raw_filename="ml-100k/u.data",
        raw_bytes=1_979_173,
        raw_sha256="06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490",
        raw_rows=100_000,
        atomic_relative_path="ml-100k/ml-100k.inter",
        atomic_bytes=1_979_230,
        atomic_sha256="4edb74e2a81178c2ba9ff381495f754f996c4aea351b1272ca36b43da0935eff",
        recformer_config="RecFormer_smoke.yaml",
        protocol_overlays=(),
        filtered=FilteredStats(
            3,
            5,
            943,
            1_203,
            81_697,
            note="Smoke-config result; this is not a paper benchmark statistic.",
        ),
        source_note=(
            "The repository already vendors the exact converted interaction "
            "file under recbole/dataset_example."
        ),
    ),
)

DATASET_BY_SLUG = {record.slug: record for record in DATASETS}
DATASET_BY_NAME = {record.dataset.lower(): record for record in DATASETS}
PAPER_DATASET_SLUGS = tuple(
    record.slug for record in DATASETS if PAPER_ROLE in record.roles
)
NEGATIVE_CONTROL_SLUGS = tuple(
    record.slug for record in DATASETS if NEGATIVE_CONTROL_ROLE in record.roles
)
SMOKE_DATASET_SLUGS = tuple(
    record.slug for record in DATASETS if SMOKE_ROLE in record.roles
)


def records_for_role(role: str | None = None) -> tuple[DatasetRecord, ...]:
    if role is None:
        return DATASETS
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; choices={ROLES}")
    return tuple(record for record in DATASETS if role in record.roles)


def _json_records(records: Iterable[DatasetRecord]) -> str:
    return json.dumps([asdict(record) for record in records], indent=2) + "\n"


def _tsv_records(records: Iterable[DatasetRecord]) -> str:
    lines = ["slug\tdataset\trole\trelease\traw filename\traw bytes\tdownload URL"]
    for record in records:
        lines.append(
            "\t".join(
                (
                    record.slug,
                    record.dataset,
                    ",".join(record.roles),
                    record.release,
                    record.raw_filename,
                    str(record.raw_bytes),
                    record.download_url,
                )
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, help="Show only one experiment role.")
    parser.add_argument(
        "--legacy-config-labels",
        action="store_true",
        help="List archival YAML dataset labels that lack a pinned runnable source.",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "json"),
        default="tsv",
        help="Output format (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    if args.legacy_config_labels:
        if args.format == "json":
            payload = [
                {"dataset": name, "status": "unsupported-unpinned", "note": note}
                for name, note in LEGACY_UNPINNED_CONFIG_DATASETS.items()
            ]
            print(json.dumps(payload, indent=2))
        else:
            print("dataset\tstatus\tnote")
            for name, note in LEGACY_UNPINNED_CONFIG_DATASETS.items():
                print(f"{name}\tunsupported-unpinned\t{note}")
        return
    selected = records_for_role(args.role)
    print(_json_records(selected) if args.format == "json" else _tsv_records(selected), end="")


if __name__ == "__main__":
    main()
