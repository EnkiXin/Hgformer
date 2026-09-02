"""Dataset-source, role, and protocol registry contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from slrec_experiments import dataset_registry
from slrec_experiments.dataset_registry import (
    DATASETS,
    DATASET_BY_SLUG,
    LEGACY_UNPINNED_CONFIG_DATASETS,
    NEGATIVE_CONTROL_SLUGS,
    PAPER_DATASET_SLUGS,
    PAPER_ROLE,
    SEPARATELY_PINNED_CONFIG_DATASETS,
    SMOKE_DATASET_SLUGS,
)
from slrec_experiments.prepare_amazon2014 import DOMAINS as AMAZON_DOMAINS
from slrec_experiments.prepare_douban import (
    ARCHIVE_BYTES as DOUBAN_ARCHIVE_BYTES,
    ARCHIVE_SHA256 as DOUBAN_ARCHIVE_SHA256,
    DOMAINS as DOUBAN_DOMAINS,
    _pending_domains as pending_douban_domains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetRoleTest(unittest.TestCase):
    def test_roles_distinguish_paper_negative_control_and_smoke(self):
        self.assertEqual(
            PAPER_DATASET_SLUGS,
            (
                "amazon-cd",
                "amazon-movies",
                "amazon-book",
                "douban-book",
                "douban-movie",
                "douban-music",
            ),
        )
        self.assertEqual(NEGATIVE_CONTROL_SLUGS, ("amazon-toy",))
        self.assertEqual(SMOKE_DATASET_SLUGS, ("ml-100k",))
        self.assertEqual(
            set(PAPER_DATASET_SLUGS)
            | set(NEGATIVE_CONTROL_SLUGS)
            | set(SMOKE_DATASET_SLUGS),
            {record.slug for record in DATASETS},
        )

    def test_every_protocol_file_and_overlay_exists(self):
        for record in DATASETS:
            config = REPO_ROOT / "baseline_config_fixed" / record.recformer_config
            self.assertTrue(config.is_file(), config)
            for overlay_name in record.protocol_overlays:
                overlay = REPO_ROOT / "baseline_config_fixed" / overlay_name
                self.assertTrue(overlay.is_file(), overlay)

    def test_every_fixed_config_dataset_label_is_classified(self):
        discovered: set[str] = set()
        config_roots = (
            REPO_ROOT / "baseline_config_fixed",
            REPO_ROOT / "slrec_experiments" / "configs",
        )
        for config_root in config_roots:
            for path in config_root.glob("*.yaml"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("dataset:"):
                        discovered.add(line.split(":", 1)[1].strip().strip("'\""))
                        break
        classified = {record.dataset for record in DATASETS} | set(
            LEGACY_UNPINNED_CONFIG_DATASETS
        ) | set(SEPARATELY_PINNED_CONFIG_DATASETS)
        self.assertEqual(discovered, classified)


class SourcePinTest(unittest.TestCase):
    def test_amazon_is_2014_ratings_only_not_2018(self):
        amazon = [record for record in DATASETS if record.dataset.startswith("Amazon_")]
        self.assertEqual(len(amazon), 4)
        for record in amazon:
            self.assertIn("2014 ratings-only", record.release)
            self.assertIn("productGraph/categoryFiles/ratings_", record.download_url)
            self.assertNotIn("amazon_v2", record.download_url)
            self.assertEqual(record.atomic_bytes, record.raw_bytes + 57)
            self.assertIsNotNone(record.raw_rows)
        self.assertEqual(
            DATASET_BY_SLUG["amazon-cd"].raw_sha256,
            "74b503152bc8f92f389a1f841ec00361f903772e5ebb2b664ce678c023dbbceb",
        )

    def test_prepare_amazon_registry_is_not_a_second_metadata_source(self):
        self.assertEqual(
            set(AMAZON_DOMAINS),
            {record.dataset for record in DATASETS if record.dataset.startswith("Amazon_")},
        )
        for record in DATASETS:
            if record.dataset not in AMAZON_DOMAINS:
                continue
            prepared = AMAZON_DOMAINS[record.dataset]
            self.assertEqual(prepared["url"], record.download_url)
            self.assertEqual(prepared["raw_bytes"], record.raw_bytes)
            self.assertEqual(prepared["raw_sha256"], record.raw_sha256)

    def test_douban_is_the_full_pinned_recbole_cdr_bundle(self):
        douban = [record for record in DATASETS if record.dataset.startswith("Douban")]
        self.assertEqual(len(douban), 3)
        self.assertEqual(DOUBAN_ARCHIVE_BYTES, 35_496_374)
        self.assertEqual(
            DOUBAN_ARCHIVE_SHA256,
            "f6883d0ac6745876b92eae22977d3694993c52a83d1f039a833c083ed812551a",
        )
        for record in douban:
            self.assertIn("RecBole-CDR", record.release)
            self.assertEqual(record.download_bytes, DOUBAN_ARCHIVE_BYTES)
            self.assertEqual(record.download_sha256, DOUBAN_ARCHIVE_SHA256)
            self.assertEqual(record.atomic_bytes, record.raw_bytes)
            self.assertEqual(record.atomic_sha256, record.raw_sha256)
            self.assertGreater(record.atomic_bytes, 30_000_000)
            self.assertEqual(DOUBAN_DOMAINS[record.dataset]["member"], record.raw_filename)
            self.assertEqual(DOUBAN_DOMAINS[record.dataset]["sha256"], record.raw_sha256)

    def test_small_douban_is_rejected_before_any_archive_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "DoubanBook" / "DoubanBook.inter"
            destination.parent.mkdir()
            destination.write_text(
                "user_id:token\titem_id:token\trating:float\n1\t1\t5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not the full RecBole-CDR"):
                pending_douban_domains(
                    [("DoubanBook", DOUBAN_DOMAINS["DoubanBook"])], root, False
                )
            self.assertFalse((root / ".source").exists())

    def test_hash_fields_are_sha256_or_explicitly_unpinned(self):
        for record in DATASETS:
            for value in (
                record.download_sha256,
                record.raw_sha256,
                record.atomic_sha256,
            ):
                if value is not None:
                    self.assertEqual(len(value), 64)
                    int(value, 16)


class FilterProtocolTest(unittest.TestCase):
    def test_paper_records_all_use_rating_three_or_higher(self):
        for slug in PAPER_DATASET_SLUGS:
            self.assertEqual(DATASET_BY_SLUG[slug].filtered.rating_min, 3)

    def test_amazon_book_uses_exact_eight_core_paper_correction(self):
        book = DATASET_BY_SLUG["amazon-book"]
        self.assertIn(PAPER_ROLE, book.roles)
        self.assertEqual(book.protocol_overlays, ("PaperProtocol_amazon_book_8core.yaml",))
        self.assertEqual(book.filtered.iterative_k_core, 8)
        self.assertEqual(
            (
                book.filtered.token_users,
                book.filtered.token_items,
                book.filtered.interactions,
            ),
            (211_169, 163_788, 5_069_747),
        )
        self.assertEqual(
            (book.filtered.framework_users, book.filtered.framework_items),
            (211_170, 163_789),
        )

    def test_douban_movie_preserves_actual_and_paper_typo_separately(self):
        movie = DATASET_BY_SLUG["douban-movie"].filtered
        self.assertEqual(movie.interactions, 2_552_305)
        self.assertEqual(movie.paper_interactions, 2_553_305)

    def test_smoke_filter_counts_match_checked_run(self):
        smoke = DATASET_BY_SLUG["ml-100k"].filtered
        self.assertEqual(
            (
                smoke.framework_users,
                smoke.framework_items,
                smoke.interactions,
            ),
            (944, 1_204, 81_697),
        )


class MovieLensAndDocumentationTest(unittest.TestCase):
    def test_vendored_movielens_atomic_file_matches_registry(self):
        record = DATASET_BY_SLUG["ml-100k"]
        path = REPO_ROOT / "recbole" / "dataset_example" / record.atomic_relative_path
        self.assertEqual(path.stat().st_size, record.atomic_bytes)
        self.assertEqual(_sha256(path), record.atomic_sha256)

    def test_json_cli_contains_machine_readable_source_urls(self):
        output = StringIO()
        with redirect_stdout(output):
            dataset_registry.main(["--role", "smoke-only", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual([entry["slug"] for entry in payload], ["ml-100k"])
        self.assertEqual(payload[0]["download_url"], DATASET_BY_SLUG["ml-100k"].download_url)

    def test_unified_document_contains_every_registered_download_url(self):
        document = (REPO_ROOT / "DATASETS.md").read_text(encoding="utf-8")
        for record in DATASETS:
            self.assertIn(record.download_url, document)
            if record.original_url is not None:
                self.assertIn(record.original_url, document)
        self.assertIn("not one of the paper six", document)
        self.assertIn("not a paper benchmark", document)


if __name__ == "__main__":
    unittest.main()
