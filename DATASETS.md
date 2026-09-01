# Dataset provenance and preparation

This is the source of truth for the datasets used by this repository's
Hgformer experiments.  The same metadata is available to scripts from
`slrec_experiments/dataset_registry.py`:

```bash
python slrec_experiments/dataset_registry.py
python slrec_experiments/dataset_registry.py --format json
python slrec_experiments/dataset_registry.py --legacy-config-labels
```

The experiment roles are deliberately separate:

| Role | Datasets | Meaning |
| --- | --- | --- |
| Hgformer paper (six) | Amazon CD, Amazon Movies, Amazon Book, Douban Book, Douban Movie, Douban Music | The six datasets reported in the ICML 2025 paper |
| Negative control | Amazon Toy | Historical control on which Hgformer underperformed LightGCN; not one of the paper six |
| Smoke only | MovieLens 100K | Fast integration check; its reduced smoke configuration is not a benchmark |

Upstream RecBole-GNN can load other datasets, but they have no pinned source
and Hgformer protocol in this repository and are therefore outside this
registry.

### Archival YAML labels that are not maintained datasets

An inventory of `baseline_config_fixed/*.yaml` also finds the labels
`Alibaba-iFashion`, `HGCFAmazonBook`, `HGCFYELP`, `netflix`, and the singular
`Amazon_movie`.  They occur only in archival/pre-paper or stale alias configs;
none is accepted by the maintained paper, negative-control, or smoke runners,
and this checkout contains no source artifact with a verified provenance pin.
No download URL is guessed for them:

| Label | Why it is not in the runnable registry |
| --- | --- |
| `Alibaba-iFashion` | Archival pre-paper configs only; no pinned source in this project |
| `Amazon_movie` | Stale singular alias; the maintained dataset is `Amazon_movies` from `ratings_Movies_and_TV.csv` |
| `HGCFAmazonBook` | HGCF-format legacy alias, not the paper `Amazon_book` source/protocol |
| `HGCFYELP` | HGCF-format legacy alias without a pinned source artifact |
| `netflix` | Appears in the misleadingly named `RecFormer_hgcf_cd.yaml`; it is not Amazon CD and is not a paper dataset |

Treating a config filename as provenance would risk silently benchmarking a
different graph.  Reviving one of these labels requires a separate source,
license, checksum, conversion, and post-filter audit before it can be added to
`dataset_registry.py`.

## Amazon Reviews 2014 ratings-only

Use Julian McAuley's **2014 ratings-only** release, whose reviews end in July
2014.  Do not use the differently shaped Amazon Reviews 2018 release.

- Official release page: [Amazon product data (2014)](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
- Raw schema: headerless `user,item,rating,timestamp` CSV
- Converter: `slrec_experiments/prepare_amazon2014.py`

| Dataset / role | Raw file and direct download | Raw rows | Raw bytes | Raw SHA256 | Prepared atomic bytes / SHA256 |
| --- | --- | ---: | ---: | --- | --- |
| Amazon CD / paper | [`ratings_CDs_and_Vinyl.csv`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv) | 3,749,004 | 152,336,022 | `74b503152bc8f92f389a1f841ec00361f903772e5ebb2b664ce678c023dbbceb` | 152,336,079 / `7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4` |
| Amazon Movies / paper | [`ratings_Movies_and_TV.csv`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv) | 4,607,047 | 187,517,953 | — | 187,518,010 / `bfe4801cc0e3382191d9b7c62869dc6eda870068030371b956d81701db9fa403` |
| Amazon Book / paper | [`ratings_Books.csv`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv) | 22,507,155 | 916,259,348 | — | 916,259,405 / — |
| Amazon Toy / negative control | [`ratings_Toys_and_Games.csv`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv) | 2,252,771 | 91,781,923 | — | 91,781,980 / — |

An em dash means that an independently verified SHA256 is not yet pinned; it
does not mean that an empty checksum is accepted.  The converter always checks
the exact raw byte and row counts.  It also checks the raw or converted SHA256
where the registry has one.  The 57-byte difference is the canonical RecBole
header; commas are replaced by equal-width tabs.  The CD raw digest was
reconstructed reversibly from its pinned atomic file (remove the header and
change tabs back to commas), and the resulting byte/row counts match the
official object; the Stanford host does not publish its own SHA256 manifest.

Prepare only the domains required by a machine.  `--domain all` includes the
916 MB Book file and downloads about 1.35 GB in total:

```bash
python slrec_experiments/prepare_amazon2014.py --list-sources
python slrec_experiments/prepare_amazon2014.py --domain Amazon_cd
python slrec_experiments/prepare_amazon2014.py --domain Amazon_movies
python slrec_experiments/prepare_amazon2014.py --domain Amazon_book
python slrec_experiments/prepare_amazon2014.py --domain Amazon_toy
```

## Full RecBole-CDR Douban release

Use the full Douban bundle published by
[RecBole-CDR v0.1.0](https://github.com/RUCAIBox/RecBole-CDR).  The
[original S3 object](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip)
currently returns HTTP 403.  The preparation script therefore uses a
[pinned 2024 Wayback capture](https://web.archive.org/web/20240401023103id_/https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip).

`Douban.zip` is 35,496,374 bytes with SHA256
`f6883d0ac6745876b92eae22977d3694993c52a83d1f039a833c083ed812551a`.
The Wayback object is byte-identical to the former RecBole-CDR object.

| Dataset | Member in `Douban.zip` | Uncompressed bytes | Member/atomic SHA256 |
| --- | --- | ---: | --- |
| Douban Book | `Douban/DoubanBook/DoubanBook.inter` | 30,034,862 | `760b815632e68fdec8a975cd5ead72e5b8cb03aa8b12f7979824d87ffd629e9e` |
| Douban Movie | `Douban/DoubanMovie/DoubanMovie.inter` | 77,249,371 | `febc5c978f8e7f1765bfd97a6095246d2af20e182877d09e92515076fa36db24` |
| Douban Music | `Douban/DoubanMusic/DoubanMusic.inter` | 37,977,505 | `f29cb8a2321cedbc84320eb1d97a00e73105f52e9def225b27a3a7d89b6a88ef` |

These members are already RecBole atomic files; preparation extracts them
without changing user tokens.  Do **not** substitute the much smaller CoPD
Douban files, or the unrelated RecBole-GNN Social-Datasets `douban-book`
release.  The script rejects such files by exact size and SHA256 before any
training starts.

```bash
python slrec_experiments/prepare_douban.py --list-sources
python slrec_experiments/prepare_douban.py --domain all

# Air-gapped/reused download:
python slrec_experiments/prepare_douban.py \
  --archive /path/to/Douban.zip --domain all
```

## MovieLens 100K smoke data

MovieLens 100K is a stable GroupLens dataset of 100,000 ratings collected
from 943 users between 19 September 1997 and 22 April 1998.  It is used here
only for CPU/integration smoke checks.

- Official README: [MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- Official archive: [`ml-100k.zip`](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
- Archive: 4,924,029 bytes; SHA256 `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`; official MD5 `0e33842e24a9c977be4e0107933c0723`
- Raw member `ml-100k/u.data`: 1,979,173 bytes, 100,000 rows; SHA256 `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`
- Atomic `ml-100k/ml-100k.inter`: 1,979,230 bytes; SHA256 `4edb74e2a81178c2ba9ff381495f754f996c4aea351b1272ca36b43da0935eff`

The exact atomic file is already versioned at
`recbole/dataset_example/ml-100k/ml-100k.inter`.  Validate it, or rebuild a
missing copy from the official archive:

```bash
python slrec_experiments/prepare_movielens100k.py --list-source
python slrec_experiments/prepare_movielens100k.py
python slrec_experiments/prepare_movielens100k.py \
  --archive /path/to/ml-100k.zip
```

Review the usage conditions in the official README before redistributing the
MovieLens archive or derived data.

## Filtering protocol and expected graph statistics

Conversion/extraction does not perform experiment filtering.  RecBole first
keeps ratings `>= 3`, then iteratively removes users and items below the listed
core until stable.  `Token U/I` counts real source IDs; `RecBole U/I` includes
reserved ID zero and is exactly one larger.  Edge counts do not include a
reserved row.

| Dataset | Role | Rating / iterative core | Token U / I / interactions | RecBole U / I | Hgformer paper table U / I / interactions |
| --- | --- | --- | ---: | ---: | ---: |
| Amazon CD | paper | `>=3`, 5-core | 66,316 / 58,868 / 952,547 | 66,317 / 58,869 | 66,317 / 58,869 / 952,547 |
| Amazon Movies | paper | `>=3`, 10-core | 26,968 / 18,563 / 762,957 | 26,969 / 18,564 | 26,969 / 18,564 / 762,957 |
| Amazon Book | paper | `>=3`, **8-core** | 211,169 / 163,788 / 5,069,747 | 211,170 / 163,789 | 211,170 / 163,789 / 5,069,747 |
| Douban Book | paper | `>=3`, 5-core | 18,085 / 33,067 / 809,248 | 18,086 / 33,068 | 18,086 / 33,068 / 809,248 |
| Douban Movie | paper | `>=3`, 5-core | 22,040 / 25,801 / **2,552,305** | 22,041 / 25,802 | 22,041 / 25,802 / **2,553,305** |
| Douban Music | paper | `>=3`, 5-core | 15,995 / 39,748 / 1,116,984 | 15,996 / 39,749 | 15,996 / 39,749 / 1,116,984 |
| Amazon Toy | negative control | `>=3`, 5-core | 15,528 / 9,696 / 133,837 | 15,529 / 9,697 | not a paper-six dataset |
| MovieLens 100K | smoke only | `>=3`, 5-core | 943 / 1,203 / 81,697 | 944 / 1,204 | not a paper benchmark |

Two protocol details are easy to miss:

1. The released `RecFormer_book.yaml` says 5-core, but the paper cardinalities
   require the checked-in `PaperProtocol_amazon_book_8core.yaml` overlay.  The
   six-dataset paper pipeline applies and audits that overlay for every model.
2. The verified full Douban Movie file yields 2,552,305 filtered interactions.
   The paper prints 2,553,305; the +1,000 discrepancy is retained above as a
   paper-table typo, not reproduced by modifying the data.

For paper runs, the remaining shared protocol is seed 2024, random user-wise
8:1:1 splitting, full-ranking evaluation, Recall/NDCG at 5/10/20/50, and
validation selection by Recall@10.  The one-epoch MovieLens configuration is
only a software check and must never be reported beside paper results.

Primary references:

- [Hgformer, ICML 2025 paper and software link](https://proceedings.mlr.press/v267/yang25o.html)
- [McAuley Amazon Reviews 2014 release](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
- [RecBole-CDR official repository and dataset links](https://github.com/RUCAIBox/RecBole-CDR)
- [GroupLens MovieLens dataset index](https://files.grouplens.org/datasets/movielens/)
