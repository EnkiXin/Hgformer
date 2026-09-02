# AGCF dataset sources and exact preprocessing

This registry pins the data contract used to reproduce *Learning on Adaptive
Manifolds for Graph Collaborative Filtering* (WWW 2026).  It is separate from
[`DATASETS.md`](DATASETS.md): the older HGformer/RecFormer experiments use
different Amazon releases, while the AGCF Table 2 counts match Amazon Reviews
2023 benchmark files.

## Status vocabulary

- **VERIFIED EXACT** means that applying the stated deterministic transform to
  the pinned source reproduces all three Table 2 counts.  It does not claim
  that the authors released code, nor that model metrics have been reproduced.
- **PENDING** means that a source or preprocessing choice is still unresolved.
  A pending dataset must not be used for a paper-number comparison.

Counts below are `users / items / interactions`.  Interaction counts exclude
the RecBole atomic-file header.

| Dataset | Status | AGCF Table 2 target | Exact-count result |
| --- | --- | ---: | ---: |
| MovieLens | **VERIFIED EXACT** | 6,039 / 3,308 / 835,789 | 6,038 / 3,307 / 835,789 real IDs; 6,039 / 3,308 at RecBole runtime after `[PAD]` |
| Amazon-CD | **VERIFIED EXACT** | 113,303 / 82,910 / 1,397,717 | 113,303 / 82,910 / 1,397,717 |
| Amazon-Book | **VERIFIED EXACT** | 118,039 / 98,793 / 2,836,041 | 118,039 / 98,793 / 2,836,041 |
| Gowalla | **VERIFIED EXACT** | 64,115 / 164,532 / 2,018,421 | 64,115 / 164,532 / 2,018,421 |
| Yelp | **PENDING** | 202,012 / 86,880 / 3,022,685 | Frozen release and transform not yet recovered |

## Shared filtering definition

“Joint iterative `k`-core” has one meaning throughout this document:

1. Count current interactions for every user and item.
2. Remove every interaction incident to a user or item with degree below `k`.
3. Recompute both sides on the reduced graph and repeat until no row is
   removed.

A single user pass followed by a single item pass is not equivalent and is not
accepted.  Unless a dataset section says otherwise, an interaction is a
user-item pair with its rating or timestamp retained only as metadata.

## MovieLens — VERIFIED EXACT

- Official release page: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- Direct archive: [`ml-1m.zip`](https://files.grouplens.org/datasets/movielens/ml-1m.zip)
- Archive MD5: `c4d9eecfca2ab87c1945afe126590906`
- Archive SHA-256: `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`
- Repository atomic conversion:
  `dataset/AGCF_MovieLens/AGCF_MovieLens.inter`
- Atomic conversion SHA-256:
  `e943abb91013a54c385828fdf5ab4ce49e957ca3a772adb30cde2a7d5539b389`

Preparation contract:

1. Read every record from `ratings.dat` without deduplicating it.
2. Convert `UserID`, `MovieID`, `Rating`, and `Timestamp` losslessly to a
   RecBole `.inter` file.
3. Retain `rating >= 3`.
4. Apply a joint iterative 5-core to users and items.

The resulting graph contains 6,038 real users, 3,307 real items, and 835,789
interactions.  RecBole reserves token index zero as `[PAD]`, so its runtime
`user_num` and `item_num` are 6,039 and 3,308.  Those padding-inclusive values
are the numbers printed in the paper.  Do not add a synthetic interaction for
the padding token.

The checked-in protocol applies steps 3–4 at load time in
`baseline_config_fixed/AGCF_movielens_protocol.yaml`; the atomic source remains
a lossless conversion of all 1,000,209 MovieLens-1M ratings.

## Amazon-CD — VERIFIED EXACT

- Source family and processing notes: [Amazon Reviews 2023 precomputed 5-core](https://amazon-reviews-2023.github.io/data_processing/5core.html)
- Direct ratings-only archive: [`CDs_and_Vinyl.csv.gz`](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/CDs_and_Vinyl.csv.gz)
- Archive SHA-256: `53811902ffb01eb22da31a35e0775f3b8351baebe7e1ab38b85b6c4aee689c20`

Preparation contract:

1. Start from the precomputed 5-core **ratings-only** archive above, not the
   similarly named Amazon Reviews 2014 or 2018 data.
2. Retain `rating >= 3`.
3. Apply a fresh joint iterative 5-core after rating filtering.

This produces exactly 113,303 users, 82,910 items, and 1,397,717
interactions.  A RecBole loader will report 113,304 and 82,911 for
`user_num`/`item_num` because of `[PAD]`; the paper's Table 2 values here are
the real-ID counts.

## Amazon-Book — VERIFIED EXACT

- Source family and processing notes: [Amazon Reviews 2023 precomputed 5-core](https://amazon-reviews-2023.github.io/data_processing/5core.html)
- Direct ratings-only archive: [`Books.csv.gz`](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz)
- Archive SHA-256: `28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615`

Preparation contract:

1. Start from the precomputed 5-core **ratings-only** archive above.
2. Retain `rating >= 3`.
3. Apply a joint iterative **10-core** after rating filtering.  The second
   threshold is 10 even though the downloaded benchmark archive is named
   `5core`.

This produces exactly 118,039 users, 98,793 items, and 2,836,041
interactions.  RecBole's padding-inclusive runtime counts are 118,040 users
and 98,794 items; Table 2 reports real IDs.

## Gowalla — VERIFIED EXACT

- Official source page: [SNAP Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
- Direct check-in archive: [`loc-gowalla_totalCheckins.txt.gz`](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz)
- Raw archive SHA-256: `c1c3e19effba649b6c89aeab3c1f9459fad88cfdc2b460fc70fd54e295d83ea0`
- Repository prepared atomic file:
  `dataset/AGCF_Gowalla/AGCF_Gowalla.inter`
- Prepared atomic SHA-256:
  `aae40dbcf3a10521093a8daddf2e1aad842cc2b0ffdd6b0d1b0ac8ea6047ee6c`

Preparation contract:

1. Read all raw check-ins.
2. Group by `(user_id, location_id)` and retain only the row with the latest
   timestamp for each pair.
3. Apply a joint iterative 5-core to users and locations.
4. Emit one implicit-feedback interaction per remaining pair.

This produces exactly 64,115 users, 164,532 locations/items, and 2,018,421
interactions.  RecBole reserves `[PAD]`, so runtime `user_num`/`item_num` are
64,116 and 164,533; Table 2 reports real IDs.

## Yelp — PENDING

- Official download page: [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
- Current locally audited archive: `tmp/agcf-data/yelp/Yelp-JSON.zip`
- Current local archive SHA-256:
  `47dd6e4d279ac9d8734ddc30bfb3d78e571b9df4bb95923d7acf9a6ef3d8a4ab`

The official download is updated over time and does not expose a stable,
versioned direct artifact URL.  The hash above identifies only the snapshot
downloaded during the 2026-09-02 audit; it does **not** establish that this is
the paper's frozen release.

The paper target is 202,012 users, 86,880 businesses/items, and 3,022,685
interactions.  The exact release date, rating threshold, duplicate policy,
and user/item core thresholds needed to reach that row remain unresolved.
Until one deterministic recipe reproduces all three counts, keep Yelp marked
PENDING and do not report comparisons against the paper's Yelp metrics.

## Preparation tooling status

There is not yet one reviewed, repository-wide preparation command for these
five sources.  Therefore this document records the deterministic algorithms
instead of publishing an unverified one-click command.  Any future preparation
script must, before being documented here, verify the source hash, execute the
steps above, and fail unless all expected real-ID and interaction counts match.

