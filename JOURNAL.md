# Experiment journal

Running log of what was tried, what it scored, and what it taught us.
Source of truth for the "what we tried" narrative in `solution.ipynb`.
Machine-readable twin: `logs/results.jsonl`.

Metric: `P@R70` from the official `metric.py`. Validation: RepeatedStratifiedKFold
5 folds x 5 seeds over the full train, unless noted.

| # | Date | Experiment | Features | P@R70 | ROC-AUC | PR-AUC | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | 2026-09-19 | EDA baseline, ~50 hand features, LGBM, single seed | 50 | 0.707 | 0.927 | 0.723 | anchor |
| 1 | 2026-09-19 | same baseline through the harness (3 seeds) | 47 | 0.7446 | 0.9273 | 0.7757 | reference |
| 2 | 2026-09-19 | core A: 11 feature blocks, LGBM | 178 | **0.7785** | 0.9348 | 0.7880 | accepted, +0.034 |
| 3 | 2026-09-19 | core A, CatBoost | 178 | 0.7555 | 0.9373 | 0.7849 | blend partner |
| 4 | 2026-09-19 | core A, XGBoost | 178 | 0.7686 | 0.9358 | 0.7866 | blend partner |
| 5 | 2026-09-19 | + catalog shares, id structure, dt granularity, dwell, navigation, deep pointer geometry | 353 | **0.7850** | 0.9394 | 0.7933 | accepted, forward-chain 0.7482 -> 0.7705 |
| 6 | 2026-09-19 | + peer-relative ranks (day and platform) + platform-scoped blocks | 457 | 0.7953 | 0.9417 | 0.7956 | **rejected by the gate**: forward-chain 0.7705 -> 0.7369 |
| 6a | 2026-09-19 | same, without the day-relative ranks | 437 | 0.7949 | 0.9410 | 0.7955 | forward-chain restored to 0.7693 |
| 6b | 2026-09-19 | same, without the platform-scoped blocks | 394 | 0.7987 | 0.9414 | 0.7966 | forward-chain 0.7637 |
| 7 | 2026-09-19 | **core set**: 6a minus `window_day_index`, 5 seeds | 436 | 0.7844 | 0.9424 | 0.7949 | accepted, forward-chain 0.7709 |
| 7a | 2026-09-19 | core set with in-fold top-200 gain selection, 3 seeds | 200/436 | 0.7967 | 0.9408 | 0.7950 | rejected, no gain |
| 7b | 2026-09-19 | core set with 13 exactly duplicated columns removed | 423 | — | — | — | housekeeping, carried into every later run |
| 8 | 2026-09-19 | **leak-free fitting**: bigram and catalogue vocabularies and peer-rank references fitted on train only | 423 | **0.7936** | 0.9416 | 0.7972 | accepted as the new reference, fc 0.7575 |
| 9 | 2026-09-19 | + time-series block (binned activity, gzip repetitiveness, circular time, gap shape), global and mobile-scoped | 491 | 0.7864 | 0.9419 | 0.7959 | rejected, PR down |
| 9a | 2026-09-19 | same, scoped blocks removed entirely | 394 | 0.7873 | 0.9422 | 0.7963 | rejected, fc 0.7498 |
| 9b | 2026-09-19 | same, only the mobile copy of the time-series block removed | 457 | 0.7924 | 0.9415 | 0.7977 | rejected, fc 0.7417 |
| 10 | 2026-09-19 | reference set, binned activity only from the time-series block | 447 | 0.7888 | 0.9415 | 0.7969 | rejected |
| 11 | 2026-09-19 | reference set + `is_unbalance` | 423 | 0.7945 | 0.9419 | **0.7991** | accepted, R@FPR1% 0.6345 -> 0.6407 |
| 12 | 2026-09-19 | 9b + `is_unbalance` | 457 | 0.7917 | 0.9422 | **0.7993** | accepted, fc 0.7575 -> 0.7692 |
| 13 | 2026-09-19 | core + recency sample weights, half-life 7 days | 457 | 0.7969 | 0.9419 | 0.7979 | rejected, PR down |
| 13a | 2026-09-19 | same, half-life 14 days | 457 | 0.7969 | 0.9425 | 0.7987 | rejected, PR down |
| 14 | 2026-09-19 | core with Optuna-tuned LightGBM (random-CV objective) | 457 | 0.7953 | 0.9412 | 0.8004 | rejected by the gate: fc 0.7692 -> 0.7380 |
| 15 | 2026-09-19 | **scripted HTTP clients split out of the User-Agent**, plus app/OS/browser versions and device rarity | 468 | **0.8034** | 0.9426 | **0.8018** | accepted, every metric up, fc 0.7740 |
| 16 | 2026-09-19 | fold-safe target encoding of exact UA, dominant category and dominant location | 471 | 0.7962 | 0.9387 | 0.7969 | rejected, every metric down |
| 17 | 2026-09-19 | Optuna with half the objective earned on forward-looking splits | 468 | 0.7974 | 0.9422 | 0.8003 | rejected, fc 0.7740 -> 0.7364 |
| 18 | 2026-09-19 | coordinate sweep around the defaults, ten points, three seeds | 468 | 0.8069 | 0.9422 | 0.8014 | defaults kept, every move loses |
| 19 | 2026-09-19 | leave-one-family probes on the catalogue and query blocks | 373 / 428 | 0.7933 / 0.7987 | 0.9432 | 0.8013 | both families kept |
| 20 | 2026-09-19 | **item popularity block** plus six micro-probes (device build, calendar, `item_id` scale, screen position, UA rarity, cookie age) | 475 | **0.8807** | 0.9489 | **0.8307** | accepted, fc 0.8778, F32 |
| 21 | 2026-09-19 | `scale_pos_weight` sweep 3 / 6 / 11.3 / 25 | 475 | 0.8849 at 25 | 0.9487 | 0.8303 | rejected, stock `is_unbalance` kept, F37 |
| 22 | 2026-09-19 | + crowd-relative block (dwell and page depth against the population) | 479 | 0.8808 | 0.9490 | 0.8306 | accepted, fc 0.8739, F41 |
| 23 | 2026-09-19 | strict rebuild of the matrix per cut exposes the popularity pool asymmetry | 479 | 0.8462 strict vs 0.8807 random | — | 0.8212 | defect, F38-F39 |
| 24 | 2026-09-19 | **popularity as a percentile inside one symmetric pool** | 479 | **0.8894** | 0.9510 | **0.8363** | accepted, fc 0.8836, strict 0.8827, F40 |
| 25 | 2026-09-19 | coordinate sweep around the defaults on the 479-column set, ten points | 479 | 0.8890 | 0.9510 | 0.8366 | defaults kept, F44 |
| 26 | 2026-09-19 | 900 and 1200 rounds, `extra_trees`, `path_smooth` | 479 | 0.8903 best | 0.9503 | 0.8371 best | rejected, gain below seed noise, F51 |
| **final** | 2026-09-19 | **LightGBM, defaults, bagged over 10 seeds** | 479 | **0.8894** (0.8911 rank-blended) | **0.9510** | **0.8363** | shipped, fc 0.8836, md5 `ffb2ec0bfa42d1c1256bdcb5d149132a` |

## Findings

### F1 — 12.4% of events lie after `window_end_ts`
All events start at `window_start_ts` (0% before), but 12.38% fall after the window
ends. The task forbids features that are not available at window end, so these rows
are dropped. This is a deliberate trap in the data.

### F2 — P@R70 is noise-dominated
Per-fold spread 0.61..0.79 on random 5-fold; per-day forward-chain 0.49..0.79. ROC-AUC
over the same splits stays in 0.90..0.96. Random CV and temporal CV agree, so there is
no meaningful drift — the spread is sample size, not time. Tune on PR-AUC, gate on a
combination, never chase a single P@R70 point.

### F3 — no single strong feature
The best individual feature reaches AUC ~0.71 (`loc_nuniq`, `page_mean`, `dt_med`,
`ev_per_sess`, `dt_iqr`). All signal is in the combination.

### F4 — UA/platform are consistent
No fingerprint mismatch exists in the data (`platform` mobile/desktop always agrees
with the UA string). `HeadlessChrome` appears in 8.3% of bots but also 2% of humans —
a weak feature, not a rule.

### F5 — `captcha_shown` is useless
Present for 0.2% of bot cookies and 0% of human cookies. Keep the count for
completeness, expect nothing from it.

### F6 — no co-visitation structure exists
Item ids are effectively unique per cookie: same-day cookie degree of an `item_id` is
1.08 on average, max 5, only 11126 co-visiting pairs over 21 days. The planned graph
increment (B3) has nothing to stand on and was dropped before implementation.
`search_query` is a closed vocabulary of 120 strings shared by everyone, so query
co-visitation is equally uninformative. What survives from that direction is the
*profile*: per-cookie share of each location / category / query.

### F7 — pointer trajectory is the dominant family
Leave-one-block-out on the 178-feature model, 3 seeds:

| dropped block | P@R70 | delta |
|---|---|---|
| none (full) | 0.7785 | — |
| pointer | 0.6040 | **-0.1745** |
| content | 0.7421 | -0.0364 |
| client | 0.7482 | -0.0303 |
| timing | 0.7623 | -0.0162 |
| volume_mix | 0.7641 | -0.0144 |
| pagination | 0.7647 | -0.0138 |
| sessions | 0.7668 | -0.0117 |
| transitions | 0.7696 | -0.0089 |
| window | 0.7779 | -0.0006 |

Cursor behaviour carries an order of magnitude more signal than anything else, which
is why the second feature wave went almost entirely into trajectory geometry
(straightness, turn angles, speed quantiles, grid snapping, quadrant coverage).

### F8 — the pointer signal is a *missingness* signal
On web/desktop events the cursor field is populated for 66.2% of human events but only
33.0% of bot events. Android and iOS never carry coordinates at all, so raw
`pointer_coverage` mixes "mobile, so no cursor" with "web, but no cursor" and dilutes
the effect. Conditioned on web/desktop cookies only, coverage alone reaches AUC 0.655
over 5917 cookies. That conditioning is what `webp_pointer_coverage` adds.

Note the consequence for the metric: only 28% of bots have any cursor data, so
reaching recall 0.70 is impossible on cursor-bearing cookies alone. The pointer family
sharpens the top of the ranking; the recall requirement is still carried by
mobile and cursor-less cookies, and that is where the remaining headroom sits.

### F9 — duplicate timestamps are not a tell
Shares of events sharing a timestamp with another event (2.2% human, 1.8% bot) and of
zero-second gaps (1.9% vs 1.6%) are indistinguishable. Dropped.

### F10 — the metric is decided on mobile and low-volume cookies
Same model (353 features, 2 seeds), OOF split by regime:

| regime | n | bots | P@R70 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| web_major | 5917 | 588 | 0.8684 | 0.9697 | 0.8709 |
| has_pointer | 4046 | 251 | 0.8462 | 0.9767 | 0.8405 |
| many_events (>23) | 2708 | 389 | 0.9320 | 0.9696 | 0.9108 |
| no_pointer | 7045 | 648 | 0.6756 | 0.9123 | 0.7645 |
| **mobile_major** | 5174 | 311 | **0.3597** | 0.8767 | 0.6277 |
| **few_events (<=8)** | 3899 | 150 | **0.2419** | 0.8825 | 0.5493 |

Web traffic is close to solved. Everything left is mobile and short sessions, so the
third feature wave targets those: platform-scoped timing and navigation recomputed on
android/ios events alone, and peer-relative percentile ranks (within the same window
day, and within the same dominant platform) so the model compares a cookie with its
peers instead of against an absolute threshold.

### F11 — epsilon guards were producing 1e6-scale garbage
Ratios written as `a / (b + 1e-9)` blew up whenever the denominator was a zero count
(`locations_per_category` reached 7.4e6). Count denominators now use `+ 1.0` and
second-scale denominators use `+ 1.0` or a clipped minimum. This surfaced in the
caught-vs-missed comparison, not in any metric — the trees simply split around it.

### F12 — within-day ranks look good and travel badly
Adding both peer-relative families at once raised random-CV P@R70 to 0.7953 but pushed
forward-chaining down from 0.7705 to 0.7369. Splitting the two families apart named the
culprit: dropping the *day*-relative ranks restores forward-chaining to 0.7693 at the
same CV score, while dropping the *platform*-relative ones does not. A percentile inside
a window day describes that day's population, which is exactly the thing that does not
carry to unseen days. Platform-relative ranks stay; day-relative ranks are out.

This is the gate doing its job — on random CV alone the rejected variant was the best
model of the session.

### F13 — `window_day_index` cannot transfer
Adversarial-style drift check over all features: only one column shifts materially
between train and test, and it shifts enormously (standardised mean shift 2.83, train
mean 6.19 vs test 17.18). Test windows lie entirely after the training range, so every
split on an absolute day index sends test rows to the same leaf. Dropped. Every other
feature shifts by at most 0.12 standard deviations and no NaN rate moves by more than
2 percentage points, which is the evidence that the pipeline itself is sound.

### F14 — feature selection buys nothing here
In-fold top-200 selection by LightGBM gain (screening model refitted inside every
fold, so the validation part never helps choose its own features) lands on PR-AUC
0.7950 against 0.7949 for the full 436 and a slightly lower ROC-AUC. The P@R70 looks
higher, but that comparison runs at 3 seeds against 5 and sits well inside the seed
spread. No selection: LightGBM's own column sampling already handles the width.

### Note on comparing runs
Seed count changes the mean, so scores are only comparable at equal seed counts.
Experiment 7 is the first run at the full 5 seeds; earlier entries are 3-seed runs and
read slightly differently. When a comparison mattered it was re-run at matching seeds.

### F15 — thirteen feature columns were exact duplicates
Platform-scoped blocks silently reproduced their global counterparts whenever the scope
covered every row that carries the field: the cursor exists only on web, so every
`webp_ptr_*` column equalled its `ptr_*` twin. Two more pairs came from different
routes to the same quantity (`uafam_HeadlessChrome` equals `ua_headless_rate`,
`funnel_swipe_per_item` equals `ratio_photo_item`). Duplicates cost nothing in accuracy
but dilute column sampling — a duplicated column is drawn twice as often as a unique
one — so the matrix builder now drops them, 436 -> 423.

### F16 — the mobile specialist is worse than the general model on mobile
A LightGBM fitted on the 5174 mobile cookies alone (418 columns that are not empty
inside the subgroup, folds stratified inside it) scores PR-AUC 0.6160 on mobile against
0.6327 for the model fitted on everything. 311 positives cannot support a separate
model, and the web rows the specialist throws away were evidently teaching it something
transferable. Both ways of folding it back in were measured:

| combination | global P@R70 | global PR | mobile P@R70 | mobile PR |
|---|---|---|---|---|
| general model alone | 0.7975 | 0.8010 | 0.3597 | 0.6327 |
| reorder, specialist weight 0.25 | 0.7975 | 0.8006 | 0.3562 | 0.6312 |
| reorder, weight 0.75 | 0.8008 | 0.7980 | 0.3318 | 0.6232 |
| second-level stack | 0.7945 | 0.7966 | 0.3488 | 0.6209 |

Every variant trades PR-AUC for a P@R70 move inside its own noise. The subgroup-model
route is closed; mobile has to be improved through features the general model can use.

### F17 — the time-series block does not survive the gate
Binned activity (1/5/30-minute bins: peak, mean, Fano, occupancy, entropy, run
structure), gzip compression ratios of the action / category / location / paced
streams, circular time-of-day concentration, log-scale gap shape and Bandt-Pompe
permutation entropy of the gap series. Inside mobile these are the strongest single
columns available — `b5m_max` alone reaches ROC-AUC 0.72 there, against 0.63 PR-AUC for
the whole model — yet none of the three ways of adding them lifts the full set. The
reason is visible in the per-subgroup AUC table: for a mobile-dominant cookie the
platform-scoped copy equals the global column exactly, so the block arrives half
duplicated, and the columns it does add overlap the gap quantiles that were already
there.

### F18 — mobile has no cheap signal left
Measured on the mobile subgroup and discarded: app version, OS version, device model
and native-app flag from the User-Agent (best: `okhttp_rate` ROC-AUC 0.628, already
carried by `uafam_other`); symbolic periodicity of the action sequence (best matching
lag over 1..8, ROC-AUC 0.49); same-query increasing-page sweeps (0.53); category-lock
share (0.60, already in `item_category_entropy`); dwell dispersion on item views
(0.48). The one strong candidate, locations per category (0.705), is the feature
`locations_per_category` that the set already has at 0.728.

### F19 — the time-series block only pays once the classes are rebalanced
Added on its own the block was a small loss (experiment 9). With `is_unbalance` on it
is the best set measured: PR-AUC 0.7993 against 0.7991 for the same rebalanced model
without it, and forward-chaining 0.7692 against 0.7547 — the clearest gap of the two.
Rebalancing moves the fitted leaves toward the 8% positive class, and the block's
columns (binned peaks, gzip repetitiveness) describe exactly the positives that the
unweighted model was under-fitting. Two changes that each look marginal alone are worth
keeping together; measuring them one at a time would have rejected both.

### F20 — recency weighting does not pay, even though the test set is later
Exponential sample weights on the window day (half-lives 7 and 14 days) move P@R70 up
by 0.005 and PR-AUC down by 0.001, with forward-chaining unchanged at 0.7692. Both
moves sit inside the seed spread, and the drift check (F13) already said why: apart
from the absolute day index, which is gone, no feature shifts by more than 0.12
standard deviations between the two week-halves. There is no drift for the weights to
correct, so all they do is throw away half the training signal.

### F21 — tuning on random CV buys PR-AUC and sells transfer
The Optuna optimum lands on far weaker regularisation than the hand-set defaults
(`lambda_l1` 0.035 and `lambda_l2` 0.53 against 5.0, `feature_fraction` 0.36 against
0.7). On random CV that is the best model measured — PR-AUC 0.8004 and recall at 1%
FPR 0.6472, both the highest of the session — and on forward-chaining it is the worst
of the accepted line, 0.7380 against 0.7692. Random folds mix days, so a model can
memorise day-specific structure and still score; the real split cannot. The search
objective was changed to earn half its score on forward-looking splits.

### F22 — the User-Agent family list was hiding the scrapers
The family regex covered browsers only — `YaBrowser|HeadlessChrome|Firefox|Chrome|
Safari` — so every non-browser client landed in `uafam_other`. That bucket held both
the six scripted HTTP clients present in the data (curl, Scrapy, node-fetch,
python-requests, python-urllib3, Go-http-client; 25-33% bot rate each) and the native
Avito app, which is the largest non-browser client at 52k events and mostly human. One
column was averaging the strongest positive evidence in the dataset with its opposite.

Naming the six explicitly, adding their aggregate share, and extracting the app / OS /
browser version numbers and device rarity moved every metric at once: P@R70 0.7917 to
0.8034, PR-AUC 0.7993 to 0.8018, forward-chaining 0.7692 to 0.7740. The largest single
gain of the session, and it came from reading the raw field rather than from a model
change.

Exact User-Agent strings also transfer: all 148 strings in test appear in train, and
the per-string bot rate correlates 0.59 between the two halves of the training week.

### F23 — the blend is not worth it here
Every subset of the three families, out-of-fold on identical folds and seeds, weights
searched on PR-AUC:

| combination | weights | P@R70 | ROC-AUC | PR-AUC | R@FPR1% |
|---|---|---|---|---|---|
| **lgb** | 1.00 | **0.8077** | 0.9451 | **0.8077** | 0.6607 |
| lgb+xgb | lgb 1.00, xgb 0.00 | 0.8077 | 0.9451 | 0.8077 | 0.6607 |
| lgb+cat | lgb 1.00, cat 0.00 | 0.8077 | 0.9451 | 0.8077 | 0.6607 |
| lgb+cat+xgb | lgb 1.00, rest 0.00 | 0.8077 | 0.9451 | 0.8077 | 0.6607 |
| cat+xgb | cat 0.20, xgb 0.80 | 0.7925 | 0.9458 | 0.7983 | 0.6196 |
| xgb | 1.00 | 0.7875 | 0.9458 | 0.7980 | 0.6229 |
| cat | 1.00 | 0.7826 | 0.9430 | 0.7916 | 0.6129 |

Measured on the 468-column set; F43 repeats it on the final 479 columns and the answer
does not move — LightGBM still takes weight 1.00 in every subset, at 0.8911 P@R70 and
0.8404 PR-AUC against 0.8317 for XGBoost and 0.8231 for CatBoost.

The grid runs in fifths, so the search had 0.80/0.20 available and still chose to give
the other two families nothing. CatBoost and XGBoost are not merely weaker here, they
are not decorrelated enough to pay for their weakness: LightGBM's errors are a subset
of theirs on the features that matter. The shipped model is one family, bagged over
seeds — the seed bag is what lifts 0.8018 (mean over seeds) to 0.8077 (rank-averaged
across them); on the final set the same step reads 0.8894 -> 0.8911.

This is worth stating plainly because the blend was in the design from the start. It
was measured rather than assumed, and it lost.

### F24 — target encoding adds nothing once the User-Agent is parsed properly
Smoothed positive rates for the exact User-Agent string, the dominant category and the
dominant location, refitted inside every fold. Standalone the User-Agent encoding is a
real signal (out-of-fold ROC-AUC 0.653 against a 0.5 baseline; category 0.512, location
0.569). Added to the set it costs on all four metrics: P@R70 0.8034 to 0.7962, ROC-AUC
0.9426 to 0.9387, PR-AUC 0.8018 to 0.7969, forward-chaining 0.7740 to 0.7552.

The reason is experiment 15. Once the client family, the scripted-client share and the
app / OS / browser version numbers are columns of their own, the encoded string carries
almost nothing the model does not already have, and what it does carry is a
per-fold-noisy 148-level statistic that the trees happily overfit. Measured before
experiment 15 it might well have passed; measured after, it is redundant.

### F25 — two hyper-parameter searches, both beaten by the hand-set defaults
The second search earned half its objective on forward-looking splits (train on days
below a cut, score the days at and after it, cuts at 7/9/11) precisely to stop the
first one's failure mode. It still landed on weak regularisation — `lambda_l2` 0.028,
`learning_rate` 0.071 — and still lost on the full protocol: P@R70 0.7974 against
0.8034, PR-AUC 0.8003 against 0.8018, forward-chaining 0.7364 against 0.7740.

What both searches have in common is the cheap objective: three folds, one seed, rounds
estimated on a single fold. That is a noisy surrogate for a five-fold five-seed mean,
and Optuna optimises the surrogate's noise. The search was replaced by a coordinate
sweep around the defaults measured on the real protocol, which costs more per point and
answers the question that is actually being asked.

### F26 — the hand-set defaults sit in a local optimum
Coordinate sweep on the real protocol (five folds, three seeds), one parameter moved at
a time from the defaults:

| point | P@R70 | PR-AUC |
|---|---|---|
| **defaults** | 0.8069 | **0.8014** |
| `feature_fraction` 0.5 | 0.8026 | 0.8003 |
| `lambda_l2` 2.0 | 0.7975 | 0.8004 |
| `learning_rate` 0.015 | 0.7941 | 0.8003 |
| `lambda_l2` 15.0 | 0.8015 | 0.7996 |
| `min_data_in_leaf` 20 | 0.7926 | 0.7993 |
| `feature_fraction` 0.85 | 0.7875 | 0.7987 |
| `num_leaves` 21 | 0.7916 | 0.7975 |
| `num_leaves` 47 | 0.7959 | 0.7975 |
| `min_data_in_leaf` 80 | 0.7908 | 0.7957 |

Every single-parameter move loses PR-AUC. Ten points measured the way decisions are
actually made beat sixty Optuna trials measured on a cheap surrogate.

### Reproducibility check
Three independent runs write `submission.csv` with md5
`8c043f2e2ed5219ca600240ce99e6c47`: `python -m src.submit --name final`, the same
command again with `artifacts/` emptied so every feature is rebuilt from the raw event
log, and `jupyter nbconvert --execute --inplace solution.ipynb`. The cold run is the one
that matters — it is the reviewer's situation, and it reproduces the submitted file
rather than a file that happens to look like it.

### Where the session ended up
Baseline 0.7446 P@R70, shipped 0.8894 — of that, the largest single step by far is the
item popularity block (+0.077, and +0.086 once it was rebuilt as a percentile inside one
pool), followed by the first feature core (+0.034) and the User-Agent client split
(+0.012); the rest is accumulated small gains. Everything in the "rejected" list was measured on the same
protocol and rolled back, including three ideas that were in the original design: the
model blend, hyper-parameter tuning, and a subgroup specialist for mobile traffic.

### F27 — pseudo-labelling: out on the metric, and out on principle
Tried for completeness: each fold's model scores the unlabelled test rows, the confident
tails are appended to that fold's training set as extra rows at 0.3 weight, and the fold
is refitted. Borrowing the top and bottom 10% of the test distribution costs PR-AUC
0.8018 -> 0.8003; borrowing 25% costs 0.8018 -> 0.7948. The confident tails are exactly
the rows the model already gets right, so they add weight without adding information,
and the errors they do carry are self-reinforcing.

It was also the wrong shape of solution regardless of the number: it makes the model
transductive — it needs the scored population present at training time — which does not
sit well with a task that asks for features available at the end of each window. The
code was removed rather than kept behind a flag.

## Session 2 — where the remaining loss actually sits

### F28 — the weak subgroups are weak by base rate, not by ranking
Per-regime P@R70 is misleading on its own. The ceiling measurement answers the real
question: freeze the model, then permute the scores inside one subgroup into perfect
order while keeping that subgroup's own score values, and re-score globally.

| subgroup | n | bots | rate | P@R70 ceiling | gain |
|---|---|---|---|---|---|
| web | 5917 | 588 | 0.099 | 0.9663 | **+0.1586** |
| no_pointer | 7045 | 648 | 0.092 | 0.9468 | +0.1391 |
| many_events (>23) | 2708 | 389 | 0.144 | 0.9296 | +0.1219 |
| mid_events (9..23) | 4484 | 360 | 0.080 | 0.9157 | +0.1080 |
| mobile | 5174 | 311 | 0.060 | 0.8965 | +0.0888 |
| few_events (<=8) | 3899 | 150 | 0.038 | 0.8564 | +0.0487 |
| mobile & few_events | 1849 | 54 | 0.029 | 0.8344 | +0.0267 |

At the operating point (threshold at recall 70: 630 true positives, 150 false
positives) web holds 66% of the false positives and 76% of the true positives; mobile
holds 34% and 24%. The few-events subgroup holds 150 of the 899 bots and contributes 10%
of the true positives — perfect ordering inside it is worth +0.049, and inside its
intersection with mobile +0.027.

So the intuition that mobile and few-events are where the score is lost does not
survive measurement: their low in-group P@R70 is a base-rate effect (6.0% and 3.8%
positives against 9.9% on web), and the largest single pool of recoverable error is on
web.

### F29 — the cross-group calibration is already right
If mobile were systematically mis-scaled against web, re-scaling each group with a
label oracle would lift the global metric. It does not: replacing every score with the
in-group, in-bin bot rate (order inside the group preserved) never beats the model —
0.8008 at 24 bins, 0.7418 at 8, 0.7216 at 12, against 0.8077 on the 468-column set. F48 repeats the check, cross-fitted, on the final set. The loss is ordering
inside the groups, not the groups' placement against each other.

### F30 — train and test are indistinguishable, so nothing needs dropping for drift
Adversarial validation (LightGBM, train vs test, 5 folds over all 468 columns) gives AUC
0.5241. No single column reaches univariate drift AUC 0.60; the highest is `window_dow`
at 0.535, which is the calendar. Feature pruning therefore cannot be justified as a
transfer fix — only as a noise-dilution fix.

### F31 — dead ends measured and dropped
* Template similarity: nearest-neighbour cosine on the event-sequence TF-IDF, against
  train cookies only, self excluded. Best column AUC 0.60 (`nn_paced_top5`); the exact
  twin count runs the wrong way (humans have more near-duplicates than bots).
* Co-visitation: nearest-neighbour cosine on the item-id bag — AUC 0.59.
* Diurnal shape against the population (KL, total variation, overlap) — AUC 0.45..0.51.
  Windows are 24 h, so every day-structure statistic is degenerate as well.
* Markov surprise under a population transition table — AUC 0.62 inverted globally, but
  the sign flips between web (0.725 inverted) and mobile (0.552 upright), and the
  existing bigram-share block already carries most of it.
* Clock alignment (`second == 0`, `% 5`, `% 10`) — AUC 0.52..0.55; timestamps have no
  sub-second part at all.

### F32 — item popularity is the strongest single family in the set
A listing's audience counted over the fitting cookies only, with the cookie's own
contribution removed: `pop_aud_mean/med/max/std`, `pop_views_mean/std`, `pop_solo_share`.
Seven columns. Univariate ROC-AUC 0.777 — higher than anything else measured (F3 put the
previous best single feature at 0.71) — and 0.745 when stratified by event count, so it
is not a volume proxy. Per subgroup: web without cursor 0.814, web with cursor 0.753,
mobile 0.748.

Bots sit on listings the crowd also opens (mean audience e^1.57 ≈ 3.8 other cookies)
while people keep landing on listings nobody else in the window touched (≈ 2.6).

Grouped permutation importance, one fold, PR-AUC drop when the group is shuffled:

| group | n | PR drop | per column |
|---|---|---|---|
| **popularity** | 7 | **0.0990** | **0.0141** |
| content | 22 | 0.0456 | 0.0021 |
| pointer_deep | 26 | 0.0413 | 0.0016 |
| pointer_basic | 13 | 0.0324 | 0.0025 |
| seq_svd | 16 | 0.0199 | 0.0012 |
| mobile_scoped | 39 | 0.0157 | 0.0004 |
| … | | | |
| platform | 7 | -0.0003 | — |
| catalog_loc | 40 | -0.0006 | — |
| id_structure | 8 | -0.0007 | — |
| navigation | 14 | -0.0012 | — |
| catalog_query | 40 | -0.0012 | — |
| catalog_cat | 15 | -0.0017 | — |

Six groups have a *negative* drop: the model scores better with them shuffled. All three
catalogue-share groups are in that list, which is the same verdict null-importance gave
from a different direction.

### F33 — F6 closed the population axis too early, and it was wrong
F6 measured same-day cookie degree of an `item_id` at 1.08 and concluded there was no
co-visitation structure to build on. Over the whole 15-day window the mean audience is
2.20, and the difference between the classes is large. The graph *edges* really are
sparse — nearest-neighbour cosine on the item bag scores AUC 0.59 (F31) — but the
*vertex degree* is the strongest feature in the set. Sparse co-visitation and
informative popularity are not the same statement.

### F34 — the popularity signal survives a change of cohort
The worry with any population statistic is that a held-out training cookie is scored
against its own contemporaries while a test cookie never is. Measured by splitting the
training week:

| reference pool | scored on | AUC |
|---|---|---|
| days 0..6 | days 0..6, self excluded | 0.7596 |
| days 0..6 | days 7..13 (test-like) | 0.7277 |
| whole train | whole train, self excluded | 0.7772 |

Moving to a disjoint later cohort costs 0.032 AUC and keeps the rest. The column is a
property of the listing, not of the cookie's neighbours, so it transfers.

### F35 — what the raw event file does and does not hide
Prompted by the quickstart's "check the event file itself" hint:
* `eid` takes ten values (100..900) and is a second spelling of `event_name`. No
  ordering information, nothing to mine.
* File order is neither `eid` order nor timestamp order, but `clip_to_window` sorts by
  `(cookie_id, event_ts)`, so no timing feature was ever computed on file order. Zero
  cookies are out of time order after the clip and there are no negative gaps.
* 4297 exact duplicate rows survive into the features, touching 2362 cookies. As a
  feature the duplicate share reaches AUC 0.549 and the count 0.562 — the same verdict
  as F9 for shared timestamps. Not worth a column.
* 68 rows sit exactly on `window_end_ts`. The task defines the window as
  `window_start_ts <= event_ts < window_end_ts`; `clip_to_window` used an inclusive
  `between`. Corrected, and `data.py` added to the feature-cache hash so the fix cannot
  be served a stale matrix.

### F36 — the popularity axis stops at the plain audience count
Everything else tried on the same axis is weaker or does not transfer:

| candidate | AUC | verdict |
|---|---|---|
| audience / category mean audience | 0.7751 | the same signal, no gain over 0.777 |
| hours since another cookie first opened the listing | 0.5899 | weak, 0.545 across cohorts |
| audience per hour of listing exposure | 0.6566 | **0.344 on a later cohort — the sign flips** |
| other cookies on the same listing within 5 / 60 min | 0.50 / 0.58 | no burst structure |
| rank correlation of visit order against audience | 0.5073 | cookies do not walk a popularity-ordered list |
| `search_page` never decreases | 0.579 inverted | already inside the pagination block |

The exposure rate is the instructive failure: the length of a listing's exposure window
depends on how long the reference pool observed it, so the statistic means something
different for a cohort observed over 7 days than over 15.

### F37 — class weight: the default rebalancing is still the best point
Five points at 3 seeds on the 475-feature set:

| setting | P@R70 | PR-AUC | R@FPR1% |
|---|---|---|---|
| **`is_unbalance`** | 0.8824 | **0.8308** | 0.7219 |
| `scale_pos_weight` 25 | **0.8849** | 0.8303 | **0.7275** |
| `scale_pos_weight` 11.3 | 0.8797 | 0.8279 | 0.7189 |
| no weighting | 0.8758 | 0.8281 | 0.7204 |
| `scale_pos_weight` 3 | 0.8759 | 0.8262 | 0.7178 |
| `scale_pos_weight` 6 | 0.8712 | 0.8245 | 0.7197 |

Also worth stating because it comes up: multiplying the predicted probabilities — prior
correction after undersampling, `p' = pw / (pw + 1 - p)` — cannot change any of these
numbers. The metric reads the ranking, and the correction is strictly monotone in `p`.
Measured to be sure: P@R70, ROC-AUC and PR-AUC are identical to six decimals under
prior correction, under an odds multiplier and under `sqrt(p)`. The only way a
multiplier moves the metric is by clipping, which creates ties and destroys it
(P@R70 0.8077 -> 0.0986 for `clip(5p, 0, 1)`).

### F38 — the popularity gain is mostly an artefact of when the reference pool was counted
The standard protocol confirmed the block: P@R70 0.8807 ±0.0042, PR-AUC 0.8307,
forward-chain 0.8778, every metric up, seed spread halved. The strict check says that
number will not survive contact with the hidden test.

The matrix is built once with `fit_ids` = every training cookie, so in any split taken
*inside* that matrix a validation cookie is described by a listing audience that counted
the validation cookie's own contemporaries. A test cookie never is: its reference pool is
the earlier weeks. Rebuilding the matrix from the first week alone and scoring the
second week reproduces the test situation.

| matrix fitted on | popularity | P@R70 | ROC-AUC | PR-AUC | R@FPR1% |
|---|---|---|---|---|---|
| whole training week | yes | 0.8773 | 0.9527 | 0.8332 | 0.7132 |
| first week only | yes | 0.7668 | 0.9389 | 0.7875 | 0.6201 |
| whole training week | no | 0.7668 | 0.9421 | 0.7912 | 0.6127 |
| first week only | no | 0.7390 | 0.9417 | 0.7908 | 0.6176 |

Read the two strict rows against each other, which is the comparison that matters:
popularity is worth +0.028 P@R70 and -0.003 PR-AUC, not +0.110 and +0.042. The rest of
the apparent gain came from the scored week being inside the pool that defines the
feature.

Two things cause it, and only the second is fixable:
1. A test cookie's listings are counted against cookies from other weeks, so its
   measured audience is systematically lower — coverage 0.932 against 0.983 and counts
   shifted down. The trees split on an absolute count that means something different on
   each side.
2. The pool is asymmetric by construction. Train cookies are compared with their
   contemporaries; test cookies are not.

The fix for both is the same: make the statistic scale-free and the pool symmetric —
every cookie compared against the cookies observed alongside it, and the audience
expressed as a percentile of that pool's audience distribution rather than a count.

### F39 — the defect was asymmetry, not the feature
Redesign: the listing audience is expressed as a **percentile of the pool's audience
distribution** instead of a count, and every cookie is scored against the **same pool**.
`src/strict.py` rebuilds the whole matrix from the days before the cut, so nothing the
scored period contains reaches a feature. Train on days 0..6, score days 7..13:

| variant | P@R70 | ROC-AUC | PR-AUC | R@FPR1% |
|---|---|---|---|---|
| no popularity | 0.7390 | 0.9417 | 0.7908 | 0.6176 |
| percentile, one pool per week | 0.8462 | 0.9485 | 0.8212 | 0.6912 |
| **percentile, a single pool** | **0.8827** | **0.9540** | **0.8375** | **0.7402** |
| count, train-only reference (F38) | 0.7668 | 0.9389 | 0.7875 | 0.6201 |

The last row is the version that failed. Same signal, same data — what broke it was that
a training cookie was compared with its contemporaries and a test cookie was not, and
that a raw count does not mean the same thing in two pools of different size.

Per-week pools work and a single pool works better, by 0.036 P@R70. Seven days of
traffic is not enough co-viewing to rank a listing against; the audience counts are
small and the percentile comes out coarse. The single pool is the shipped choice.

It does mean the statistic is computed over every observed cookie, the scored batch
included. That is a population feature, not a label-derived one: no target is read, the
event log is the log the scoring system already has at window end, and the pool is
identical on both sides, which is the property that was missing before. The strict
protocol is the evidence — it never lets the scored period into any fitted part, and the
block still carries +0.144 P@R70 and +0.047 PR-AUC there.

`src/strict.py` joins the gate permanently. Forward chaining slices a matrix built once
and cannot see this class of defect.

### F40 — the redesign replicates on a second cut, and the ordinary protocol is honest again
Repeating the strict run at a later cut, train on days 0..9 and score days 10..13:

| cut | popularity | P@R70 | ROC-AUC | PR-AUC | R@FPR1% |
|---|---|---|---|---|---|
| 7 | no | 0.7390 | 0.9417 | 0.7908 | 0.6176 |
| 7 | yes | 0.8827 | 0.9540 | 0.8375 | 0.7402 |
| 10 | no | 0.7524 | 0.9465 | 0.7952 | 0.6045 |
| 10 | yes | 0.8587 | 0.9587 | 0.8322 | 0.6864 |

Two independent temporal cuts, +0.144 and +0.106 P@R70, +0.047 and +0.037 PR-AUC. Not a
property of one split.

Once the pool is symmetric the ordinary protocol agrees with the strict one instead of
running away from it. Five folds, five seeds, 479 columns: P@R70 0.8894 ±0.0025, ROC-AUC
0.9510, PR-AUC 0.8363, R@FPR1% 0.7359, forward-chaining 0.8836. The gap between random
CV and the strict rebuild is now 0.007 P@R70, against 0.110 for the broken version.

### F41 — the crowd-relative block pays, but only once the popularity block is correct
Dwell against the population median for the same event type, and page depth against the
population median for the same query. Individually AUC 0.690 and 0.714 inverted, both
stable across cohorts.

Measured three times, and the first two readings were wrong for the same reason:

| reading | with crowd | without crowd |
|---|---|---|
| ordinary protocol, count-based popularity | 0.8808 / PR 0.8306 | 0.8807 / PR 0.8307 |
| strict protocol, one seed-bagged split | 0.8827 / PR 0.8375 | 0.8773 / PR 0.8388 |
| **ordinary protocol, 5 seeds, percentile popularity** | **0.8894 / PR 0.8363 / fc 0.8836** | 0.8830 / PR 0.8325 / fc 0.8728 |

The first was taken on the broken matrix, where the popularity columns were absorbing
signal that belongs elsewhere. The second is a single temporal split, and P@R70 on one
split has the spread F2 describes. The third is five folds by five seeds and moves every
metric at once, forward chaining included. Kept.

The lesson is procedural: a block cannot be judged against a set that is itself wrong,
and a single split cannot settle a 0.005 difference.

### The line, after the redesign

| step | features | P@R70 | ROC-AUC | PR-AUC | forward-chain | strict, cut 7 |
|---|---|---|---|---|---|---|
| baseline | 47 | 0.7446 | 0.9273 | 0.7757 | 0.7095 | — |
| feature core | 178 | 0.7785 | 0.9348 | 0.7880 | — | — |
| + cursor geometry | 353 | 0.7850 | 0.9394 | 0.7933 | 0.7705 | — |
| vocabularies fitted on train only | 423 | 0.7936 | 0.9416 | 0.7972 | 0.7575 | — |
| + time series, class rebalancing | 457 | 0.7917 | 0.9422 | 0.7993 | 0.7692 | — |
| + User-Agent client split | 468 | 0.8034 | 0.9426 | 0.8018 | 0.7740 | 0.7390 |
| **+ popularity and crowd-relative blocks** | **479** | **0.8894** | **0.9510** | **0.8363** | **0.8836** | **0.8827** |
| shipped, bagged over 10 seeds | 479 | 0.8911 | — | — | — | — |

`submission.csv` md5 `ffb2ec0bfa42d1c1256bdcb5d149132a`.


### F42 — what can still be said about the hidden test, without its labels
Three readings, all on the shipped 479-column set.

**Train and test are indistinguishable.** Adversarial validation over every column:
AUC 0.5193. Not one column shifts by more than 0.25 standard deviations — the largest is
`window_dow` at 0.124, which is the calendar — and not one NaN rate moves by more than
3 percentage points. The new blocks are the quietest part of the matrix: the popularity
columns shift by 0.002..0.027 sd with NaN rates moving 0.2..0.5 pp, and `pop_pct_mean`
reads 0.4864 on train against 0.4907 on test. That is the percentile design doing its
job; the count version it replaced had coverage 0.983 against 0.932.

**Sampling spread at test size.** Bootstrapping the out-of-fold scores at n = 4909, the
size of the test split: P@R70 mean 0.8900, sd 0.0209, 5th to 95th percentile 0.857 to
0.920. At the training size the same bootstrap gives sd 0.0138. The hidden number will
differ from ours by a couple of points for sample-size reasons alone, in either
direction.

**Every single training day now scores well.** Out-of-fold P@R70 per window day ranges
0.776..0.979 over 14 days of 600..900 cookies each, with no day below 0.775. The old
468-column model spread 0.49..0.79 on the same days, so the worst day of the current
model beats the average day of the previous one.

What none of this can rule out: a collector active in the test week whose profile is
absent from the training weeks. Nothing in the data answers that, and no validation
scheme can.

### F43 — the families make the same mistakes
The weight search gives LightGBM 1.00 on the 479-column set. Out of fold, five folds by
five seeds: LightGBM 0.8894 P@R70 / 0.8363 PR-AUC, XGBoost 0.8802 / 0.8274, CatBoost
0.8762 / 0.8193; blended in rank space the best subset is LightGBM alone at 0.8911 /
0.8404.

These numbers replace an earlier reading that was taken on a stale cache — see F50. To check the verdict
is not an artefact of a grid in fifths, the lgb/xgb blend was swept in twentieths:
PR-AUC falls monotonically from 0.8347 at weight 0 to 0.8288 at weight 1, with no local
maximum anywhere. P@R70 peaks at weight 0.10 (0.8862 against 0.8850), which is a
twelfth of the seed spread.

The reason is visible at the operating point. LightGBM produces 82 false positives,
XGBoost 83, and **78 of them are the same cookies**; the misses are 269 and 269 with 257
shared. Spearman correlation of the two rankings is 0.9639. There is no disagreement to
average away.

### F44 — the hyper-parameter defaults survive the new feature set
The coordinate sweep was rerun on 479 columns, ten points, three seeds. PR-AUC:
defaults 0.8366, then 0.8358 (`num_leaves` 47), 0.8348, 0.8347, 0.8347, 0.8346, 0.8341,
0.8340, 0.8322, 0.8307 (`num_leaves` 21). Every single move still loses.

### F45 — with a complete block map, no feature family is dead weight
The earlier ablation covered 11 of the 22 families in the matrix — cursor geometry, the
time-series block, peer ranks, platform-scoped copies, dwell, navigation, catalogue
shares, id structure and gap granularity were in no block at all, which is why
null-importance and permutation importance disagreed with it. Rerun over all 22, PR-AUC
drop when the family is removed:

| family | Δ PR-AUC | | family | Δ PR-AUC |
|---|---|---|---|---|
| popularity | **-0.0351** | | transitions | -0.0032 |
| cursor geometry | -0.0169 | | navigation | -0.0031 |
| content | -0.0079 | | catalogue shares | -0.0029 |
| client and UA | -0.0061 | | dwell | -0.0027 |
| timing | -0.0050 | | peer ranks | -0.0022 |
| platform-scoped | -0.0049 | | window | -0.0017 |
| crowd-relative | -0.0048 | | id structure | -0.0016 |
| cursor | -0.0040 | | time series | -0.0014 |
| pagination | -0.0039 | | sessions | -0.0008 |
| volume and mix | -0.0035 | | | |
| gap granularity | -0.0033 | | | |

Every family is positive. The catalogue shares that null-importance and permutation
importance both called junk are worth -0.0029 once the set around them is correct. No
pruning.

### F46 — the web anatomy, after the popularity block
Splitting web by whether the cursor field is populated at all changes the reading. At
the recall-70 operating point:

| cell | n | bots | rate | TP | FP | FN | in-cell P@R70 | in-cell ROC |
|---|---|---|---|---|---|---|---|---|
| web with cursor | 4046 | 251 | 0.062 | 197 | 37 | 54 | 0.8436 | 0.9776 |
| web without cursor | 1871 | 337 | 0.180 | 271 | 23 | 66 | 0.9425 | 0.9547 |
| mobile | 5174 | 311 | 0.060 | 162 | 17 | 149 | 0.5813 | 0.9084 |

The harder web cell is the one *with* a cursor, not the one without: 6.2% positives
against 18.0%. Cursor-less web cookies are the densest bot population in the dataset and
the model already resolves them.

The web false positives are people who behave like collectors on every axis at once but
less so — `dtg_mode_count` 2.5 against 6.3 for caught bots and 1.6 for humans,
`b1m_mean` 2.1 against 3.3 and 1.5, `page_max` 8.3 against 12.5 and 4.5. The web misses
are the mirror image: slow, irregularly paced, and sitting on listings of merely average
popularity (`pop_pct_mean` 0.499 against 0.733 for caught bots).

### F47 — the quiet web region is already resolved to ROC-AUC 0.964
108 of the 120 web misses live among the 5443 low-intensity web cookies
(`b1m_mean` < 2.5, 342 bots, 6.3%). Inside that region the model's own out-of-fold
ranking reaches ROC-AUC 0.9641, against 0.9729 on the rest of web. 46 of the 421 usable
columns exceed AUC 0.65 there on their own, led by cursor dispersion at 0.92 — bots move
the pointer over a *smaller* range — and the popularity percentiles at 0.74.

That is not a feature gap. What is left in the region is the genuine overlap between
people who browse like collectors and collectors that browse like people.

Cursor kinematics were the one direction with a physical argument behind it, and they
are exhausted too: a Fitts-like correlation between step length and gap reaches AUC
0.537, speed autocorrelation 0.580, step autocorrelation 0.621, acceleration dispersion
0.619, median speed change 0.681, distinct-y-per-event 0.705 — all below the plain
`ptr_x_std` already in the set, which reaches 0.92 inside the region.

### F48 — per-cell calibration loses again, now cross-fitted rather than oracular
F29 used a label oracle and lost. The honest version loses as well: a monotone map from
score to probability fitted per cell (mobile / web-with-cursor / web-without-cursor) on
four folds and applied to the fifth. Isotonic costs 0.0025 P@R70 and 0.0046 PR-AUC;
Platt scaling costs 0.16 P@R70, because a two-parameter sigmoid flattens the top of the
ranking where the metric lives. The three cells are already placed correctly against
each other.

### F49 — seed bagging saturates around five seeds; twenty is not better than ten
Twenty independent out-of-fold runs, each five folds, blended cumulatively in rank space:

| seeds | P@R70 | ROC-AUC | PR-AUC | R@FPR1% |
|---|---|---|---|---|
| 1 | 0.8861 | 0.9516 | 0.8394 | 0.7341 |
| 2 | 0.8911 | 0.9527 | **0.8410** | 0.7386 |
| 5 | 0.8911 | 0.9529 | 0.8404 | **0.7486** |
| 10 | 0.8892 | 0.9531 | 0.8400 | 0.7475 |
| 20 | 0.8898 | 0.9530 | 0.8395 | 0.7464 |

Single seeds spread 0.8788..0.8930 P@R70 and 0.8301..0.8394 PR-AUC, and averaging
removes that spread almost entirely by the second seed. Past five there is no measurable
gain: PR-AUC drifts down by 0.0009 between 5 and 20, which is a third of the
seed-to-seed standard deviation. `BAG_SEEDS` stays at 10 — comfortably past saturation,
and the shipped scores are produced once so the extra five cost nothing.

One caveat on reading this curve: it blends out-of-fold predictions, where the seeds
differ in both the fold split and the model, while the shipped bag refits on the whole
training set and differs only in the model. The test-time bag therefore carries less
diversity per seed than this curve suggests. It does not change the conclusion — the
curve is flat from seed five onward by every metric — but it is the reason for keeping
ten rather than trimming to five.

### F50 — a cache keyed on the column count served a stale model for two experiments
`src/ensemble.py` named its out-of-fold cache
`oof_{kind}_{tag}_{seeds}seeds_{n_columns}f.npy`. Rewriting the popularity block from a
count to a percentile did not change the width — 479 columns before and after — so the
name collided and every later call loaded arrays produced by the previous model. The
cached LightGBM array scored 0.8850 P@R70 / 0.8347 PR-AUC against 0.8911 / 0.8404 for
the real one, with Spearman 0.9677 between them: a different ranking, not a noisy copy.

Reach of the defect: the family comparison in F43 and the two notebook sections that
read those arrays. Not the shipped model, not `submission.csv`, and not any number from
`src.experiment` or `src.strict` — `src/submit.py` imports `config`, `validate`, `model`
and `pipeline`, and never touches the ensemble module.

Fixed by putting the feature-code hash in the cache name, the same hash that already
guards the feature matrix in `src/pipeline.py`. The column count is not a safe key: a
block can be rewritten without changing the width.

### F51 — longer boosting and two structural knobs are all inside the seed noise
A last sweep on the 479-column set, three seeds each, against the defaults at
P@R70 0.8890 / PR-AUC 0.8366:

| variant | P@R70 | ROC-AUC | PR-AUC |
|---|---|---|---|
| defaults, 601 rounds by early stopping | 0.8890 ±0.0032 | 0.9510 | 0.8366 |
| 900 rounds | 0.8898 ±0.0022 | 0.9504 | 0.8370 |
| 1200 rounds | 0.8903 ±0.0008 | 0.9503 | 0.8371 |
| `extra_trees` | 0.8902 ±0.0049 | 0.9516 | 0.8340 |
| `path_smooth = 10` | 0.8891 ±0.0037 | 0.9504 | 0.8335 |

The best PR-AUC move is +0.0005 with a seed spread of ±0.002 to ±0.005, and ROC-AUC
goes down in both longer runs. The gate rejects it by the same rule that rejected
feature blocks: a gain smaller than the seed spread is not a gain. Defaults stand.
