# Bot detection

Scores each `cookie_id` in `data/test.csv` between 0 and 1 for belonging to
automated data-collection traffic. Metric: precision at recall >= 0.70, computed with `metric.py`.

## Running it

```bash
pip install -r requirements.txt          # Python 3.14.7
jupyter nbconvert --execute --inplace solution.ipynb
```

`solution.ipynb` is the deliverable: it loads the data, builds the features, reproduces
the baseline and the final model, and writes `submission.csv`. Every seed is fixed, so a
second run produces a byte-identical file — the notebook and the command below both
write md5 `8c043f2e2ed5219ca600240ce99e6c47`.

The same result without a notebook:

```bash
python -m src.submit --name final
```

A first run takes about 25 minutes: building the 468 features from the event log is a
few minutes, the rest is cross-validation. Feature matrices are cached under
`artifacts/` keyed by a hash of the feature code, so edits can never serve a stale
matrix and reruns are fast.

## Layout

| path | role |
|---|---|
| `solution.ipynb` | the narrated end-to-end run |
| `src/config.py` | paths, seeds, constants |
| `src/data.py` | loading, platform normalisation, the window filter |
| `src/features.py` | all feature blocks |
| `src/timeseries.py` | the event stream as a point process: binned activity, compression, circular time |
| `src/encoding.py` | fold-safe target encoding of the raw categorical keys (measured, rejected) |
| `src/pipeline.py` | dataset assembly and caching, submission writing |
| `src/model.py` | LightGBM / CatBoost / XGBoost behind one interface, rank blending |
| `src/validate.py` | repeated CV, forward chaining, scoring with `metric.py` |
| `src/experiment.py` | CLI for a single named experiment |
| `src/ablation.py` | leave-one-block-out over the feature families |
| `src/diagnose.py` | metrics split by cookie regime |
| `src/tune.py` | Optuna search, half the objective earned on forward-looking splits |
| `src/ensemble.py` | every subset of the three model families, weights searched on OOF |
| `src/mobile.py` | mobile specialist and two ways of folding it in (measured, rejected) |
| `src/submit.py` | final fit, seed bagging, `submission.csv` |
| `src/report.py` | renders `results.jsonl` as the experiment table |
| `tools/make_notebook.py` | generates `solution.ipynb` |
| `results.jsonl` | every experiment that was run, with its scores |

## Method in one paragraph

Events are clipped to each cookie's observation window — 12.4% of the raw rows are
timestamped after `window_end_ts` and would not be available at scoring time. From the
surviving events, 457 features are built in 16 families covering event volume and mix,
transition bigrams, the micro-structure of inter-event gaps, session shape, content
diversity, pagination sweeps, cursor trajectory geometry, platform and User-Agent,
dwell times, context switching, catalogue profile, item-id structure, sequence n-grams, the event stream seen as a point process,
percentile ranks within the cookie's own platform, and the event stream described as a
point process. The shipped model is a single LightGBM bagged over ten seeds in rank
space: a blend of three gradient boosting families was measured on identical folds and
the weight search gave CatBoost and XGBoost zero weight, and two Optuna searches both
lost to the hand-set parameters. Decisions are made on repeated stratified CV over the
whole training set, cross-checked against day-by-day forward chaining; a change is kept
only when PR-AUC rises and neither P@R70 nor the forward-chaining score falls.

## Third-party components

pandas, numpy, scikit-learn, LightGBM, CatBoost, XGBoost, Optuna, matplotlib — all
open source, all run locally. No external APIs and no language models are used.
