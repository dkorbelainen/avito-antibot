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
second run produces a byte-identical file.

The same result without a notebook:

```bash
python -m src.submit --name final
```

## Layout

| path | role |
|---|---|
| `solution.ipynb` | the narrated end-to-end run |
| `src/config.py` | paths, seeds, constants |
| `src/data.py` | loading, platform normalisation, the window filter |
| `src/features.py` | all feature blocks |
| `src/pipeline.py` | dataset assembly and caching, submission writing |
| `src/model.py` | LightGBM / CatBoost / XGBoost behind one interface, rank blending |
| `src/validate.py` | repeated CV, forward chaining, scoring with `metric.py` |
| `src/experiment.py` | CLI for a single named experiment |
| `src/ablation.py` | leave-one-block-out over the feature families |
| `src/diagnose.py` | metrics split by cookie regime |
| `src/tune.py` | Optuna search on PR-AUC |
| `src/submit.py` | final fit, blend weights, `submission.csv` |
| `src/report.py` | renders `results.jsonl` as the experiment table |
| `tools/make_notebook.py` | generates `solution.ipynb` |
| `results.jsonl` | every experiment that was run, with its scores |

## Method in one paragraph

Events are clipped to each cookie's observation window — 12.4% of the raw rows are
timestamped after `window_end_ts` and would not be available at scoring time. From the
surviving events, 423 features are built in 15 families covering event volume and mix,
transition bigrams, the micro-structure of inter-event gaps, session shape, content
diversity, pagination sweeps, cursor trajectory geometry, platform and User-Agent,
dwell times, context switching, catalogue profile, item-id structure, sequence n-grams,
and percentile ranks within the cookie's own platform. Three gradient boosting models
are tuned on PR-AUC and blended in rank space. Decisions are made on repeated
stratified CV over the whole training set, cross-checked against day-by-day forward
chaining; a feature family is kept only when PR-AUC rises and neither P@R70 nor the
forward-chaining score falls.

## Third-party components

pandas, numpy, scikit-learn, LightGBM, CatBoost, XGBoost, Optuna, matplotlib — all
open source, all run locally. No external APIs and no language models are used.
