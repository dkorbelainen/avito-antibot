"""Generate solution.ipynb from a single source of truth.

The notebook is the deliverable; this script is how it is authored, so the cells
stay in sync with `src/` instead of drifting as copy-pasted code.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md(r"""
# Детектирование автоматизированного сбора данных по событиям куки

**Метрика:** Precision при Recall ≥ 0.70 (`metric.py` организаторов).
**Данные:** 11 091 куки в train (6–19 апреля), 4 909 в test (20–26 апреля), 328 905 событий.

---

## Подход — коротко

**1. Временная природа задачи.** Train и test разделены по времени и не пересекаются
по `cookie_id`. Поэтому проверяются две схемы валидации: повторный стратифицированный
K-fold на всём train (основная, даёт максимум позитивов в OOF) и forward-chaining по
дням (контрольная, ловит то, что не переносится во времени).

**2. Утечка из будущего.** 12.4 % событий датированы **позже** `window_end_ts`.
Условие задачи требует, чтобы признак был доступен на момент конца окна, поэтому эти
строки отбрасываются. Фильтр живёт в одной функции `data.clip_to_window()` — это
единственное место, где утечка вообще возможна, и оно закрыто проверкой-инвариантом.

**3. Признаки.** Готовых поведенческих признаков нет, собрано 436 в 15 блоках:
объём и микс событий, биграммы переходов, микроструктура интервалов (энтропия, IQR,
burstiness, автокорреляция), структура сессий, разнообразие контента, развёртки
пагинации, **геометрия траектории курсора**, платформа и User-Agent, dwell-время по
типам событий, переключение контекста, профиль по локациям/категориям/запросам,
структура множества `item_id`, TF-IDF по n-граммам последовательности событий,
перцентильные ранги внутри своей платформы.

**4. Что оказалось решающим.** Leave-one-block-out показал, что курсор даёт на порядок
больше остальных блоков (−0.175 P@R70 при удалении против −0.01…−0.04 у прочих).
Причина — не форма траектории, а **сам факт наличия координат**: на web/desktop поле
заполнено у 66 % событий людей и у 33 % событий ботов.

**5. Как принимались решения.** Целевая метрика крайне шумная (размах 0.61–0.79 по
случайным фолдам и 0.49–0.79 по дням при стабильном ROC-AUC 0.90–0.96). Поэтому
оптимизация шла по PR-AUC, а блок принимался только при одновременном росте PR-AUC,
неухудшении среднего P@R70 по 5 сидам и неухудшении forward-chaining. Этот гейт
отклонил лучший по случайному CV вариант сессии — перцентильные ранги внутри дня.

**6. Модель.** Ансамбль LightGBM + CatBoost + XGBoost, гиперпараметры подобраны Optuna
по PR-AUC, смешивание в ранговом пространстве (метрика порядковая), бэггинг по сидам.

Использованы только open-source библиотеки, всё считается локально, внешних API и
языковых моделей нет. Все сиды зафиксированы, версии — в `requirements.txt`.
""")

code(r"""
import sys, platform, hashlib, json
import numpy as np, pandas as pd

sys.path.insert(0, ".")
from src import config, data, features, plots, report, validate
from src.model import ModelSpec, estimate_rounds, fit_predict, rank_average
from src.pipeline import build_dataset, write_submission

plots.use_style()
pd.set_option("display.width", 160)

print("python", platform.python_version())
for module in ("pandas", "numpy", "sklearn", "lightgbm", "catboost", "xgboost"):
    print(module, __import__(module).__version__)
print("seed", config.SEED)
""")

md(r"""
## 1. Данные и границы окон

Одна строка train/test — одна кука с суточным окном наблюдения. Окна train и test
идут подряд и не пересекаются, поэтому любая валидация обязана быть согласована со
временем.
""")

code(r"""
train, test = data.load_splits()
meta = pd.concat([train.drop(columns=["target"]), test], ignore_index=True)

print(f"train {train.shape}  test {test.shape}")
print(f"доля положительного класса: {train.target.mean():.4f}  ({int(train.target.sum())} из {len(train)})")
print(f"train окна: {train.window_start_ts.min().date()} .. {train.window_start_ts.max().date()}")
print(f"test  окна: {test.window_start_ts.min().date()} .. {test.window_start_ts.max().date()}")
print(f"пересечение cookie_id: {len(set(train.cookie_id) & set(test.cookie_id))}")
print(f"длительность окна, суток: {sorted((meta.window_end_ts - meta.window_start_ts).dt.days.unique())}")

plots.window_timeline(train, test);
""")

md(r"""
## 2. События и фильтр окна

`platform` приходит в разном регистре (`WEB`/`Web`/`web`, `ANDROID`/`Android`/`android`,
`iphone`/`IOS`/`iOS`/`ios`) — приводится к четырём значениям `web`, `desktop`,
`android`, `ios`.

Главное: **12.4 % событий датированы позже конца окна наблюдения**. До начала окна
событий нет вообще. Условие задачи прямо запрещает признаки, недоступные на момент
`window_end_ts`, поэтому такие строки отбрасываются до построения любых признаков.
""")

code(r"""
events_raw = data.load_events()
print(f"всего событий: {len(events_raw)}, кук в событиях: {events_raw.cookie_id.nunique()}")
print("\nтипы событий:")
print(events_raw.event_name.value_counts().to_string())
print("\nплатформы после нормализации:", sorted(events_raw.platform.cat.categories))
print("\nдоля пропусков по полям:")
print((events_raw.isna().mean() * 100).round(1).to_string())
""")

code(r"""
bounds = meta.set_index("cookie_id")
late = events_raw.event_ts > events_raw.cookie_id.map(bounds.window_end_ts)
early = events_raw.event_ts < events_raw.cookie_id.map(bounds.window_start_ts)
print(f"событий позже window_end_ts: {late.mean():.4%}")
print(f"событий раньше window_start_ts: {early.mean():.4%}")

events = data.clip_to_window(events_raw, meta)
data.assert_no_future_leak(events, meta)
print(f"\nпосле фильтра: {len(events)} событий ({len(events) / len(events_raw):.2%}), кук {events.cookie_id.nunique()}")
print("инвариант отсутствия утечки: пройден")
""")

md(r"""
## 3. Как выглядит положительный класс

Портрет складывается из ритма, широты обхода и отсутствия «человеческих» действий.
""")

code(r"""
labelled = events.merge(train[["cookie_id", "target"]], on="cookie_id")
gaps = labelled.groupby("cookie_id").event_ts.diff().dt.total_seconds()

summary = pd.DataFrame({
    "медианный интервал между событиями, с": gaps.groupby(labelled.target).median(),
    "максимальная страница выдачи": labelled.groupby("target").search_page.max(),
    "уникальных локаций на куку": labelled.groupby(["target", "cookie_id"]).item_location.nunique().groupby("target").mean(),
    "доля photo_swipe": labelled.groupby("target").event_name.apply(lambda s: (s == "photo_swipe").mean()),
    "доля favorite_add": labelled.groupby("target").event_name.apply(lambda s: (s == "favorite_add").mean()),
    "доля login": labelled.groupby("target").event_name.apply(lambda s: (s == "login").mean()),
}).T
summary.columns = ["человек (0)", "бот (1)"]
summary.round(3)
""")

code(r"""
web = labelled[labelled.platform.astype(str).isin(["web", "desktop"])]
coverage = web.groupby("target").pointer_x.apply(lambda s: s.notna().mean())
print("доля событий с координатами курсора (только web/desktop):")
print(f"  человек: {coverage[0]:.3f}")
print(f"  бот:     {coverage[1]:.3f}")
print("\nна android и ios координат нет ни у кого — поэтому признак считается отдельно по платформам")
""")

md(r"""
## 4. Метрика и схема валидации

`P@R70` берёт максимум precision среди порогов с recall ≥ 0.70. На выборке с ~900
позитивами это решение одной точки на кривой, и оно очень шумное. Ниже — одна и та же
модель, оценённая по случайным фолдам и по дням: целевая метрика гуляет в пределах
0.15, тогда как ROC-AUC держится в узком коридоре.

Отсюда рабочее правило: **оптимизируем PR-AUC, решение принимаем по связке метрик**,
и ни один блок не принимается по улучшению P@R70 меньше разброса по сидам.
""")

code(r"""
from sklearn.model_selection import StratifiedKFold

dataset = build_dataset()
x, y = dataset.x_train, dataset.y_train
print(f"матрица признаков: {x.shape[0]} кук × {x.shape[1]} признаков")

probe = ModelSpec("lgb", n_rounds=400)
fold_scores = []
for train_idx, valid_idx in StratifiedKFold(5, shuffle=True, random_state=config.SEED).split(x, y):
    pred = fit_predict(probe, x.iloc[train_idx], y.iloc[train_idx], x.iloc[valid_idx])
    fold_scores.append(validate.precision_at_recall(y.iloc[valid_idx], pred))

day_scores = []
days = sorted(dataset.day_index.unique())
for day in days[7:]:
    mask_tr, mask_va = dataset.day_index < day, dataset.day_index == day
    pred = fit_predict(probe, x[mask_tr], y[mask_tr], x[mask_va])
    day_scores.append(validate.precision_at_recall(y[mask_va], pred))

print("P@R70 по случайным фолдам:", np.round(fold_scores, 3))
print("P@R70 по дням forward-chain:", np.round(day_scores, 3))
plots.metric_noise(fold_scores, day_scores);
""")

md(r"""
## 5. Baseline

Первый ориентир — около 50 простых агрегатов: счётчики событий, доли типов, базовые
квантили интервалов, число уникальных категорий/локаций/объявлений, средняя и
максимальная страница выдачи, доли платформ, возраст куки.
""")

code(r"""
BASELINE_PATTERN = (
    r"^(n_events|cnt_|rate_|dt_q|dt_std|dt_min|span_h|events_per_h|item_category_nuniq"
    r"|item_location_nuniq|item_id_nuniq|search_query_nuniq|page_mean|page_max|plat_"
    r"|pointer_coverage|ptr_x_std|ptr_y_std|cookie_age_days)"
)
x_baseline = x.filter(regex=BASELINE_PATTERN)
folds = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y))

baseline_spec = ModelSpec("lgb", n_rounds=estimate_rounds("lgb", x_baseline, y, folds[:3]))
baseline = validate.repeated_cv(baseline_spec, x_baseline, y, n_seeds=3)
print(validate.format_report(f"baseline ({x_baseline.shape[1]} признаков)", baseline))
""")

md(r"""
## 6. Признаки: 15 блоков

| блок | что описывает |
|---|---|
| объём и микс | счётчики и доли 10 типов событий, активные часы, ночная доля |
| переходы | доли топ-25 биграмм, self-loop, энтропия переходов |
| тайминги | квантили интервалов, IQR, MAD, CV, энтропия log-интервала, burstiness, Fano, автокорреляция |
| гранулярность интервалов | число уникальных интервалов, их доля, «круглые» интервалы |
| сессии (gap > 30 мин) | число сессий, событий и длительность на сессию, доля крупнейшей |
| разнообразие контента | nunique и энтропия по категориям, локациям, объявлениям, запросам |
| пагинация | средняя/максимальная страница, длина возрастающих пробегов, страниц на запрос |
| курсор | покрытие, разброс, шаг, скорость |
| геометрия курсора | прямолинейность, углы поворота, квантили скорости, привязка к сетке, квадранты |
| платформа и UA | доли платформ, семейство браузера, headless, редкость UA |
| dwell | время до следующего события по типам |
| навигация | частота смены категории/локации/запроса, длины пробегов, воронка |
| профиль каталога | доли локаций, категорий и запросов |
| структура item_id | разброс, диапазон, монотонность обхода |
| последовательность | TF-IDF n-грамм событий и «событие + корзина интервала» → SVD |
| относительные ранги | перцентиль внутри своей платформы |

Блоки, посчитанные отдельно по платформам (`mob_`, `web_`, `webp_`), — ответ на то,
что платформы заполняют разные поля: интервалы на мобильном и на вебе физически
разные измерения.
""")

code(r"""
BLOCKS = {
    "объём и микс": r"^(n_events|cnt_|rate_|ratio_|n_active_hours|events_per_active_hour|hour_entropy|night_rate)",
    "переходы": r"^(bg_|n_bigrams_uniq|bigram_entropy|self_loop_rate)",
    "тайминги": r"^(dt_|span_h|events_per_h)",
    "гранулярность dt": r"^dtg_",
    "сессии": r"^(sess_|n_sessions)",
    "контент": r"^(item_|search_query_|seller_)",
    "пагинация": r"^(page_|queries_nuniq|pages_per_query|query_repeat_rate|query_len_mean)",
    "курсор": r"^(ptr_|pointer_coverage)",
    "геометрия курсора": r"^ptrd_",
    "платформа и UA": r"^(plat_|n_platforms|platform_entropy|n_user_agents|ua|uafam_|dominant_platform)",
    "dwell": r"^dwell_",
    "навигация": r"^(nav_|funnel_|swipe_run_max)",
    "профиль каталога": r"^(shloc_|shcat_|shq_)",
    "структура item_id": r"^idst_",
    "последовательность": r"^(seqo_|seqp_)",
    "по платформам": r"^(mob_|web_|webp_)",
    "относительные ранги": r"^rel_plat_",
    "мета куки и окна": r"^(cookie_|window_dow|first_event_offset_h|last_event_offset_h|window_coverage)",
}
sizes = pd.Series({name: x.filter(regex=pattern).shape[1] for name, pattern in BLOCKS.items()})
covered = set().union(*(set(x.filter(regex=p).columns) for p in BLOCKS.values()))
print(f"всего признаков: {x.shape[1]}, не отнесены ни к одному блоку: {len(set(x.columns) - covered)}")
sizes.sort_values(ascending=False).to_frame("признаков")
""")

md(r"""
## 7. Вклад блоков

Leave-one-block-out на промежуточном наборе из 178 признаков (3 сида). Результаты
взяты из журнала экспериментов; воспроизводятся командой `python -m src.ablation`.
""")

code(r"""
history = report.load()
ablation_rows = history[history.experiment.str.startswith("ablation_")]
full_score = float(ablation_rows.loc[ablation_rows.experiment == "ablation_full", "P@R70"].iloc[0])
deltas = (
    ablation_rows[ablation_rows.experiment != "ablation_full"]
    .assign(block=lambda f: f.experiment.str.replace("ablation_drop_", "", regex=False))
    .set_index("block")["P@R70"]
    - full_score
)
plots.ablation(deltas, full_score)
deltas.sort_values().round(4).to_frame("Δ P@R70")
""")

md(r"""
Разрыв между курсором и всем остальным — на порядок. При этом координат нет у 72 %
ботов, поэтому одним этим блоком recall 0.70 не набрать: он вычищает верх ранжирования,
а саму полноту обеспечивают мобильные и бескурсорные куки.

## 8. Где модель проигрывает
""")

code(r"""
final_lgb = ModelSpec("lgb", n_rounds=estimate_rounds("lgb", x, y, folds[:3]))
core = validate.repeated_cv(final_lgb, x, y, n_seeds=3)
oof_core = core["oof"]
print(validate.format_report(f"полный набор ({x.shape[1]} признаков)", core))

regime_masks = {
    "web-трафик": x[["plat_web", "plat_desktop"]].sum(axis=1) > 0.5,
    "есть курсор": x["pointer_coverage"].fillna(0) > 0,
    "много событий (>23)": x["n_events"] > 23,
    "нет курсора": x["pointer_coverage"].fillna(0) == 0,
    "mobile-трафик": x[["plat_android", "plat_ios"]].sum(axis=1) > 0.5,
    "мало событий (≤8)": x["n_events"] <= 8,
}
regime_frame = pd.DataFrame([
    {"regime": name, "n": int(mask.sum()), "bots": int(y[mask.to_numpy()].sum()),
     **validate.score(y[mask.to_numpy()], oof_core[mask.to_numpy()])}
    for name, mask in regime_masks.items()
])
plots.regimes(regime_frame)
regime_frame.round(4)
""")

md(r"""
## 9. Что было отвергнуто

| гипотеза | результат |
|---|---|
| граф совместных просмотров объявлений | у `item_id` нет структуры совместности: средняя степень куки у объявления за сутки 1.08, максимум 5 — строить нечего |
| перцентильные ранги внутри окна-дня | лучший результат по случайному CV (0.7953) и падение forward-chaining 0.7705 → 0.7369: ранг внутри дня описывает популяцию дня, а не куку |
| `window_day_index` | окна test целиком позже train (среднее 6.19 против 17.18), любое разбиение по абсолютному индексу дня не переносится |
| отбор топ-200 признаков по gain | PR-AUC 0.7950 против 0.7949 — в пределах шума |
| `captcha_shown` как правило | встречается у 0.2 % ботов и 0 % людей, работать не с чем |
| совпадающие метки времени как признак параллельных запросов | 1.8 % у ботов против 2.2 % у людей, различий нет |
| несоответствие User-Agent и платформы | в данных отсутствует: мобильный UA всегда приходит с мобильной платформы |

## 10. Финальная модель

Три градиентных бустинга с подобранными Optuna параметрами, смешивание в ранговом
пространстве, бэггинг по сидам. Веса подбираются по OOF на PR-AUC.
""")

code(r"""
from src.submit import load_spec, search_weights

KINDS = ["lgb", "cat", "xgb"]
specs = {kind: load_spec(kind, x, y, folds) for kind in KINDS}
oof = {}
for kind, spec in specs.items():
    res = validate.repeated_cv(spec, x, y, n_seeds=config.N_SEEDS)
    oof[kind] = res["oof"]
    print(validate.format_report(f"{kind} (rounds={spec.n_rounds})", res))

weights = search_weights(oof, y)
oof_blend = rank_average([oof[k] for k in KINDS], [weights[k] for k in KINDS])
blend_score = validate.score(y, oof_blend)
print("\nвеса бленда:", weights)
print("бленд OOF:", {k: round(v, 4) for k, v in blend_score.items()})
""")

code(r"""
plots.pr_curve(y, oof_blend);
""")

code(r"""
from src.model import feature_gain

gains = sum(
    feature_gain(specs["lgb"], x, y, seed=config.SEED + offset) for offset in range(3)
).sort_values(ascending=False)
plots.importances(gains, top=20);
""")

md(r"""
## 11. История экспериментов
""")

code(r"""
TRACK = {
    "B0_baseline_simple": "baseline, 47 признаков",
    "ablation_full": "ядро A, 178",
    "A3_pointer_deep_lgb": "+ геометрия курсора, 353",
    "A4_relative_scoped": "+ ранги дня (отвергнут), 457",
    "A5_core": "рабочий набор, 423",
    "final_blend": "бленд трёх моделей",
}
timeline = report.table(list(TRACK)).assign(шаг=lambda f: f.experiment.map(TRACK))
plots.progression(timeline.dropna(subset=["P@R70"]))
timeline
""")

md(r"""
## 12. Предсказание и файл ответа

Модели переобучаются на полном train, предсказания по каждому семейству усредняются
по сидам в ранговом пространстве, затем смешиваются с найденными весами. Итоговый
score линейно растягивается в [0, 1] — метрика читает только порядок.
""")

code(r"""
BAG = 10  # усреднение по сидам только для боевых предсказаний, валидация идёт на 5
test_predictions = []
for kind, spec in specs.items():
    bagged = [
        fit_predict(spec, x, y, dataset.x_test, seed=config.SEED + offset)
        for offset in range(BAG)
    ]
    test_predictions.append(rank_average(bagged))

scores = rank_average(test_predictions, [weights[k] for k in KINDS])
submission = write_submission(dataset.test_ids, scores)

assert len(submission) == len(test), "число строк не совпадает с test.csv"
assert submission.cookie_id.is_unique, "дубликаты cookie_id"
assert set(submission.cookie_id) == set(test.cookie_id), "набор cookie_id не совпадает"
assert submission.score.between(0, 1).all(), "score вне [0, 1]"
assert submission.score.notna().all(), "пропуски в score"

print(submission.head().to_string(index=False))
print(f"\nстрок: {len(submission)}")
print(f"md5 submission.csv: {hashlib.md5(config.SUBMISSION_PATH.read_bytes()).hexdigest()}")
""")

md(r"""
## 13. Ограничения

- **Метрика шумная.** Разброс P@R70 по сидам ±0.01, по дням до ±0.15. Разница между
  двумя моделями меньше 0.01 ничего не значит, и оценка на скрытом тесте может
  отличаться от OOF на сопоставимую величину.
- **Курсор несёт непропорционально много.** Если источник заполнения `pointer_x`
  изменится (другой сборщик телеметрии, другая версия клиента), качество просядет
  сильнее, чем можно было бы ожидать от одного поля.
- **Мобильный трафик решается плохо** (P@R70 около 0.36 внутри подгруппы против 0.87
  на вебе). Там нет курсора, и оставшиеся сигналы слабее.
- **Короткие сессии** (≤ 8 событий) почти не отличимы: на них приходится треть кук.
- **Обучение на одной неделе.** Сезонность длиннее недели в данных не представлена;
  устойчивость к смене поведения сервисов сбора данных не проверялась.
- **Положительный класс — это известные сервисы.** Модель учится отличать именно их,
  а не автоматизацию вообще; новый сборщик с другим профилем может не ловиться.
""")

notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
    for kind, source in CELLS
]
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
path = ROOT / "solution.ipynb"
nbf.write(notebook, path)
print(f"wrote {path} ({len(notebook.cells)} cells)")
