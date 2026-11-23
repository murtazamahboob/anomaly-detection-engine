# Anomaly Detection Engine — Learn What Normal Looks Like, Flag What Is Not

**Find rare, unusual behaviour in logs, sensor data, prices and images without needing labelled examples — fully offline, pure NumPy, deterministic.** Most anomaly demos ship one detector and a cherry-picked chart. This one ships three — an Isolation Forest, an MLP autoencoder and an LSTM sequence autoencoder — behind one `fit / score / explain` interface, plus the controls that decide whether an anomaly system is usable in practice: percentile thresholds from normal validation data, a persistence rule, alert grouping, reviewer feedback, and an explanation for every alert.

## Results (measured, not claimed)

Benchmark: 800 normal training rows, 600 test rows with 4 injected anomaly segments (spike, level shift, drift, dropout — 80 anomalous rows). Thresholds are the 99.5th percentile of normal validation scores. Run it yourself with `anomalydetect benchmark`.

| Method | Precision | Recall | F1 | PR-AUC | Pt-adj recall | False alarms | Delay |
|---|---|---|---|---|---|---|---|
| forest | 1.000 | 0.100 | 0.182 | 0.672 | 0.750 | 0.000 | 0.7 |
| autoencoder | 1.000 | 0.338 | 0.505 | 0.780 | 0.750 | 0.000 | 0.0 |
| ensemble | 1.000 | 0.113 | 0.202 | 0.793 | 0.750 | 0.000 | 0.7 |
| lstm | 1.000 | 0.714 | 0.833 | 0.959 | 1.000 | 0.000 | 0.8 |
| forest_windows | 1.000 | 0.571 | 0.727 | 0.781 | 0.750 | 0.000 | 0.0 |
| autoencoder_windows | 1.000 | 0.786 | 0.880 | 0.897 | 1.000 | 0.000 | 0.8 |
| ensemble_windows | 1.000 | 0.571 | 0.727 | 0.891 | 0.750 | 0.000 | 0.0 |

The honest finding is in the table, not hidden from it: **row-by-row detectors catch spikes and dropouts but miss level shifts and drift**, because a +6° shift still sits inside that channel's normal range — it is only anomalous as a *change*. Windowed features and the LSTM see the change: the windowed autoencoder reaches F1 0.880 and the LSTM PR-AUC 0.959, both catching all 4 segments. Every method holds precision 1.000 with zero false alarms at the 99.5th-percentile threshold. Strict recall and point-adjusted recall are both reported — point-adjusted numbers are always higher, so a benchmark that only shows them is flattering itself.

| Metric | Measured |
|---|---|
| Test suite | **22/22 tests pass**, verified twice — including in a fresh venv |
| Demo | ensemble flags 6/600 rows, all inside the injected dropout segment; top explanation correctly blames `current_a` (0.40) |
| Runtime dependencies | **NumPy only** — no scikit-learn, no PyTorch, no GPU, no network |

## Features

- **Three detectors, one interface** — `AnomalyEngine(detector="forest" | "autoencoder" | "lstm" | "ensemble")` with `fit / score / predict / explain`. The Isolation Forest (300 trees, `max_samples=256`, classic `2^(-E[h]/c(n))` score) and the MLP autoencoder (d→32→16→8→16→32→d, Adam, early stopping) are written from scratch in NumPy; the LSTM sequence autoencoder is a real LSTM encoder (BPTT) with a dense decoder that rebuilds whole windows.
- **Ensemble by rank, not by raw score** — forest scores and reconstruction errors live on different scales, so each is replaced by its percentile rank on normal validation data and averaged. A majority vote (`votes()`) reports how many members flag each point.
- **Thresholds from normal validation only** — 99.5th percentile by default (99th–99.5th per the blueprint). A threshold fitted on scored data would move with the anomalies and hide them.
- **Explanations for every alert** — per-feature reconstruction error from the autoencoder plus per-feature split credit from the forest, combined by rank-average and normalised to "share of blame". The demo's top anomalies correctly blame `current_a` for the injected dropout.
- **Persistence + alert grouping** — an alert fires only after 2 consecutive anomalous windows, and alerts within 10 minutes merge into one incident, so one fault does not page anyone five times.
- **Reviewer feedback loop** — mark alerts real or false and `retune_threshold()` moves the cutoff: up to silence the worst false alarm seen, down to catch a missed real anomaly. With no reviewed labels the threshold never moves.
- **Adapters for logs, sensors and prices** — log lines → requests/error-rate/latency features, price ticks → spread/spread-over-ATR/tick-rate/return, sensor rows → validated channels (a NaN reading is a dropout, never an invented number).
- **Plug-in ready** — the same engine is the shared module the blueprint's restaurant monitor (#19) and gold trading agent (#24) build on: clean-table crops, abnormal spreads and volatility all score through `fit / score / explain`.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Bundled demo: train on normal sensor data, score a stream with 4 injected anomalies
anomalydetect demo

# Full benchmark: every detector, strict + point-adjusted metrics
anomalydetect benchmark

# Score your own CSV of feature rows (one row per line)
anomalydetect score data.csv --detector ensemble --top 5

# Or run the demo script directly
python examples/demo.py
```

## Python API

```python
from anomalydetect import AnomalyEngine
from anomalydetect.data import make_dataset, FEATURE_NAMES

data = make_dataset()  # 800 normal train rows, 600 test rows, 4 injected segments
engine = AnomalyEngine(detector="ensemble").fit(data["train"], feature_names=FEATURE_NAMES)

scores = engine.score(data["test"])          # higher = more anomalous
flags = engine.predict(data["test"])         # after the persistence rule
print(engine.top_features(data["test"][:3])) # [('current_a', 0.40), ('humidity', 0.17), ...]

# Sequence view: the LSTM scores 20-step windows instead of single rows
lstm = AnomalyEngine(detector="lstm").fit(data["train"])
print(lstm.score(data["test"])[:3])
```

## How it works

1. **Collect normal data** — a period of known-normal operation (bundled: 6-channel sensor stream, correlated noise on slow sinusoids).
2. **Window and clean** — time series are cut into 20-step windows (stride 10); features are mean/std/min/max/range/slope per channel. Scaling is fitted on training data only.
3. **Train** — forest on scaled rows, autoencoder to minimise reconstruction error on normal rows, LSTM autoencoder to rebuild normal windows.
4. **Score and threshold** — each detector scores new data; the threshold is the 99.5th percentile of its scores on held-out normal validation data.
5. **Explain and alert** — per-feature blame, persistence (2 consecutive windows), grouping within 10 minutes into one incident.
6. **Feedback and retrain** — reviewer labels move the threshold; normal drifts over time, so retrain monthly or when the false-alarm rate rises.

## Project structure

```
anomaly-detection-engine/
├── src/anomalydetect/
│   ├── __init__.py        # public API
│   ├── engine.py          # one fit/score/explain/predict interface
│   ├── forest.py          # Isolation Forest from scratch
│   ├── autoencoder.py     # MLP autoencoder from scratch (Adam, early stopping)
│   ├── lstm_ae.py         # LSTM encoder + dense decoder (BPTT)
│   ├── ensemble.py        # rank-averaged forest + autoencoder
│   ├── scaling.py         # scaler fitted on training data only
│   ├── features.py        # windowing + per-window aggregates
│   ├── threshold.py       # percentile / mean+std thresholds
│   ├── explain.py         # per-feature blame (AE error + forest split credit)
│   ├── alerts.py          # persistence rule + 10-minute incident grouping
│   ├── feedback.py        # reviewer-driven threshold retuning
│   ├── adapters.py        # logs / sensors / prices -> feature rows
│   ├── data.py            # deterministic synthetic data + injected anomalies
│   ├── benchmark.py       # precision/recall/F1/PR-AUC/delay, strict + point-adjusted
│   └── cli.py             # anomalydetect demo | benchmark | score
├── examples/demo.py       # one-command demo (writes outputs/)
├── tests/                 # 22 tests: detectors, thresholds, alerts, feedback, adapters
└── outputs/               # demo report + benchmark.json (regenerated by the demo)
```

## Limits (read before trusting it)

- Synthetic data only: the bundled benchmark uses generated sensor streams with injected anomalies, not NAB or MVTec AD — the numbers prove the pipeline works, not that it beats published results on public data.
- The row-wise ensemble is the weakest combination in the table (F1 0.202): rank-averaging dilutes a single strong signal. Use it with window features (`ensemble_windows`, F1 0.727) or the LSTM for anything with temporal structure.
- The LSTM decoder is a dense head on the encoder's final hidden state, not a recurrent decoder — documented here and in the code, so the architecture is what the README says it is.
- A slow drift that stays inside normal ranges for a long time will always be the hardest case (drift recall is the lowest of the four kinds); that is what the monthly retrain schedule and the feedback loop are for.
- Never a trading signal on its own: for the gold agent it is a pause-during-abnormal-conditions monitor, not an entry rule.
