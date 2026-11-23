"""CLI: ``anomalydetect demo | benchmark | score``.

* ``demo`` — build the synthetic dataset, train the ensemble, score the
  injected test set and write ``outputs/report.md`` + ``outputs/benchmark.json``.
* ``benchmark`` — print the method-comparison table for every detector.
* ``score`` — score a CSV of feature rows with a chosen detector and print
  the top anomalies with their explanations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .benchmark import method_table, run_benchmark
from .data import FEATURE_NAMES, make_dataset
from .engine import AnomalyEngine


def _demo(output_dir: Path) -> dict:
    data = make_dataset()
    engine = AnomalyEngine(detector="ensemble", seed=42)
    engine.fit(data["train"], feature_names=FEATURE_NAMES)
    scores = engine.score(data["test"])
    flags = engine.predict(data["test"])
    top = np.argsort(-scores)[:5]
    explanations = engine.top_features(data["test"][top])
    report = {
        "threshold": engine.threshold_,
        "n_test": len(data["test"]),
        "n_flagged": int(flags.sum()),
        "top_anomalies": [
            {"row": int(i), "score": float(scores[i]),
             "top_features": explanations[j]}
            for j, i in enumerate(top)
        ],
        "segments": data["segments"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Anomaly Detection — demo report",
             f"\nThreshold (99.5th pct of normal validation): "
             f"{engine.threshold_:.4f}",
             f"\nFlagged {int(flags.sum())} of {len(data['test'])} test rows.",
             "\n## Top 5 anomalies\n",
             "| Row | Score | Top features |",
             "|---|---|---|"]
    for item in report["top_anomalies"]:
        feats = ", ".join(f"{n} ({s:.2f})" for n, s in item["top_features"])
        lines.append(f"| {item['row']} | {item['score']:.3f} | {feats} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anomalydetect")
    sub = parser.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser("demo", help="run the bundled demo")
    demo_p.add_argument("--output-dir", default="outputs")

    bench_p = sub.add_parser("benchmark", help="run the full benchmark")
    bench_p.add_argument("--output", default=None)

    score_p = sub.add_parser("score", help="score a CSV of feature rows")
    score_p.add_argument("csv", help="CSV file, one feature row per line")
    score_p.add_argument("--detector", default="ensemble",
                         choices=["forest", "autoencoder", "lstm", "ensemble"])
    score_p.add_argument("--top", type=int, default=5)

    args = parser.parse_args(argv)

    if args.command == "demo":
        report = _demo(Path(args.output_dir))
        print(f"threshold={report['threshold']:.4f} "
              f"flagged={report['n_flagged']}/{report['n_test']}")
        for item in report["top_anomalies"]:
            feats = ", ".join(f"{n} ({s:.2f})"
                              for n, s in item["top_features"])
            print(f"row {item['row']}: score {item['score']:.3f} — {feats}")
        return 0

    if args.command == "benchmark":
        results = run_benchmark()
        table = method_table(results)
        print(table)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
        return 0

    if args.command == "score":
        rows = np.loadtxt(args.csv, delimiter=",")
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        data = make_dataset()
        engine = AnomalyEngine(detector=args.detector, seed=42)
        engine.fit(data["train"],
                   feature_names=FEATURE_NAMES[:rows.shape[1]])
        scores = engine.score(rows)
        order = np.argsort(-scores)[:args.top]
        for i in order:
            print(f"row {i}: score {scores[i]:.4f} "
                  f"(threshold {engine.threshold_:.4f})")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
