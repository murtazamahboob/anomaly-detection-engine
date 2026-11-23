"""Run the bundled demo: train the ensemble on normal sensor data, score a
test stream with four injected anomalies, and print the top anomalies with
their explanations. Writes outputs/report.md and outputs/report.json.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomalydetect.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--output-dir",
                           str(Path(__file__).resolve().parents[1] / "outputs")]))
