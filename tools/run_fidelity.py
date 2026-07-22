"""Score simulation fidelity across a set of illumination configurations.

Runs the simulator over every configured scene and reports how closely the
synthetic PV traces match the real measurements, using the three metrics from
the paper (Sec. 4.2): Pearson correlation for temporal shape, RMSE in
microwatts for absolute error, and range-normalised RMSE so configurations
whose peak power differs by two orders of magnitude stay comparable.

Calibration is refit per recording. Pearson correlation is invariant to a
scalar, so for single-emitter scenes this changes nothing and isolates the
question to trace shape; for multi-emitter scenes it gives each recording the
best-case mixture, which is the fair reading of a geometry check.

Requires the recordings, which are not distributed with this repository. Run
from the repository root::

    python tools/run_fidelity.py
    python tools/run_fidelity.py --data-root /path/to/data --report results/fidelity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pysolarsim import RecordingSpec, Scene, results_frame, simulate_batch  # noqa: E402
from pysolarsim.pipeline import discover_recordings  # noqa: E402

#: Scene stems to evaluate. Each data directory is named identically to its
#: scene file, so no separate path mapping is needed.
CONFIGURATIONS = [
    "intensity_circular_6000K_40klx",
    "intensity_circular_6000K_30klx",
    "intensity_circular_6000K_15klx",
    "intensity_circular_6000K_0p4klx",
    "cct_circular_4000K_40klx",
    "cct_circular_2400K_40klx",
    "geometry_rectangular_6000K_40klx",
    "multi_dual_circular_d50cm",
    "multi_dual_circular_d110cm",
    "multi_circular_rectangular_d70cm",
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="directory holding the recordings")
    parser.add_argument("--scene-dir", default="configs/scene", help="directory of scene files")
    parser.add_argument("--mocap", default="configs/mocap/leap_csv.yaml")
    parser.add_argument("--report", help="write the per-recording table to this CSV")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    scene_dir = Path(args.scene_dir)
    spec = RecordingSpec.load(args.mocap)

    print(f"{'configuration':<38} {'rho':>7} {'RMSE uW':>9} {'nRMSE':>7} {'files':>6}")
    print("-" * 71)

    tables = []
    for stem in CONFIGURATIONS:
        scene_path = scene_dir / f"{stem}.yaml"
        recordings_dir = data_root / stem
        if not scene_path.is_file():
            print(f"{stem:<38} {'no scene':>7}")
            continue
        if not recordings_dir.is_dir():
            print(f"{stem:<38} {'no data':>7}")
            continue

        recordings = discover_recordings(recordings_dir, pattern="*.csv", recursive=False)
        if not recordings:
            print(f"{stem:<38} {'no files':>7}")
            continue

        results = simulate_batch(
            recordings, Scene.load(scene_path), spec, output_dir=None,
            calibration_mode="per_recording", on_error="warn", progress=False,
        )
        if not results:
            print(f"{stem:<38} {'failed':>7}")
            continue

        frame = results_frame(results)
        frame.insert(0, "configuration", stem)
        tables.append(frame)

        def mean_of(column: str) -> float:
            values = frame[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else float("nan")

        print(
            f"{stem:<38} {mean_of('pearson'):>7.3f} {mean_of('rmse'):>9.2f} "
            f"{mean_of('nrmse'):>7.3f} {len(results):>6}"
        )

    if not tables:
        print("\nNo configurations evaluated. Check --data-root.")
        return 1

    combined = pd.concat(tables, ignore_index=True)
    rho = combined["pearson"].to_numpy(dtype=float)
    rho = rho[np.isfinite(rho)]
    print("-" * 71)
    print(f"{'overall':<38} {rho.mean():>7.3f} {'':>9} {'':>7} {len(combined):>6}")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"\nPer-recording report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
