"""Command-line interface.

Five subcommands cover the whole workflow::

    pysolarsim describe   --scene S                 inspect a scene file
    pysolarsim calibrate  --scene S --mocap M ...   fit kappa for an environment
    pysolarsim simulate   --scene S --mocap M ...   synthesise power traces
    pysolarsim validate   --scene S --mocap M ...   score fidelity vs. real PV
    pysolarsim convert-scene --layout L --out O     migrate a recorded layout note

``simulate`` and ``validate`` accept either a single recording or a directory,
so the same code path serves a quick one-off check and a full dataset sweep.
Any flag may instead be supplied through ``--config run.yaml``; explicit flags
win over the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .convert import scene_from_layout_file
from .io.recording import RecordingSpec
from .pipeline import (
    calibrate_scene,
    discover_recordings,
    results_frame,
    simulate_batch,
)
from .scene import Scene


def _load_run_config(path: Optional[str]) -> Dict[str, Any]:
    """Read an optional run config, which supplies defaults for every flag."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"run config not found: {p}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: run config must contain a mapping at the top level")
    return data


def _setting(args: argparse.Namespace, config: Dict[str, Any], key: str, default=None):
    """Resolve one setting: explicit flag, then run config, then default."""
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config.get(key, default)


def _resolve_inputs(args: argparse.Namespace, config: Dict[str, Any]) -> tuple:
    """Resolve the scene, recording spec, and the list of input recordings."""
    scene_path = _setting(args, config, "scene")
    mocap_path = _setting(args, config, "mocap")
    if not scene_path:
        raise ValueError("no scene given; pass --scene or set 'scene:' in the run config")
    if not mocap_path:
        raise ValueError("no mocap config given; pass --mocap or set 'mocap:' in the run config")

    scene = Scene.load(scene_path)
    spec = RecordingSpec.load(mocap_path)

    input_path = _setting(args, config, "input")
    if not input_path:
        raise ValueError("no input given; pass --input or set 'input:' in the run config")

    pattern = _setting(args, config, "pattern", "*.csv")
    recordings = discover_recordings(input_path, pattern=pattern)
    if not recordings:
        raise ValueError(f"no recordings matching {pattern!r} found under {input_path}")
    return scene, spec, recordings, Path(input_path)


def _resolve_calibration(
    args: argparse.Namespace,
    config: Dict[str, Any],
    scene: Scene,
    spec: RecordingSpec,
    recordings: List[Path],
) -> tuple:
    """Work out the calibration mode and, when needed, fit kappa.

    Returns:
        ``(mode, method, kappas, calibration_result)``; ``kappas`` is ``None``
        unless the mode is ``reference``.
    """
    block = {**(scene.calibration or {}), **(config.get("calibration") or {})}
    mode = str(_setting(args, config, "calibration_mode") or block.get("mode", "reference")).lower()
    method = _setting(args, config, "calibration_method") or block.get("method")

    if mode != "reference":
        return mode, method, None, None

    references = getattr(args, "reference", None) or block.get("references")
    if isinstance(references, (str, Path)):
        references = [references]
    if not references:
        # Fitting on the whole set is the honest fallback: it still yields one
        # kappa per environment, just from more data than the paper's protocol.
        references = recordings
        print(
            f"No calibration reference given; fitting kappa across all "
            f"{len(recordings)} recording(s)."
        )

    max_seconds = block.get("reference", {}).get("max_seconds") if isinstance(
        block.get("reference"), dict
    ) else None
    max_seconds = _setting(args, config, "max_seconds") or max_seconds

    fit = calibrate_scene(
        scene, references, spec, max_seconds=max_seconds, method=method
    )
    print(
        f"Calibrated kappa ({fit.method}, {fit.n_frames} frames, "
        f"residual {fit.residual:.2f} uW):"
    )
    for eid, k in zip(fit.emitter_ids, fit.kappas):
        print(f"  {eid}: {k:.6g}")
    return mode, method, fit.kappas, fit


def _report(results, report_path: Optional[str]) -> None:
    """Print aggregate fidelity and optionally write the per-recording table."""
    if not results:
        print("No recordings simulated.")
        return

    frame = results_frame(results)
    if "pearson" in frame.columns:
        rho = frame["pearson"].to_numpy(dtype=float)
        rho = rho[np.isfinite(rho)]
        if rho.size:
            print(
                f"\nFidelity over {rho.size} recording(s): "
                f"rho mean={rho.mean():.3f}, median={np.median(rho):.3f}, "
                f"min={rho.min():.3f}, max={rho.max():.3f}"
            )
        for column, label in (("rmse", "RMSE (uW)"), ("nrmse", "nRMSE")):
            if column in frame.columns:
                values = frame[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    print(f"  {label}: mean={values.mean():.4f}")

    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        print(f"\nPer-recording report written to {out}")


def cmd_describe(args: argparse.Namespace) -> int:
    """Print a scene file's contents in human-readable form."""
    print(Scene.load(args.scene).describe())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit kappa for an environment and optionally bake it into a scene file."""
    config = _load_run_config(args.config)
    scene, spec, recordings, _ = _resolve_inputs(args, config)

    references = args.reference or recordings
    fit = calibrate_scene(
        scene,
        references,
        spec,
        max_seconds=args.max_seconds,
        method=args.calibration_method,
    )
    print(
        f"Fitted {fit.method} over {fit.n_frames} frames "
        f"(residual {fit.residual:.3f} uW)"
    )
    for eid, k in zip(fit.emitter_ids, fit.kappas):
        print(f"  {eid}: {k:.6g}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fit.as_dict(), indent=2), encoding="utf-8")
        print(f"Calibration written to {out}")

    if args.update_scene:
        by_id = dict(zip(fit.emitter_ids, fit.kappas))
        for emitter in scene.emitters:
            if emitter.id in by_id:
                emitter.kappa = float(by_id[emitter.id])
        scene.calibration = {**scene.calibration, "mode": "fixed"}
        path = scene.save(args.update_scene)
        print(f"Scene with baked-in kappa written to {path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Synthesise power traces for one recording or a whole directory."""
    config = _load_run_config(args.config)
    scene, spec, recordings, input_root = _resolve_inputs(args, config)
    print(scene.describe())
    print(f"Found {len(recordings)} recording(s) under {input_root}")

    mode, method, kappas, _ = _resolve_calibration(args, config, scene, spec, recordings)

    output_dir = _setting(args, config, "output")
    results = simulate_batch(
        recordings,
        scene,
        spec,
        output_dir=output_dir,
        input_root=input_root if input_root.is_dir() else input_root.parent,
        kappas=kappas,
        calibration_mode=mode,
        calibration_method=method,
        power_column=_setting(args, config, "power_column", "P_sim_uW"),
        on_error="raise" if args.strict else "warn",
    )
    if output_dir:
        print(f"\nWrote {len(results)} simulated recording(s) to {output_dir}")
    _report(results, _setting(args, config, "report"))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Score simulated traces against the measured PV in the same recordings."""
    config = _load_run_config(args.config)
    scene, spec, recordings, input_root = _resolve_inputs(args, config)
    if not spec.power.enabled:
        raise ValueError(
            "validation needs measured power; add a 'power:' block to the mocap config"
        )
    print(scene.describe())
    print(f"Validating against {len(recordings)} recording(s)")

    mode, method, kappas, _ = _resolve_calibration(args, config, scene, spec, recordings)
    results = simulate_batch(
        recordings,
        scene,
        spec,
        output_dir=None,
        input_root=input_root if input_root.is_dir() else input_root.parent,
        kappas=kappas,
        calibration_mode=mode,
        calibration_method=method,
        on_error="raise" if args.strict else "warn",
    )
    _report(results, _setting(args, config, "report") or args.report)
    return 0


def cmd_convert_scene(args: argparse.Namespace) -> int:
    """Migrate a recorded ``Light Sources Positions.txt`` into a scene file."""
    scene = scene_from_layout_file(
        args.layout,
        name=args.name or "",
        rect_width_mm=args.rect_width,
        rect_height_mm=args.rect_height,
    )
    print(scene.describe())
    path = scene.save(args.out)
    print(f"Scene written to {path}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the subcommands that run the pipeline."""
    parser.add_argument("--config", help="run config supplying defaults for these flags")
    parser.add_argument("--scene", help="scene file describing the illumination environment")
    parser.add_argument("--mocap", help="mocap config describing the input schema")
    parser.add_argument("--input", help="recording file or directory of recordings")
    parser.add_argument("--pattern", help="glob matched against file names (default *.csv)")
    parser.add_argument(
        "--calibration-mode",
        dest="calibration_mode",
        choices=["reference", "per_recording", "fixed"],
        help="where kappa comes from (default: the scene's setting, else reference)",
    )
    parser.add_argument(
        "--calibration-method",
        dest="calibration_method",
        choices=["auto", "scalar_ls", "nnls"],
        help="kappa fitting method (default: auto)",
    )
    parser.add_argument(
        "--reference",
        action="append",
        help="paired recording to calibrate from; repeatable",
    )
    parser.add_argument(
        "--max-seconds",
        dest="max_seconds",
        type=float,
        help="trim each calibration reference to its first N seconds",
    )
    parser.add_argument("--report", help="write a per-recording metrics CSV here")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop on the first unreadable recording instead of skipping it",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="pysolarsim",
        description="Physics-based simulation of wearable solar harvester signals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_describe = sub.add_parser("describe", help="print a scene file")
    p_describe.add_argument("--scene", required=True)
    p_describe.set_defaults(func=cmd_describe)

    p_cal = sub.add_parser("calibrate", help="fit kappa for an environment")
    _add_common(p_cal)
    p_cal.add_argument("--out", help="write the fitted calibration as JSON")
    p_cal.add_argument(
        "--update-scene",
        dest="update_scene",
        help="write a copy of the scene with kappa baked in",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_sim = sub.add_parser("simulate", help="synthesise PV power traces")
    _add_common(p_sim)
    p_sim.add_argument("--output", help="directory for simulated recordings")
    p_sim.add_argument(
        "--power-column",
        dest="power_column",
        help="name of the simulated power column (default P_sim_uW)",
    )
    p_sim.set_defaults(func=cmd_simulate)

    p_val = sub.add_parser("validate", help="score fidelity against measured PV")
    _add_common(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_conv = sub.add_parser(
        "convert-scene", help="migrate a Light Sources Positions.txt into a scene file"
    )
    p_conv.add_argument("--layout", required=True, help="the recorded layout note")
    p_conv.add_argument("--out", required=True, help="scene file to write")
    p_conv.add_argument("--name", help="scene name (default: parent directory name)")
    p_conv.add_argument(
        "--rect-width", dest="rect_width", type=float,
        help="rectangle width in mm, when the layout omits Rec_width",
    )
    p_conv.add_argument(
        "--rect-height", dest="rect_height", type=float,
        help="rectangle height in mm, when the layout omits Rec_height",
    )
    p_conv.set_defaults(func=cmd_convert_scene)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
