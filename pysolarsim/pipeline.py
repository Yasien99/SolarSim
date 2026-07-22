"""End-to-end orchestration: motion capture + illumination -> PV power.

This is the layer the CLI and any notebook should call.  It ties the three
stages of the paper's Fig. 1 pipeline together -- read the trajectory, build
per-emitter view factors, scale by calibrated kappa and sum -- and adds the
bookkeeping that every experiment needs: where kappa comes from, how a batch
of recordings is discovered, and how results are written back out.

Calibration modes (paper Sec. 3.3 prescribes ``reference``):

``reference``
    Fit kappa once from a short paired recording and reuse it for every other
    recording in that environment.  Isolates fidelity to the geometric model
    rather than to per-recording calibration.
``per_recording``
    Refit kappa on each recording.  Useful for fidelity checks where the
    question is purely about trace shape.
``fixed``
    Take kappa straight from the scene file; no paired power needed.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as metrics_mod
from .calibrate import CalibrationResult, fit_kappa
from .engine import power_from_view_factors, view_factor_matrix
from .io.recording import Recording, RecordingSpec, read_recording
from .motion import MotionTrack
from .scene import Scene

#: File extensions treated as recordings during batch discovery.
RECORDING_SUFFIXES = (".csv", ".parquet")


@dataclass
class SimulationResult:
    """One simulated recording and everything needed to audit it."""

    name: str
    path: Path
    track: MotionTrack
    view_factors: np.ndarray
    kappas: np.ndarray
    power_uW: np.ndarray
    measured_uW: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    frame: Optional[pd.DataFrame] = None
    emitter_ids: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Flat, JSON-serialisable row for aggregate reports."""
        row: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "frames": len(self.track),
            "valid_frames": self.track.n_valid,
            "duration_s": round(self.track.duration_s, 3),
        }
        for eid, k in zip(self.emitter_ids, self.kappas):
            row[f"kappa[{eid}]"] = float(k)
        row.update({k: float(v) for k, v in self.metrics.items()})
        return row


def _score(result_power: np.ndarray, measured: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """Fidelity metrics computed over valid frames only."""
    if measured is None:
        return {}
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return {}
    return metrics_mod.all_metrics(measured[idx], result_power[idx])


def discover_recordings(
    root: str | Path,
    pattern: str = "*.csv",
    recursive: bool = True,
) -> List[Path]:
    """Find recordings under ``root``, sorted for reproducible ordering.

    Args:
        root: a directory to search, or a single file to return as-is.
        pattern: glob matched against the file name.
        recursive: descend into participant subdirectories.

    Returns:
        Sorted list of matching paths.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"input path not found: {root}")

    walker = root.rglob("*") if recursive else root.glob("*")
    found = [
        p
        for p in walker
        if p.is_file()
        and p.suffix.lower() in RECORDING_SUFFIXES
        and fnmatch.fnmatch(p.name, pattern)
    ]
    return sorted(found)


def calibrate_scene(
    scene: Scene,
    references: Sequence[str | Path],
    spec: RecordingSpec,
    max_seconds: Optional[float] = None,
    method: Optional[str] = None,
) -> CalibrationResult:
    """Fit kappa once for an environment from paired motion-PV recordings.

    Several reference recordings are stacked into a single least-squares
    problem, so a short excerpt from each can be combined.

    Args:
        scene: the illumination environment being calibrated.
        references: paired recordings containing both motion and measured PV.
        spec: how to read those recordings; its power spec must be enabled.
        max_seconds: trim each reference to its first N seconds.
        method: override the scene's calibration method.

    Returns:
        The fitted :class:`~pysolarsim.calibrate.CalibrationResult`.

    Raises:
        ValueError: if no reference is given or none carries measured power.
    """
    if not references:
        raise ValueError("calibration needs at least one reference recording")
    if not spec.power.enabled:
        raise ValueError(
            "calibration needs measured power; add a 'power:' block to the "
            "mocap config or switch the scene to calibration mode 'fixed'"
        )

    F_blocks: List[np.ndarray] = []
    P_blocks: List[np.ndarray] = []
    for ref in references:
        rec = read_recording(ref, spec)
        if rec.power_uW is None:
            raise ValueError(f"{Path(ref).name}: no measured power to calibrate against")
        track = rec.track.head(max_seconds) if max_seconds else rec.track
        n = len(track)
        F = view_factor_matrix(scene, track)
        idx = np.flatnonzero(track.valid)
        if idx.size == 0:
            continue
        F_blocks.append(F[idx])
        P_blocks.append(rec.power_uW[:n][idx])

    if not F_blocks:
        raise ValueError("no valid paired frames found across the reference recordings")

    return fit_kappa(
        np.vstack(F_blocks),
        np.concatenate(P_blocks),
        method=method or scene.calibration_method,
        emitter_ids=[e.id for e in scene.active],
    )


def simulate_recording(
    path: str | Path,
    scene: Scene,
    spec: RecordingSpec,
    kappas: Optional[np.ndarray] = None,
    calibration_mode: str = "reference",
    calibration_method: Optional[str] = None,
) -> SimulationResult:
    """Synthesise a PV power trace for one recording.

    Args:
        path: the recording to simulate.
        scene: the illumination environment.
        spec: how to read the recording.
        kappas: pre-fitted calibration scalars, required for ``reference`` mode.
        calibration_mode: ``reference``, ``per_recording``, or ``fixed``.
        calibration_method: override the scene's kappa-fitting method.

    Returns:
        The :class:`SimulationResult`, scored against measured power when the
        recording carries it.

    Raises:
        ValueError: if the chosen mode has no kappa available.
    """
    active = scene.active
    rec = read_recording(path, spec)
    F = view_factor_matrix(scene, rec.track)

    mode = str(calibration_mode).lower()
    if mode == "per_recording":
        if rec.power_uW is None:
            raise ValueError(
                f"{rec.path.name}: calibration mode 'per_recording' needs measured "
                "power in the recording"
            )
        idx = np.flatnonzero(rec.track.valid)
        fitted = fit_kappa(
            F[idx],
            rec.power_uW[idx],
            method=calibration_method or scene.calibration_method,
            emitter_ids=[e.id for e in active],
        )
        k = fitted.kappas
    elif mode == "fixed":
        missing = [e.id for e in active if e.kappa is None]
        if missing:
            raise ValueError(
                f"calibration mode 'fixed' needs a kappa for every active emitter; "
                f"missing for {missing}"
            )
        k = np.array([e.kappa for e in active], dtype=np.float64)
    elif mode == "reference":
        if kappas is None:
            raise ValueError(
                "calibration mode 'reference' needs kappas fitted beforehand; "
                "call calibrate_scene() or pass kappas explicitly"
            )
        k = np.asarray(kappas, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(
            f"unknown calibration mode {calibration_mode!r}; expected 'reference', "
            "'per_recording', or 'fixed'"
        )

    if k.shape[0] != len(active):
        raise ValueError(
            f"scene has {len(active)} active emitters but {k.shape[0]} kappas given"
        )

    power = power_from_view_factors(F, k)
    return SimulationResult(
        name=rec.path.stem,
        path=rec.path,
        track=rec.track,
        view_factors=F,
        kappas=k,
        power_uW=power,
        measured_uW=rec.power_uW,
        metrics=_score(power, rec.power_uW, rec.track.valid),
        frame=rec.frame,
        emitter_ids=[e.id for e in active],
    )


def write_simulated(
    result: SimulationResult,
    out_path: str | Path,
    spec: RecordingSpec,
    power_column: str = "P_sim_uW",
    drop_measured: bool = True,
    drop_invalid: bool = True,
) -> Path:
    """Write a simulated recording that mirrors the input schema.

    Every non-power column is preserved, so downstream code that consumed the
    real recordings can consume the simulated ones unchanged.

    Args:
        result: what to write.
        out_path: destination file.
        spec: the spec the input was read with, used to locate power columns.
        power_column: name of the simulated power column, in microwatts.
        drop_measured: remove the measured voltage/current/power columns, so a
            simulated file cannot be mistaken for a real one.
        drop_invalid: keep only frames the simulator could evaluate.

    Returns:
        The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.frame.copy() if result.frame is not None else pd.DataFrame()
    df[power_column] = result.power_uW
    if drop_measured:
        df = df.drop(columns=[c for c in spec.power.required_columns() if c in df.columns])
    if drop_invalid:
        df = df.loc[result.track.valid]

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def simulate_batch(
    inputs: Sequence[str | Path],
    scene: Scene,
    spec: RecordingSpec,
    output_dir: Optional[str | Path] = None,
    input_root: Optional[str | Path] = None,
    kappas: Optional[np.ndarray] = None,
    calibration_mode: str = "reference",
    calibration_method: Optional[str] = None,
    power_column: str = "P_sim_uW",
    on_error: str = "warn",
    progress: bool = True,
) -> List[SimulationResult]:
    """Simulate many recordings, mirroring the input tree into ``output_dir``.

    Args:
        inputs: recordings to simulate.
        scene: the illumination environment.
        spec: how to read the recordings.
        output_dir: where to write simulated files; ``None`` skips writing.
        input_root: base directory whose structure is mirrored under
            ``output_dir``; defaults to the common parent of ``inputs``.
        kappas: pre-fitted scalars for ``reference`` mode.
        calibration_mode: see :func:`simulate_recording`.
        calibration_method: override the scene's kappa-fitting method.
        power_column: name of the simulated power column.
        on_error: ``warn`` skips a failing recording, ``raise`` propagates.
        progress: print one line per recording.

    Returns:
        Results for every recording that simulated successfully.
    """
    inputs = [Path(p) for p in inputs]
    if not inputs:
        return []
    root = Path(input_root) if input_root else None

    results: List[SimulationResult] = []
    for i, path in enumerate(inputs, start=1):
        try:
            result = simulate_recording(
                path,
                scene,
                spec,
                kappas=kappas,
                calibration_mode=calibration_mode,
                calibration_method=calibration_method,
            )
        except Exception as exc:  # noqa: BLE001 - reported per recording
            if on_error == "raise":
                raise
            print(f"  [{i}/{len(inputs)}] SKIP {path.name}: {exc}")
            continue

        if output_dir is not None:
            try:
                relative = path.relative_to(root) if root else Path(path.name)
            except ValueError:
                relative = Path(path.name)
            write_simulated(
                result,
                Path(output_dir) / relative,
                spec,
                power_column=power_column,
            )

        results.append(result)
        if progress:
            rho = result.metrics.get("pearson")
            score = f", rho={rho:.3f}" if rho is not None and np.isfinite(rho) else ""
            print(
                f"  [{i}/{len(inputs)}] {path.name}: "
                f"{result.track.n_valid}/{len(result.track)} frames{score}"
            )
    return results


def results_frame(results: Sequence[SimulationResult]) -> pd.DataFrame:
    """Per-recording summary table, one row per result."""
    return pd.DataFrame([r.summary() for r in results])
