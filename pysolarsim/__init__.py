"""SolarSim -- physics-based simulation of wearable solar harvester signals.

SolarSim synthesises photovoltaic power traces directly from motion-capture
trajectories under a configurable illumination model, with no training and no
paired recordings beyond a one-off calibration.  It takes exactly two inputs:

1. a motion trajectory -- timestamped 3-D cell positions and outward normals,
   read through a configurable adapter (:mod:`pysolarsim.io`);
2. an illumination environment -- the geometry, position, and orientation of
   every light source, authored as a scene file (:mod:`pysolarsim.scene`).

Because the radiometric kernel depends only on geometry, hardware and source
properties collapse into one calibration scalar per emitter.  Moving to a new
room means editing a scene file; moving to a new harvester means refitting
kappa.  Neither requires retraining anything.

Typical use::

    from pysolarsim import Scene, RecordingSpec, calibrate_scene, simulate_recording

    scene = Scene.load("configs/scene/baseline_circular_6000K_40klx.yaml")
    spec = RecordingSpec.load("configs/mocap/leap_csv.yaml")

    fit = calibrate_scene(scene, ["data/P01/Free Move.csv"], spec, max_seconds=60)
    result = simulate_recording("data/P02/Circle.csv", scene, spec, kappas=fit.kappas)
    print(result.power_uW, result.metrics)
"""

from .calibrate import CalibrationResult, fit_kappa
from .engine import power_from_view_factors, view_factor_matrix
from .io import MocapSpec, PowerSpec, Recording, RecordingSpec, read_recording
from .motion import MotionTrack
from .pipeline import (
    SimulationResult,
    calibrate_scene,
    discover_recordings,
    results_frame,
    simulate_batch,
    simulate_recording,
    write_simulated,
)
from .scene import Emitter, Scene

__version__ = "1.0.0"

__all__ = [
    "CalibrationResult",
    "Emitter",
    "MocapSpec",
    "MotionTrack",
    "PowerSpec",
    "Recording",
    "RecordingSpec",
    "Scene",
    "SimulationResult",
    "calibrate_scene",
    "discover_recordings",
    "fit_kappa",
    "power_from_view_factors",
    "read_recording",
    "results_frame",
    "simulate_batch",
    "simulate_recording",
    "view_factor_matrix",
    "write_simulated",
    "__version__",
]
