"""Tests for scene loading, ingestion, calibration, and the end-to-end run.

A synthetic recording is generated on the fly, so these run anywhere without
the dataset. Because the trace is produced by the simulator itself, an exact
round trip is the correct expectation: calibration must recover the kappa the
data was built with.
"""

import numpy as np
import pandas as pd
import pytest

from pysolarsim import (
    RecordingSpec,
    Scene,
    calibrate_scene,
    read_recording,
    simulate_recording,
    write_simulated,
)
from pysolarsim.engine import view_factor_matrix
from pysolarsim.calibrate import fit_kappa

SCENE_DICT = {
    "name": "test_scene",
    "units": "mm",
    "emitters": [
        {
            "id": "overhead",
            "shape": "disc",
            "position": [-160.0, 870.0, 168.0],
            "normal": [0.0, -1.0, 0.0],
            "dimensions": {"diameter": 200.0},
        }
    ],
    "integration": {"disc": {"n_r": 12, "n_phi": 12}},
}

MOCAP_DICT = {
    "name": "test_mocap",
    "columns": {
        "time": "time",
        "position": ["px", "py", "pz"],
        "normal": ["nx", "ny", "nz"],
    },
    "units": {"position": "mm", "time": "s"},
    "normal_convention": "as_is",
    "sample_rate_hz": 100,
    "power": {"mode": "column", "columns": {"power": "p_uw"}, "units": "uw"},
}

TRUE_KAPPA = 4.2e7


@pytest.fixture
def scene():
    return Scene.from_dict(SCENE_DICT)


@pytest.fixture
def spec():
    return RecordingSpec.from_dict(MOCAP_DICT)


@pytest.fixture
def recording(tmp_path, scene, spec):
    """A synthetic recording whose power is exactly kappa * view factor."""
    rng = np.random.default_rng(7)
    n = 240
    t = np.arange(n) / 100.0
    position = np.stack(
        [
            200.0 * np.sin(2 * np.pi * 0.4 * t),
            300.0 + 80.0 * np.cos(2 * np.pi * 0.3 * t),
            150.0 * np.cos(2 * np.pi * 0.25 * t),
        ],
        axis=1,
    )
    normal = np.tile([0.0, 1.0, 0.0], (n, 1)) + 0.15 * rng.normal(size=(n, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    df = pd.DataFrame(
        {
            "time": t,
            "px": position[:, 0], "py": position[:, 1], "pz": position[:, 2],
            "nx": normal[:, 0], "ny": normal[:, 1], "nz": normal[:, 2],
            "p_uw": 0.0,
            "label": "synthetic",
        }
    )
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)

    track = read_recording(path, spec).track
    F = view_factor_matrix(scene, track)
    df["p_uw"] = TRUE_KAPPA * F[:, 0]
    df.to_csv(path, index=False)
    return path


def test_scene_round_trip(tmp_path, scene):
    """Saving and reloading a scene preserves its geometry."""
    path = scene.save(tmp_path / "scene.yaml")
    reloaded = Scene.load(path)
    assert reloaded.active[0].id == "overhead"
    assert np.allclose(reloaded.active[0].position, scene.active[0].position)
    assert reloaded.active[0].dimensions == scene.active[0].dimensions


def test_scene_units_are_converted_to_mm():
    """A scene authored in metres is stored internally in millimetres."""
    metric = Scene.from_dict(
        {
            "units": "m",
            "emitters": [
                {
                    "id": "e",
                    "shape": "disc",
                    "position": [0.0, 0.87, 0.0],
                    "dimensions": {"diameter": 0.2},
                }
            ],
        }
    )
    assert np.allclose(metric.active[0].position, [0.0, 870.0, 0.0])
    assert metric.active[0].dimensions["diameter"] == pytest.approx(200.0)


def test_scene_rejects_missing_dimensions():
    """A geometry without its required size is a configuration error."""
    with pytest.raises(ValueError, match="missing dimension"):
        Scene.from_dict(
            {"emitters": [{"id": "e", "shape": "rectangular", "position": [0, 1, 0],
                           "dimensions": {"width": 30}}]}
        )


def test_scene_rejects_duplicate_ids():
    """Emitter ids must be unique, since kappa is reported per id."""
    entry = {"shape": "disc", "id": "same", "position": [0, 1, 0],
             "dimensions": {"diameter": 10}}
    with pytest.raises(ValueError, match="duplicate emitter id"):
        Scene.from_dict({"emitters": [entry, dict(entry)]})


def test_disabled_emitters_are_excluded(scene):
    """A disabled emitter stays in the file but out of the sum."""
    scene.emitters[0].enabled = False
    assert scene.active == []


def test_reader_maps_columns_and_flags_dropout(tmp_path, spec):
    """Zero-position frames are dropout, and other columns survive untouched."""
    df = pd.DataFrame(
        {
            "time": [0.0, 0.01, 0.02],
            "px": [10.0, 0.0, 30.0], "py": [1.0, 0.0, 3.0], "pz": [1.0, 0.0, 3.0],
            "nx": [0.0, 0.0, 0.0], "ny": [1.0, 1.0, 1.0], "nz": [0.0, 0.0, 0.0],
            "p_uw": [1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "r.csv"
    df.to_csv(path, index=False)
    rec = read_recording(path, spec)
    assert rec.track.valid.tolist() == [True, False, True]
    assert rec.track.n_valid == 2
    assert rec.power_uW[2] == pytest.approx(3.0)


def test_negate_convention_flips_the_normal(tmp_path):
    """`negate` turns a recorded palm normal into an outward cell normal."""
    spec = RecordingSpec.from_dict({**MOCAP_DICT, "normal_convention": "negate"})
    df = pd.DataFrame(
        {"time": [0.0], "px": [1.0], "py": [1.0], "pz": [1.0],
         "nx": [0.0], "ny": [1.0], "nz": [0.0], "p_uw": [0.0]}
    )
    path = tmp_path / "r.csv"
    df.to_csv(path, index=False)
    assert np.allclose(read_recording(path, spec).track.normal[0], [0.0, -1.0, 0.0])


def test_power_unit_conversion(tmp_path):
    """Watts in the file become microwatts in the pipeline."""
    spec = RecordingSpec.from_dict(
        {**MOCAP_DICT, "power": {"mode": "v_times_i",
                                 "columns": {"voltage": "v", "current": "i"},
                                 "units": "w"}}
    )
    df = pd.DataFrame(
        {"time": [0.0], "px": [1.0], "py": [1.0], "pz": [1.0],
         "nx": [0.0], "ny": [1.0], "nz": [0.0], "v": [0.5], "i": [2e-6]}
    )
    path = tmp_path / "r.csv"
    df.to_csv(path, index=False)
    assert read_recording(path, spec).power_uW[0] == pytest.approx(1.0)


def test_calibration_recovers_the_generating_kappa(recording, scene, spec):
    """Fitting on data the simulator produced returns the kappa used to make it."""
    fit = calibrate_scene(scene, [recording], spec)
    assert fit.kappas[0] == pytest.approx(TRUE_KAPPA, rel=1e-6)
    assert fit.method == "scalar_ls"
    assert fit.residual < 1e-3


def test_simulate_reproduces_the_measured_trace(recording, scene, spec):
    """With the right kappa, the simulated trace matches the input exactly."""
    fit = calibrate_scene(scene, [recording], spec)
    result = simulate_recording(recording, scene, spec, kappas=fit.kappas)
    assert result.metrics["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert result.metrics["nrmse"] == pytest.approx(0.0, abs=1e-9)


def test_per_recording_mode_needs_no_prefitted_kappa(recording, scene, spec):
    """`per_recording` refits kappa from the recording itself."""
    result = simulate_recording(recording, scene, spec, calibration_mode="per_recording")
    assert result.kappas[0] == pytest.approx(TRUE_KAPPA, rel=1e-6)


def test_fixed_mode_requires_kappa_in_the_scene(recording, scene, spec):
    """`fixed` fails loudly rather than silently inventing a scale."""
    with pytest.raises(ValueError, match="needs a kappa"):
        simulate_recording(recording, scene, spec, calibration_mode="fixed")

    scene.emitters[0].kappa = TRUE_KAPPA
    result = simulate_recording(recording, scene, spec, calibration_mode="fixed")
    assert result.kappas[0] == pytest.approx(TRUE_KAPPA)


def test_reference_mode_without_kappa_is_an_error(recording, scene, spec):
    """`reference` mode must be given a fitted kappa."""
    with pytest.raises(ValueError, match="needs kappas fitted beforehand"):
        simulate_recording(recording, scene, spec, calibration_mode="reference")


def test_nnls_keeps_kappa_non_negative():
    """Non-negativity stops noise being absorbed into offsetting weights."""
    rng = np.random.default_rng(1)
    F = np.abs(rng.normal(size=(200, 2))) + 0.1
    target = F @ np.array([3.0, 0.0]) + 0.01 * rng.normal(size=200)
    fit = fit_kappa(F, target, method="nnls", emitter_ids=["a", "b"])
    assert np.all(fit.kappas >= 0)
    assert fit.kappas[0] == pytest.approx(3.0, rel=0.05)


def test_scalar_ls_rejects_multi_emitter_scenes():
    """Scalar least squares is only defined for a single emitter."""
    with pytest.raises(ValueError, match="exactly one active emitter"):
        fit_kappa(np.ones((10, 2)), np.ones(10), method="scalar_ls")


def test_written_output_preserves_schema(tmp_path, recording, scene, spec):
    """Simulated files keep every non-power column, so downstream code is unaffected."""
    fit = calibrate_scene(scene, [recording], spec)
    result = simulate_recording(recording, scene, spec, kappas=fit.kappas)
    out = write_simulated(result, tmp_path / "sim.csv", spec)

    written = pd.read_csv(out)
    assert "P_sim_uW" in written.columns
    assert "p_uw" not in written.columns
    assert "label" in written.columns
    assert len(written) == result.track.n_valid
