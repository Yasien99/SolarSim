# SolarSim

**Physics-based simulation of wearable solar harvester signals for motion sensing.**

Reference implementation for *SolarSim: Physics-Based Simulation of Wearable
Solar Harvester Signals for Motion Sensing*.

Wearable solar harvesters can sense motion while they harvest energy, but
every existing solar-sensing system needs real photovoltaic recordings
collected per participant and per lighting environment. SolarSim removes that
bottleneck: it synthesises PV traces directly from motion trajectories using a
radiometric view-factor model, with no training and no paired recordings
beyond a one-off calibration per environment.

It takes exactly two inputs:

1. **A motion trajectory** — timestamped 3-D cell positions and outward
   normals from any motion-capture system.
2. **An illumination environment** — the geometry, position, and orientation
   of every light source in the scene.

Because the radiometric kernel depends only on geometry, everything
hardware-specific — cell efficiency and area, source intensity, spectral
response — collapses into a single calibration scalar `κ` per emitter.
Changing the room means editing a scene file. Changing the harvester means
refitting `κ`. Neither requires retraining anything.

Single- and multi-emitter scenes are supported, with mixed geometries
(circular, rectangular, spherical) composing additively under one calibration.

## Install

```bash
pip install -e .
```

Python 3.9+. Depends only on NumPy, SciPy, pandas, and PyYAML.

## Quick start

```bash
# Inspect a lighting configuration.
pysolarsim describe --scene configs/scene/baseline_circular_6000K_40klx.yaml
```

```bash
# Fit kappa once for an environment, from ~60 s of paired motion and PV.
pysolarsim calibrate --scene configs/scene/baseline_circular_6000K_40klx.yaml --mocap configs/mocap/leap_csv.yaml --input data/baseline_15participants/real --max-seconds 60 --out results/kappa.json
```

```bash
# Synthesise power for a whole dataset, mirroring the input tree.
pysolarsim simulate --config configs/run/baseline.yaml
```

```bash
# Score simulated traces against measured PV.
pysolarsim validate --scene configs/scene/geometry_rectangular_6000K_40klx.yaml --mocap configs/mocap/leap_csv.yaml --input data/geometry_rectangular_6000K_40klx --calibration-mode per_recording
```

`--input` accepts a single recording or a directory.

### Python API

```python
from pysolarsim import Scene, RecordingSpec, calibrate_scene, simulate_recording

scene = Scene.load("configs/scene/baseline_circular_6000K_40klx.yaml")
spec = RecordingSpec.load("configs/mocap/leap_csv.yaml")

fit = calibrate_scene(scene, ["data/baseline_15participants/real/P01/Free Move__Synchronized.csv"], spec, max_seconds=60)
result = simulate_recording("data/baseline_15participants/real/P02/Circle__Synchronized.csv", scene, spec, kappas=fit.kappas)

print(result.power_uW)   # simulated trace, microwatts
print(result.metrics)    # pearson, rmse, nrmse, ...
```

## Describing your own setup

**The illumination environment** is a scene file. Add a second emitter and
multi-emitter superposition happens automatically, with non-negative least
squares fitting one `κ` per source.

```yaml
units: mm
emitters:
  - id: overhead_disc
    shape: disc              # disc | rectangular | spherical
    position: [-159.81, 868.82, 168.23]
    normal: [0, -1, 0]       # outward; defaults to straight down
    dimensions: {diameter: 200.0}
    kappa: null              # null means "fit me"
integration:
  disc: {n_r: 20, n_phi: 20}
  rectangular: {n_u: 20, n_v: 20}
calibration:
  method: auto               # auto | scalar_ls | nnls
  mode: reference            # reference | per_recording | fixed
```

**The motion input** is a mocap config that maps your capture system's column
names, units, and sign conventions onto the simulator's canonical trajectory.
Supporting Vicon, OptiTrack, or a Xsens export is a config edit, not a code
change.

```yaml
columns:
  time: time
  position: [leap_palm_x, leap_palm_y, leap_palm_z]
  normal: [leap_palm_normal_x, leap_palm_normal_y, leap_palm_normal_z]
units: {position: mm, time: s}
normal_convention: negate    # Leap reports the palm normal; the cell faces the other way
sample_rate_hz: 100
power:                       # only needed for calibration and fidelity scoring
  mode: v_times_i
  columns: {voltage: v_dc, current: i_dc}
  units: w
```

`configs/scene/` ships a scene for every illumination configuration evaluated
in the paper: four source intensities, three colour temperatures, a
rectangular emitter, and three multi-emitter layouts.

## Repository layout

| Path | Contents |
|---|---|
| `pysolarsim/` | The simulator package ([details](pysolarsim/README.md)) |
| `configs/scene/` | One scene file per illumination configuration |
| `configs/mocap/` | Input adapters for motion-capture formats |
| `configs/run/` | Complete run configurations |
| `tools/build_scenes.py` | Regenerate scene files from recorded layout notes |
| `tools/run_fidelity.py` | Score fidelity across all configurations |

## Data

The recordings evaluated in the paper are included in `data/`: synchronised
PV, IMU, and hand-trajectory measurements from 15 participants, collected
under QUT Human Research Ethics approval #9665. Participants are identified
only by pseudonymous codes (`P01`–`P15`).

Each configuration directory is named identically to its scene file, so
`data/<name>/` and `configs/scene/<name>.yaml` always correspond. See
[`data/README.md`](data/README.md) for the full layout and column reference.

The code runs against any recording a mocap config can describe, not just
these.

## Tests

```bash
python -m pytest pysolarsim/tests -q
```

## Citation

```bibtex
@inproceedings{ghalwash2026solarsim,
  title     = {SolarSim: Physics-Based Simulation of Wearable Solar Harvester
               Signals for Motion Sensing},
  author    = {Ghalwash, Yasien and Khamis, Abdelwahed and Khalifa, Sara and
               Sandhu, Moid and Jurdak, Raja},
  booktitle = {Proceedings of the ACM on Interactive, Mobile, Wearable and
               Ubiquitous Technologies},
  year      = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
