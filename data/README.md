# SolarSim recordings

Synchronised photovoltaic, IMU, and hand-trajectory recordings used to
evaluate SolarSim. Collected under QUT Human Research Ethics approval #9665.

Each directory is named identically to its scene file in `configs/scene/`, so
a configuration and the data it describes always share a name:

```
data/<name>/            <-->   configs/scene/<name>.yaml
```

## Layout

| Directory | Emitter configuration | Table 1 column |
|---|---|---|
| `baseline_15participants/` | Single circular, 6000 K, 40 klx, 15 participants | Circ. 6K/40 |
| `intensity_circular_6000K_40klx/` | Single circular, 40 klx | Circ. 6K/40 |
| `intensity_circular_6000K_30klx/` | Single circular, 30 klx | Circ. 6K/30 |
| `intensity_circular_6000K_15klx/` | Single circular, 15 klx | Circ. 6K/15 |
| `intensity_circular_6000K_0p4klx/` | Single circular, 0.4 klx | Circ. 6K/0.4 |
| `cct_circular_6000K_40klx/` | Single circular, 6000 K | Circ. 6K/40 |
| `cct_circular_4000K_40klx/` | Single circular, 4000 K | Circ. 4K/40 |
| `cct_circular_2400K_40klx/` | Single circular, 2400 K | Circ. 2.4K/40 |
| `geometry_rectangular_6000K_40klx/` | Single rectangular emitter | Rect. 6K/40 |
| `multi_dual_circular_d50cm/` | Two circular emitters, 50 cm apart | Dual Circ. d=50cm |
| `multi_dual_circular_d110cm/` | Two circular emitters, 110 cm apart | Dual Circ. d=110cm |
| `multi_circular_rectangular_d70cm/` | Circular + rectangular, 70 cm apart | Circ.+Rect. d=70cm |
| `multi_circular_rectangular_d60cm/` | Circular + rectangular, 60 cm apart | not reported |

Every configuration directory holds one CSV per gesture (`Circle`, `Square`,
`Triangle`, `One`, `Two`, `Three`, and usually `Free Move`) plus a
`light_sources.txt` recording the measured emitter positions and dimensions.
`tools/build_scenes.py` parses those notes to regenerate the scene files.

`baseline_15participants/` is the 15-participant SolarTrack dataset and is
split differently:

```
baseline_15participants/
  real/       P01 .. P15    measured recordings
  simulated/  P01 .. P15    SolarSim output for the same trajectories
```

Its geometry was never written down as a layout note; the measured light
source position `(-159.81, 868.82, 168.23)` mm is recorded directly in
`configs/scene/baseline_circular_6000K_40klx.yaml`.

## Columns

All recordings are sampled at 100 Hz with `time` in seconds.

| Column | Meaning |
|---|---|
| `v_dc`, `i_dc` | Measured PV voltage (V) and current (A); power is their product |
| `P_dcs` | Simulated PV power (W), in `simulated/` only |
| `leap_palm_x/y/z` | Hand position from the Leap Motion controller, mm |
| `leap_palm_normal_x/y/z` | Palm normal; the cell normal is its negation |
| `leap_palm_orientation_x/y/z` | Palm orientation |
| `acc_x/y/z` or `xsen_acc_*`, `xsen_gyr_*`, `xsen_euler_*` | IMU, unused by SolarSim |


## Distribution

These recordings are published with the repository under the same MIT licence
as the code. They were collected under QUT Human Research Ethics approval
#9665 and contain no direct identifiers: participants appear only as
pseudonymous codes `P01`–`P15`, and every column is sensor data.

If you use this dataset, please cite the paper (see the top-level README).
