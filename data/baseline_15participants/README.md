# SolarTrack Dataset - PERCOM 2026 Artifact

## Overview

This dataset contains synchronized multi-modal sensor data from the **SolarTrack** study—the first fundamental investigation of **continuous hand trajectory tracking using wearable solar harvesters**.

Unlike prior work on energy harvesting sensing that focused on categorical tasks (e.g., gesture/activity recognition), SolarTrack addresses the more challenging problem of **continuous mapping**: regressing real-time hand coordinates from photovoltaic power signals. This represents a paradigm shift from discrete classification to continuous trajectory reconstruction.

The data was collected from **15 participants** performing various hand movement tasks while wearing a custom wearable prototype equipped with a wrist-worn solar cell and Xsens IMU. Ground-truth hand positions were captured using a Leap Motion controller.

The dataset comprises approximately **700,000 synchronized samples**, enabling research in radiometric modeling, physics-informed deep learning, and sensor fusion for self-powered motion tracking.

## Data Collection Environment

### Physical Setup
- **Light Source Position** (relative to Leap Motion origin): `(-159.81, 868.82, 168.23)` mm
- **Ground Truth System**: Leap Motion Controller mounted on desk surface
- **Wearable Device**: Custom prototype with wrist-worn solar cell and Xsens IMU

### Coordinate System
The Leap Motion uses a right-handed coordinate system:
- **X-axis**: Horizontal (left/right)
- **Y-axis**: Vertical (up from the controller)
- **Z-axis**: Depth (toward/away from the user)

The light source is positioned above and to the side of the tracking area to create meaningful power variations as the hand moves through space.

## Dataset Structure

### Participants
- **Total Participants**: 15
- **Participant Folders**: P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15

### Data Collection Tasks
The dataset includes seven distinct hand movement tasks:

| Task | Description | Avg. Duration |
|------|-------------|---------------|
| Circle | Circular hand movements | ~42s |
| Square | Square-shaped hand movements | ~53s |
| Triangle | Triangular hand movements | ~47s |
| One | Number "1" tracing movements | ~40s |
| Two | Number "2" tracing movements | ~68s |
| Three | Number "3" tracing movements | ~69s |
| Free Move | Unconstrained hand movements | ~118s |

### Total Recording Time
- **Total duration per participant**: ~7.3 minutes average
- **Total dataset duration**: ~109 minutes across all participants
- **Data sampling rate**: 100 Hz

## Data Format

### File Structure
Each participant directory (P1-P15) contains **7 CSV files**, one per movement task:
- `Circle__Synchronized.csv`
- `Square__Synchronized.csv`
- `Triangle__Synchronized.csv`
- `One__Synchronized.csv`
- `Two__Synchronized.csv`
- `Three__Synchronized.csv`
- `Free Move__Synchronized.csv`

### Data Columns
Each synchronized CSV file contains **21 columns** organized as follows:

| # | Column | Description | Unit |
|---|--------|-------------|------|
| 1 | `time` | Timestamp (starts from 0) | seconds |
| 2 | `v_dc` | Solar DC voltage measurement | Volts |
| 3 | `i_dc` | Solar DC current measurement | Amperes |
| 4 | `xsen_acc_x` | Xsens accelerometer X | m/s² |
| 5 | `xsen_acc_y` | Xsens accelerometer Y | m/s² |
| 6 | `xsen_acc_z` | Xsens accelerometer Z | m/s² |
| 7 | `xsen_euler_x` | Xsens Euler angle X (roll) | degrees |
| 8 | `xsen_euler_y` | Xsens Euler angle Y (pitch) | degrees |
| 9 | `xsen_euler_z` | Xsens Euler angle Z (yaw) | degrees |
| 10 | `xsen_gyr_x` | Xsens gyroscope X | rad/s |
| 11 | `xsen_gyr_y` | Xsens gyroscope Y | rad/s |
| 12 | `xsen_gyr_z` | Xsens gyroscope Z | rad/s |
| 13 | `leap_palm_x` | Leap Motion palm position X | mm |
| 14 | `leap_palm_y` | Leap Motion palm position Y | mm |
| 15 | `leap_palm_z` | Leap Motion palm position Z | mm |
| 16 | `leap_palm_normal_x` | Leap Motion palm normal X | normalized |
| 17 | `leap_palm_normal_y` | Leap Motion palm normal Y | normalized |
| 18 | `leap_palm_normal_z` | Leap Motion palm normal Z | normalized |
| 19 | `leap_palm_orientation_x` | Leap Motion palm orientation X | normalized |
| 20 | `leap_palm_orientation_y` | Leap Motion palm orientation Y | normalized |
| 21 | `leap_palm_orientation_z` | Leap Motion palm orientation Z | normalized |

## Sensor Modalities

### Solar Energy Harvester
- Wrist-worn photovoltaic cell capturing ambient light
- DC voltage (`v_dc`) and current (`i_dc`) measurements
- Power can be computed as: `P = v_dc × i_dc`
- **Radiometric principle**: `Power = constant × geometric_coupling(cell, light_source)`
  - The geometric coupling term encompasses all motion-dependent variations
  - The constant encapsulates fixed hardware properties (cell efficiency, area, light intensity)

### Xsens IMU
- Professional-grade inertial measurement unit
- Provides accelerometer, gyroscope, and orientation (Euler angles)
- Used as reference motion data for sensor fusion

### Leap Motion Controller
- Optical hand tracking system (ground truth)
- Sub-millimeter precision for palm position
- Palm normal and orientation vectors for hand pose
- Provides reference trajectory for training and evaluation

## Data Characteristics

- **Time normalization**: All recordings start from t=0
- **Synchronization**: All sensor streams are temporally aligned
- **Missing data**: None - all datasets are complete
- **Sampling**: ~100 Hz across all modalities

## Technical Specifications

| Property | Value |
|----------|-------|
| Coordinate System (Leap) | Right-handed, Y-axis up |
| Coordinate System (Xsens) | Standard IMU conventions |
| File Encoding | UTF-8 |
| File Format | CSV (comma-separated) |
| Data Precision | Float64 |

## Applications

This dataset is suitable for:
- **Continuous motion tracking** from energy harvesting signals
- **Radiometric modeling** of solar cell power variations
- **Sensor fusion** combining solar EH with IMU data
- **Hand trajectory prediction** using deep learning
- **Energy-neutral sensing** research for self-powered wearables

## Quick Start

### Using the Data Loader API

```python
from scripts.dataloader_api import SolarTrackDataset

# Load dataset (from project root)
dataset = SolarTrackDataset(root='data/')

# Get a sample
sample = dataset.get_sample('P1', 'Circle')

# Access data as numpy arrays
power = sample.power           # Solar power (W)
trajectory = sample.trajectory # Hand position (N, 3) in mm
imu = sample.imu_acceleration  # IMU data (N, 3) in m/s²

# Visualize
sample.plot_trajectory_3d(color_by='power')
sample.plot_multi_view()
```

### Related Files
- `../scripts/dataloader_api.py` - Data loader API module
- `../scripts/dataset_walkthrough.ipynb` - Interactive exploration notebook
- `README.md` - This documentation

## Citation

When using this dataset, please cite the associated PERCOM 2026 paper:

```bibtex
@inproceedings{ghalwash2026solartrack,
  title     = {SolarTrack: Exploring the Continuous Tracking Capabilities of Wearable Solar Harvesters},
  author    = {Yasien Ghalwash and Abdelwahed Khamis and Moid Sandhu and Sara Khalifa and Raja Jurdak},
  booktitle = {Proceedings of the 24th IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  year      = {2026}
}
```

*PERCOM 2026 Artifact - 15 participants × 7 tasks = 104 synchronized recordings (~700,000 samples)*

**Note:** P4 is missing the "Free Move" task, resulting in 104 total recordings instead of 105.
