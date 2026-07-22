"""Reading motion capture (and optional measured power) from tabular files.

This is input #1 to the simulator.  A :class:`MocapSpec` maps whatever column
names, units, and sign conventions a capture system happens to use onto the
canonical :class:`~pysolarsim.motion.MotionTrack`.  Swapping Leap Motion for
Vicon, OptiTrack, or a Xsens export is therefore a config edit, not a code
change -- the radiometric engine never sees the raw schema.

A :class:`PowerSpec` describes the measured PV signal in the same file, which
is needed only to fit kappa and to score fidelity.  Pure synthesis from a
trajectory needs no power column at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..motion import MotionTrack, to_mm

#: Multipliers converting a declared time unit into seconds.
TIME_UNITS_TO_S = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

#: Multipliers converting a declared power unit into microwatts.
POWER_UNITS_TO_UW = {"uw": 1.0, "µw": 1.0, "mw": 1e3, "w": 1e6}


def _require_columns(df: pd.DataFrame, columns: List[str], role: str, path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing {role} column(s) {missing}. "
            f"Available columns: {list(df.columns)}"
        )


@dataclass
class MocapSpec:
    """How to read cell position and orientation out of a recording.

    Attributes:
        columns: mapping with ``position`` and ``normal`` triples, and an
            optional ``time`` column name.
        position_units: length unit of the position columns.
        time_units: unit of the time column.
        normal_convention: ``negate`` when the file stores the palm normal and
            the cell faces the other way (Leap Motion), ``as_is`` when the
            file already stores the outward cell normal.
        sample_rate_hz: used to synthesise timestamps when no time column
            exists, and carried into the track as metadata.
        drop_zero_position: treat all-zero position rows as tracker dropout.
        drop_na: treat rows with missing required values as dropout.
    """

    columns: Dict[str, Any] = field(default_factory=dict)
    position_units: str = "mm"
    time_units: str = "s"
    normal_convention: str = "as_is"
    sample_rate_hz: Optional[float] = None
    drop_zero_position: bool = True
    drop_na: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MocapSpec":
        """Build a spec from a parsed mocap config."""
        data = dict(data or {})
        columns = dict(data.get("columns") or {})
        for key in ("position", "normal"):
            if key not in columns:
                raise ValueError(f"mocap config is missing columns.{key}")
            if len(columns[key]) != 3:
                raise ValueError(
                    f"mocap config columns.{key} must list exactly 3 column names, "
                    f"got {columns[key]}"
                )

        units = dict(data.get("units") or {})
        dropout = dict(data.get("dropout") or {})
        convention = str(data.get("normal_convention", "as_is")).lower()
        if convention not in ("as_is", "negate", "negate_palm"):
            raise ValueError(
                f"unknown normal_convention {convention!r}; expected 'as_is' or 'negate'"
            )

        return cls(
            columns=columns,
            position_units=str(units.get("position", "mm")).lower(),
            time_units=str(units.get("time", "s")).lower(),
            normal_convention="negate" if convention.startswith("negate") else "as_is",
            sample_rate_hz=(
                float(data["sample_rate_hz"]) if data.get("sample_rate_hz") else None
            ),
            drop_zero_position=bool(dropout.get("drop_zero_position", True)),
            drop_na=bool(dropout.get("drop_na", True)),
        )


@dataclass
class PowerSpec:
    """How to read the measured PV power out of a recording.

    Attributes:
        mode: ``v_times_i`` multiplies a voltage and a current column;
            ``column`` reads a single power column; ``none`` disables reading.
        columns: column names, keyed ``voltage``/``current`` or ``power``.
        units: unit of the resulting power, converted to microwatts.
    """

    mode: str = "none"
    columns: Dict[str, str] = field(default_factory=dict)
    units: str = "w"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PowerSpec":
        """Build a spec from a parsed power config; ``None`` disables it."""
        if not data:
            return cls(mode="none")
        data = dict(data)
        mode = str(data.get("mode", "none")).lower()
        columns = dict(data.get("columns") or {})

        if mode == "v_times_i":
            for key in ("voltage", "current"):
                if key not in columns:
                    raise ValueError(f"power config mode 'v_times_i' needs columns.{key}")
        elif mode == "column":
            if "power" not in columns:
                raise ValueError("power config mode 'column' needs columns.power")
        elif mode != "none":
            raise ValueError(
                f"unknown power mode {mode!r}; expected 'v_times_i', 'column', or 'none'"
            )

        units = str(data.get("units", "w")).lower()
        if units not in POWER_UNITS_TO_UW:
            raise ValueError(
                f"unknown power unit {units!r}; expected one of {sorted(POWER_UNITS_TO_UW)}"
            )
        return cls(mode=mode, columns=columns, units=units)

    @property
    def enabled(self) -> bool:
        """Whether this spec reads anything at all."""
        return self.mode != "none"

    def required_columns(self) -> List[str]:
        """Columns this spec needs present in the file."""
        if self.mode == "v_times_i":
            return [self.columns["voltage"], self.columns["current"]]
        if self.mode == "column":
            return [self.columns["power"]]
        return []


@dataclass
class RecordingSpec:
    """The complete description of an input file: motion plus optional power."""

    mocap: MocapSpec
    power: PowerSpec = field(default_factory=PowerSpec)
    name: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str = "") -> "RecordingSpec":
        """Build from a parsed mocap config file."""
        if not isinstance(data, dict):
            raise TypeError("mocap config must contain a mapping at the top level")
        return cls(
            mocap=MocapSpec.from_dict(data),
            power=PowerSpec.from_dict(data.get("power")),
            name=str(data.get("name") or name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RecordingSpec":
        """Read a mocap config from ``.yaml``/``.yml``/``.json``."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"mocap config not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        return cls.from_dict(data, name=path.stem)


@dataclass
class Recording:
    """One parsed input file.

    Attributes:
        track: the canonical motion trajectory.
        power_uW: measured power in microwatts, or ``None`` when the file has
            no PV column or the spec disables reading it.
        frame: the original table, so outputs can preserve the input schema.
        path: where it came from.
    """

    track: MotionTrack
    power_uW: Optional[np.ndarray]
    frame: pd.DataFrame
    path: Path


def read_recording(path: str | Path, spec: RecordingSpec) -> Recording:
    """Load a recording and map it onto the canonical contract.

    Args:
        path: CSV (or ``.parquet``) file to read.
        spec: how to interpret its columns, units, and conventions.

    Returns:
        The parsed :class:`Recording`.

    Raises:
        ValueError: if required columns are absent or the file is empty.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"recording not found: {path}")

    df = (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path)
    )
    if len(df) == 0:
        raise ValueError(f"{path.name}: file contains no rows")

    mocap = spec.mocap
    pos_cols = list(mocap.columns["position"])
    nrm_cols = list(mocap.columns["normal"])
    _require_columns(df, pos_cols, "position", path)
    _require_columns(df, nrm_cols, "normal", path)

    position = to_mm(df[pos_cols].to_numpy(dtype=np.float64), mocap.position_units)
    normal = df[nrm_cols].to_numpy(dtype=np.float64)
    if mocap.normal_convention == "negate":
        normal = -normal

    time_col = mocap.columns.get("time")
    if time_col and time_col in df.columns:
        scale = TIME_UNITS_TO_S.get(mocap.time_units)
        if scale is None:
            raise ValueError(
                f"unknown time unit {mocap.time_units!r}; expected one of "
                f"{sorted(TIME_UNITS_TO_S)}"
            )
        time = df[time_col].to_numpy(dtype=np.float64) * scale
    else:
        rate = mocap.sample_rate_hz or 1.0
        time = np.arange(len(df), dtype=np.float64) / rate

    valid = np.ones(len(df), dtype=bool)
    if mocap.drop_na:
        valid &= np.isfinite(position).all(axis=1) & np.isfinite(normal).all(axis=1)
        valid &= np.isfinite(time)
    if mocap.drop_zero_position:
        # A Leap Motion frame with no hand in view reports the origin; those
        # rows are dropout, not a cell sitting at (0, 0, 0).
        valid &= ~np.all(position == 0.0, axis=1)

    power_uW = None
    if spec.power.enabled:
        required = spec.power.required_columns()
        _require_columns(df, required, "power", path)
        if spec.power.mode == "v_times_i":
            raw = (
                df[spec.power.columns["voltage"]].to_numpy(dtype=np.float64)
                * df[spec.power.columns["current"]].to_numpy(dtype=np.float64)
            )
        else:
            raw = df[spec.power.columns["power"]].to_numpy(dtype=np.float64)
        power_uW = raw * POWER_UNITS_TO_UW[spec.power.units]
        if mocap.drop_na:
            valid &= np.isfinite(power_uW)

    track = MotionTrack(
        time=time,
        position=position,
        normal=normal,
        valid=valid,
        name=path.stem,
        sample_rate_hz=mocap.sample_rate_hz,
        meta={"path": str(path), "spec": spec.name},
    )
    return Recording(track=track, power_uW=power_uW, frame=df, path=path)
