"""The illumination environment: input #2 to the simulator.

A :class:`Scene` is the machine-readable description of a lighting setup --
the position, orientation, and geometry of every emitter, plus the numerical
integration settings and calibration policy.  It is the whole of what changes
when you move to a different room (paper Sec. 3.1): the radiometric engine and
the motion data stay exactly as they are.

Scenes are authored as YAML (JSON is accepted too) so a lighting
configuration is a versioned artifact rather than constants buried in a
script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .kernel import (
    DEFAULT_CHUNK,
    DEFAULT_EMITTER_NORMAL,
    disc_view_factor,
    rect_view_factor,
    resolve_emitter_normal,
    sphere_view_factor,
)
from .motion import LENGTH_UNITS_TO_MM

#: Geometry names accepted in a scene file, mapped to their required dimensions.
SHAPE_DIMENSIONS = {
    "disc": ("diameter",),
    "rectangular": ("width", "height"),
    "spherical": ("radius",),
}

#: Aliases so scenes written against other vocabularies still load.
SHAPE_ALIASES = {
    "disk": "disc",
    "circle": "disc",
    "circular": "disc",
    "rect": "rectangular",
    "rectangle": "rectangular",
    "panel": "rectangular",
    "sphere": "spherical",
    "point": "spherical",
}

#: Dimension aliases, including the millimetre-suffixed names used by the
#: original scripts.
DIMENSION_ALIASES = {
    "diameter_mm": "diameter",
    "d": "diameter",
    "width_mm": "width",
    "w": "width",
    "height_mm": "height",
    "length": "height",
    "length_mm": "height",
    "h": "height",
    "radius_mm": "radius",
    "r": "radius",
}


def _canonical_shape(shape: str) -> str:
    key = str(shape).strip().lower()
    key = SHAPE_ALIASES.get(key, key)
    if key not in SHAPE_DIMENSIONS:
        raise ValueError(
            f"Unknown emitter shape {shape!r}; expected one of "
            f"{sorted(SHAPE_DIMENSIONS)}"
        )
    return key


@dataclass
class Emitter:
    """One light source in the scene.

    Attributes:
        id: stable name used in reports and per-emitter calibration output.
        shape: ``disc``, ``rectangular``, or ``spherical``.
        position: ``(3,)`` centre in millimetres, in the mocap frame.
        normal: ``(3,)`` unit outward normal; defaults to straight down.
        dimensions: geometry sizes in millimetres, keyed per shape.
        kappa: calibration scalar, or ``None`` to fit it from paired data.
        enabled: ``False`` keeps the emitter in the file but out of the sum.
    """

    id: str
    shape: str
    position: np.ndarray
    normal: np.ndarray
    dimensions: Dict[str, float]
    kappa: Optional[float] = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, spec: Dict[str, Any], index: int, scale: float) -> "Emitter":
        """Build an emitter from one entry of a scene file.

        ``scale`` converts the scene's declared units into millimetres and is
        applied to the position and every dimension.
        """
        if not isinstance(spec, dict):
            raise TypeError(f"emitter #{index} must be a mapping, got {type(spec)}")

        shape = _canonical_shape(spec.get("shape", ""))
        emitter_id = str(spec.get("id") or f"{shape}_{index}")

        if "position" not in spec:
            raise ValueError(f"emitter {emitter_id!r} is missing 'position'")
        position = np.asarray(spec["position"], dtype=np.float64).reshape(3) * scale

        normal_spec = spec.get("normal", spec.get("orientation"))
        normal = resolve_emitter_normal(
            None if normal_spec is None else np.asarray(normal_spec, dtype=np.float64)
        )

        raw_dims = spec.get("dimensions") or {}
        dims: Dict[str, float] = {}
        for key, value in raw_dims.items():
            canonical = DIMENSION_ALIASES.get(str(key).lower(), str(key).lower())
            dims[canonical] = float(value) * scale

        missing = [d for d in SHAPE_DIMENSIONS[shape] if d not in dims]
        if missing:
            raise ValueError(
                f"emitter {emitter_id!r} ({shape}) is missing dimension(s) "
                f"{missing}; got {sorted(dims)}"
            )
        for name in SHAPE_DIMENSIONS[shape]:
            if dims[name] <= 0:
                raise ValueError(
                    f"emitter {emitter_id!r} has non-positive {name}={dims[name]}"
                )

        kappa = spec.get("kappa")
        return cls(
            id=emitter_id,
            shape=shape,
            position=position,
            normal=normal,
            dimensions=dims,
            kappa=None if kappa is None else float(kappa),
            enabled=bool(spec.get("enabled", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a scene-file entry, in millimetres."""
        return {
            "id": self.id,
            "shape": self.shape,
            "position": [float(v) for v in self.position],
            "normal": [float(v) for v in self.normal],
            "dimensions": {k: float(v) for k, v in self.dimensions.items()},
            "kappa": None if self.kappa is None else float(self.kappa),
            "enabled": bool(self.enabled),
        }

    def view_factor(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        integration: Dict[str, Dict[str, int]],
        chunk: int = DEFAULT_CHUNK,
    ) -> np.ndarray:
        """Geometric view factor of this emitter over a trajectory.

        Dispatches to the quadrature for this shape; all shapes share the same
        radiometric kernel.
        """
        if self.shape == "disc":
            grid = integration.get("disc", {})
            return disc_view_factor(
                positions,
                normals,
                self.position,
                self.dimensions["diameter"],
                emitter_normal=self.normal,
                n_r=int(grid.get("n_r", 20)),
                n_phi=int(grid.get("n_phi", 20)),
                chunk=chunk,
            )
        if self.shape == "rectangular":
            grid = integration.get("rectangular", {})
            return rect_view_factor(
                positions,
                normals,
                self.position,
                self.dimensions["width"],
                self.dimensions["height"],
                emitter_normal=self.normal,
                n_u=int(grid.get("n_u", 20)),
                n_v=int(grid.get("n_v", 20)),
                chunk=chunk,
            )
        if self.shape == "spherical":
            return sphere_view_factor(
                positions,
                normals,
                self.position,
                self.dimensions["radius"],
                emitter_normal=self.normal,
            )
        raise ValueError(f"Unsupported emitter shape: {self.shape}")


@dataclass
class Scene:
    """A complete illumination environment.

    Attributes:
        emitters: every light source, including disabled ones.
        integration: per-shape quadrature resolution.
        calibration: how kappa is obtained (``method`` and optional reference).
        name: human-readable label carried into outputs.
        meta: free-form provenance, e.g. how the scene was measured.
    """

    emitters: List[Emitter] = field(default_factory=list)
    integration: Dict[str, Dict[str, int]] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> List[Emitter]:
        """Emitters that contribute to the power sum."""
        return [e for e in self.emitters if e.enabled]

    @property
    def calibration_method(self) -> str:
        """Configured kappa-fitting method, defaulting to ``auto``.

        ``auto`` picks scalar least squares for a single emitter and NNLS for
        multi-emitter scenes, matching paper Sec. 3.3.
        """
        return str(self.calibration.get("method", "auto")).lower()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str = "") -> "Scene":
        """Build a scene from a parsed scene file."""
        if not isinstance(data, dict):
            raise TypeError("scene file must contain a mapping at the top level")

        units = str(data.get("units", "mm")).lower()
        if units not in LENGTH_UNITS_TO_MM:
            raise ValueError(
                f"scene units {units!r} not recognised; expected one of "
                f"{sorted(LENGTH_UNITS_TO_MM)}"
            )
        scale = LENGTH_UNITS_TO_MM[units]

        raw_emitters = data.get("emitters")
        if not raw_emitters:
            raise ValueError("scene defines no emitters")

        emitters = [
            Emitter.from_dict(spec, i, scale) for i, spec in enumerate(raw_emitters)
        ]
        ids = [e.id for e in emitters]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate emitter id(s): {sorted(duplicates)}")

        integration = {
            str(k).lower(): dict(v)
            for k, v in (data.get("integration") or {}).items()
        }
        # Accept the alias here too, so 'rect:' works in the integration block.
        for alias, canonical in SHAPE_ALIASES.items():
            if alias in integration and canonical not in integration:
                integration[canonical] = integration.pop(alias)

        return cls(
            emitters=emitters,
            integration=integration,
            calibration=dict(data.get("calibration") or {}),
            name=str(data.get("name") or name),
            meta=dict(data.get("meta") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        """Read a scene from a ``.yaml``/``.yml``/``.json`` file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"scene file not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        return cls.from_dict(data, name=path.stem)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the scene, always in millimetres."""
        return {
            "name": self.name,
            "units": "mm",
            "meta": self.meta,
            "emitters": [e.to_dict() for e in self.emitters],
            "integration": self.integration,
            "calibration": self.calibration,
        }

    def save(self, path: str | Path) -> Path:
        """Write the scene to YAML (or JSON, by extension)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            path.write_text(
                yaml.safe_dump(data, sort_keys=False, default_flow_style=None),
                encoding="utf-8",
            )
        return path

    def describe(self) -> str:
        """One line per emitter, for CLI output and run logs."""
        lines = [f"scene {self.name!r}: {len(self.active)}/{len(self.emitters)} active"]
        for e in self.emitters:
            dims = ", ".join(f"{k}={v:g}mm" for k, v in sorted(e.dimensions.items()))
            state = "" if e.enabled else "  [disabled]"
            kappa = "fit" if e.kappa is None else f"{e.kappa:g}"
            lines.append(
                f"  {e.id}: {e.shape} at {np.round(e.position, 2).tolist()} mm, "
                f"normal {np.round(e.normal, 3).tolist()}, {dims}, kappa={kappa}{state}"
            )
        return "\n".join(lines)


def default_integration() -> Dict[str, Dict[str, int]]:
    """The quadrature resolution used throughout the paper."""
    return {"disc": {"n_r": 20, "n_phi": 20}, "rectangular": {"n_u": 20, "n_v": 20}}
