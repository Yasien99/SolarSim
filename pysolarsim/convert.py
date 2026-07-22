"""Turning the recorded ``Light Sources Positions.txt`` notes into scene files.

The lighting geometry for each configuration in the paper was written down as
free-form text next to the recordings, for example::

    {"Disk position left": [-143.26, 823.33, 176.53],
      {"Rectangle position right": [446.45, 891.72, 48.38  ]

    D1 = 290
    Rec_width = 30
    Rec_height = 330

That is enough for a human but not reproducible: the shape is implied by the
label wording, and dimensions are matched to emitters by position in the file.
This module parses that convention once and emits proper scene files, so the
geometry behind every published number becomes a versioned artifact.

Rules recovered from the recorded files:

- Each ``{"<label>": [x, y, z]`` line is one emitter, in millimetres.
- The label decides the shape: ``rect``/``rectangle`` means rectangular,
  anything else means a disc.
- ``D1``, ``D2``, ... are disc diameters, assigned to the disc emitters in the
  order they appear.
- ``Rec_width``/``Rec_height`` size every rectangular emitter.
- All emitters point straight down; they were ceiling-mounted overhead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scene import Scene, default_integration

#: ``{"some label": [x, y, z]`` with tolerant whitespace and optional trailing comma.
_POSITION_RE = re.compile(
    r'"(?P<label>[^"]+)"\s*:\s*\[\s*'
    r"(?P<x>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<z>-?\d+(?:\.\d+)?)\s*\]"
)

#: ``D1 = 290`` style diameter declarations.
_DIAMETER_RE = re.compile(r"\bD(?P<index>\d+)\s*=\s*(?P<value>-?\d+(?:\.\d+)?)")

#: ``Rec_width = 30`` / ``Rec_height = 330``.
_RECT_DIM_RE = re.compile(
    r"\bRec_(?P<dim>width|height)\s*=\s*(?P<value>-?\d+(?:\.\d+)?)", re.IGNORECASE
)

#: Label words that mark a rectangular emitter; everything else is a disc.
_RECT_WORDS = ("rect", "rectangle", "panel", "strip")


def parse_layout(
    text: str,
    rect_width_mm: Optional[float] = None,
    rect_height_mm: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Parse one ``Light Sources Positions.txt`` body into emitter specs.

    Args:
        text: the file contents.
        rect_width_mm: fallback rectangle width when the file omits it.
        rect_height_mm: fallback rectangle height when the file omits it.

    Returns:
        Emitter dicts in scene-file form, ready for :meth:`Scene.from_dict`.

    Raises:
        ValueError: if no emitter is found, or a dimension is missing with no
            fallback supplied.
    """
    entries = []
    for match in _POSITION_RE.finditer(text):
        label = match.group("label").strip()
        lowered = label.lower()
        shape = "rectangular" if any(w in lowered for w in _RECT_WORDS) else "disc"
        entries.append(
            {
                "label": label,
                "shape": shape,
                "position": [
                    float(match.group("x")),
                    float(match.group("y")),
                    float(match.group("z")),
                ],
            }
        )

    if not entries:
        raise ValueError("no emitter positions found in layout text")

    diameters = [
        float(m.group("value"))
        for m in sorted(_DIAMETER_RE.finditer(text), key=lambda m: int(m.group("index")))
    ]
    rect_dims = {
        m.group("dim").lower(): float(m.group("value")) for m in _RECT_DIM_RE.finditer(text)
    }
    width = rect_dims.get("width", rect_width_mm)
    height = rect_dims.get("height", rect_height_mm)

    emitters: List[Dict[str, Any]] = []
    disc_seen = 0
    for entry in entries:
        # Labels like "position left" become ids like "disc_left".
        suffix = re.sub(r"[^a-z0-9]+", "_", entry["label"].lower())
        suffix = re.sub(r"^(disk|disc|rectangle|rect|position)_?", "", suffix).strip("_")
        emitter_id = f"{entry['shape']}_{suffix}" if suffix else entry["shape"]

        if entry["shape"] == "disc":
            if disc_seen >= len(diameters):
                raise ValueError(
                    f"emitter {entry['label']!r} is a disc but the layout declares only "
                    f"{len(diameters)} diameter(s) (D1..D{len(diameters)})"
                )
            dimensions = {"diameter": diameters[disc_seen]}
            disc_seen += 1
        else:
            if width is None or height is None:
                raise ValueError(
                    f"emitter {entry['label']!r} is rectangular but the layout declares "
                    "no Rec_width/Rec_height; pass rect_width_mm and rect_height_mm"
                )
            dimensions = {"width": width, "height": height}

        emitters.append(
            {
                "id": emitter_id,
                "shape": entry["shape"],
                "position": entry["position"],
                "normal": [0.0, -1.0, 0.0],
                "dimensions": dimensions,
                "kappa": None,
                "enabled": True,
            }
        )
    return emitters


def scene_from_layout_file(
    path: str | Path,
    name: str = "",
    rect_width_mm: Optional[float] = None,
    rect_height_mm: Optional[float] = None,
    calibration: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Scene:
    """Build a :class:`Scene` from a recorded layout note.

    Args:
        path: the ``Light Sources Positions.txt`` file.
        name: scene name; defaults to the parent directory name.
        rect_width_mm: fallback rectangle width in millimetres.
        rect_height_mm: fallback rectangle height in millimetres.
        calibration: calibration block to embed in the scene.
        meta: extra provenance recorded in the scene file.

    Returns:
        The parsed scene, with its source path recorded in ``meta``.
    """
    path = Path(path)
    emitters = parse_layout(
        path.read_text(encoding="utf-8", errors="replace"),
        rect_width_mm=rect_width_mm,
        rect_height_mm=rect_height_mm,
    )
    provenance = {"source_layout": str(path)}
    provenance.update(meta or {})
    return Scene.from_dict(
        {
            "name": name or path.parent.name,
            "units": "mm",
            "emitters": emitters,
            "integration": default_integration(),
            "calibration": calibration
            or {"method": "auto", "mode": "reference", "reference": {"max_seconds": 60}},
            "meta": provenance,
        }
    )
