"""Generate scene files for every illumination configuration in the paper.

The lighting geometry behind Table 1 currently lives as free-text notes next
to the recordings (``Light Sources Positions.txt``) plus dimensions typed at
the command line. This script converts all of it into versioned scene files
under ``configs/scene/``, so each published number has a machine-readable
configuration behind it.

Run from the repository root::

    python tools/build_scenes.py
    python tools/build_scenes.py --data-root data --out-dir configs/scene
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pysolarsim.convert import scene_from_layout_file  # noqa: E402
from pysolarsim.scene import Scene, default_integration  # noqa: E402

#: The rectangular emitter's size in millimetres. Recorded in the multi-emitter
#: layout notes but omitted from the single-rectangle note, which described the
#: same physical lamp.
RECT_WIDTH_MM = 30.0
RECT_HEIGHT_MM = 330.0

#: Every configuration directory carries its own copy of the layout note.
LAYOUT_NAME = "light_sources.txt"

#: Default calibration policy: fit kappa once per environment from a short
#: paired recording and reuse it, per paper Sec. 3.3.
DEFAULT_CALIBRATION = {
    "method": "auto",
    "mode": "reference",
    "reference": {"max_seconds": 60},
}

#: The 15-participant baseline has no layout note; its geometry was carried in
#: the simulation scripts. Recorded here so it stops being a magic number.
BASELINE_EMITTER = {
    "id": "overhead_disc",
    "shape": "disc",
    "position": [-159.81, 868.82, 168.23],
    "normal": [0.0, -1.0, 0.0],
    "dimensions": {"diameter": 200.0},
    "kappa": None,
    "enabled": True,
}

#: Every configuration to emit, keyed by scene stem. Each data directory is
#: named identically to its scene file and holds its own ``light_sources.txt``,
#: so the layout note is always ``<stem>/light_sources.txt``.
#: ``table1`` names the column each scene reproduces in the paper.
CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "stem": "baseline_circular_6000K_40klx",
        "has_layout": False,
        "meta": {
            "table1": "Circ. 6K/40",
            "cct_k": 6000,
            "illuminance_klx": 40,
            "participants": 15,
            "note": (
                "SolarTrack 15-participant baseline. No layout note was recorded; "
                "the geometry was carried in the original simulation scripts."
            ),
        },
    },
    {"stem": "intensity_circular_6000K_40klx",
     "meta": {"table1": "Circ. 6K/40 (sweep)", "cct_k": 6000, "illuminance_klx": 40}},
    {"stem": "intensity_circular_6000K_30klx",
     "meta": {"table1": "Circ. 6K/30", "cct_k": 6000, "illuminance_klx": 30}},
    {"stem": "intensity_circular_6000K_15klx",
     "meta": {"table1": "Circ. 6K/15", "cct_k": 6000, "illuminance_klx": 15}},
    {"stem": "intensity_circular_6000K_0p4klx",
     "meta": {"table1": "Circ. 6K/0.4", "cct_k": 6000, "illuminance_klx": 0.4}},
    {"stem": "cct_circular_6000K_40klx",
     "meta": {"table1": "Circ. 6K/40 (CCT sweep)", "cct_k": 6000, "illuminance_klx": 40}},
    {"stem": "cct_circular_4000K_40klx",
     "meta": {"table1": "Circ. 4K/40", "cct_k": 4000, "illuminance_klx": 40,
              "note": "Same fixture and geometry as the 6000 K session; only CCT changed."}},
    {"stem": "cct_circular_2400K_40klx",
     "meta": {"table1": "Circ. 2.4K/40", "cct_k": 2400, "illuminance_klx": 40,
              "note": "Same fixture and geometry as the 6000 K session; only CCT changed."}},
    {"stem": "geometry_rectangular_6000K_40klx",
     "meta": {"table1": "Rect. 6K/40", "cct_k": 6000, "illuminance_klx": 40,
              "note": (
                  "The layout note omits Rec_width/Rec_height; the same physical "
                  f"{RECT_WIDTH_MM:g}x{RECT_HEIGHT_MM:g} mm lamp is recorded in the "
                  "multi-emitter notes."
              )}},
    {"stem": "multi_dual_circular_d50cm",
     "meta": {"table1": "Dual Circ. d=50cm", "cct_k": 6000, "illuminance_klx": 40}},
    {"stem": "multi_dual_circular_d110cm",
     "meta": {"table1": "Dual Circ. d=110cm", "cct_k": 6000, "illuminance_klx": 40}},
    {"stem": "multi_circular_rectangular_d70cm",
     "meta": {"table1": "Circ.+Rect. d=70cm", "cct_k": 6000, "illuminance_klx": 40}},
    {"stem": "multi_circular_rectangular_d60cm",
     "meta": {"table1": None, "cct_k": 6000, "illuminance_klx": 40,
              "note": "Additional mixed-geometry session, not reported in Table 1."}},
]


def emitter_separation_mm(scene: Scene) -> Optional[float]:
    """Horizontal separation between the first two emitters, for cross-checking.

    Table 1 labels the multi-emitter rows by the spacing between fixtures, so
    recomputing it from the parsed geometry catches a mis-parsed layout.
    """
    active = scene.active
    if len(active) < 2:
        return None
    a, b = active[0].position, active[1].position
    return float(((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)


def build_scene(config: Dict[str, Any], data_root: Path) -> Scene:
    """Build one scene from its layout note, or from recorded constants."""
    stem = config["stem"]
    meta = dict(config["meta"])
    meta["data_dir"] = stem

    if not config.get("has_layout", True):
        return Scene.from_dict(
            {
                "name": stem,
                "units": "mm",
                "emitters": [BASELINE_EMITTER],
                "integration": default_integration(),
                "calibration": DEFAULT_CALIBRATION,
                "meta": meta,
            }
        )

    layout_path = data_root / stem / LAYOUT_NAME
    if not layout_path.is_file():
        raise FileNotFoundError(f"layout note not found: {layout_path}")

    return scene_from_layout_file(
        layout_path,
        name=stem,
        rect_width_mm=RECT_WIDTH_MM,
        rect_height_mm=RECT_HEIGHT_MM,
        calibration=DEFAULT_CALIBRATION,
        meta=meta,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="directory holding the recordings")
    parser.add_argument("--out-dir", default="configs/scene", help="where to write scene files")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    written, failed = 0, 0
    for config in CONFIGURATIONS:
        try:
            scene = build_scene(config, data_root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP {config['stem']}: {exc}")
            failed += 1
            continue

        path = scene.save(out_dir / f"{config['stem']}.yaml")
        shapes = "+".join(e.shape for e in scene.active)
        separation = emitter_separation_mm(scene)
        detail = f", separation {separation / 10:.1f} cm" if separation else ""
        print(f"  {path.name}: {len(scene.active)} emitter(s) [{shapes}]{detail}")
        written += 1

    print(f"\nWrote {written} scene file(s) to {out_dir}" + (f", {failed} skipped" if failed else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
