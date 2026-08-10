"""Verify Xhand USD asset paths in env cfgs point to the xhand_green assets.

Background: legacy `assets/xhand/xhand_right_urdf/xhand_right.usda` exists but is
an incomplete USD wrapper (no base mesh sublayer). The complete asset is at
`assets/xhand_green/<side>/urdf/<side>.usd` with sublayers under `configuration/`.

These tests catch both regressions:
  1. cfg pointing at a non-existent file
  2. cfg pointing at the legacy incomplete wrapper (which loads silently with empty mesh)
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CFG_DIR = PROJECT_ROOT / "rl_isaaclab" / "tasks" / "inhand_rotate"


def _resolve_usd_paths(cfg_file: Path):
    text = cfg_file.read_text()
    matches = re.findall(
        r'usd_path=os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\),\s*f?"([^"]+)"',
        text,
    )
    resolved = [(cfg_file.parent / rel).resolve() for rel in matches]
    return list(zip(matches, resolved))


def _is_xhand_path(rel: str) -> bool:
    """True if the relative path refers to an xhand asset (vs object/cylinder)."""
    return "xhand" in rel.lower()


def _is_xhand_green(rel: str) -> bool:
    """True if the path is in the xhand_green directory."""
    return "xhand_green" in rel


def test_xhand_wave_env_cfg_paths_resolve():
    cfg = CFG_DIR / "xhand_wave_env_cfg.py"
    entries = _resolve_usd_paths(cfg)
    assert entries, f"No usd_path entries in {cfg}"
    for rel, abs_path in entries:
        assert abs_path.exists(), f"USD does not exist: {abs_path} (from {rel!r})"


def test_xhand_wave_env_cfg_xhand_points_to_xhand_green():
    cfg = CFG_DIR / "xhand_wave_env_cfg.py"
    entries = _resolve_usd_paths(cfg)
    for rel, _ in entries:
        if not _is_xhand_path(rel):
            continue
        assert _is_xhand_green(rel), (
            f"Xhand cfg should point to xhand_green assets, got: {rel!r}. "
            f"Legacy assets/xhand/ is an incomplete USD wrapper without base mesh."
        )


def test_xhand_wave_grasp_env_cfg_paths_resolve():
    cfg = CFG_DIR / "xhand_wave_grasp_env_cfg.py"
    entries = _resolve_usd_paths(cfg)
    assert entries, f"No usd_path entries in {cfg}"
    for rel, abs_path in entries:
        assert abs_path.exists(), f"USD does not exist: {abs_path} (from {rel!r})"


def test_xhand_wave_grasp_env_cfg_xhand_points_to_xhand_green():
    cfg = CFG_DIR / "xhand_wave_grasp_env_cfg.py"
    entries = _resolve_usd_paths(cfg)
    for rel, _ in entries:
        if not _is_xhand_path(rel):
            continue
        assert _is_xhand_green(rel), (
            f"Xhand cfg should point to xhand_green assets, got: {rel!r}"
        )
