"""Verify all contact sensor prim_paths + fingertip body names + hard-coded
elastomer body names reference existing URDF link names.

Tests catch regressions where cfg was carried over from a previous robot's
link naming (e.g. SharpaWave `*_link2`) but the new URDF uses `*_dip_sensor_link`.
"""
import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CFG_DIR = PROJECT_ROOT / "rl_isaaclab" / "tasks" / "inhand_rotate"
ENV_FILES = [
    CFG_DIR / "xhand_wave_env_cfg.py",
    CFG_DIR / "xhand_wave_grasp_env_cfg.py",
    CFG_DIR / "xhand_wave_env.py",
    CFG_DIR / "xhand_wave_grasp_env.py",
]


def _urdf_link_names() -> set:
    urdf = PROJECT_ROOT / "assets" / "xhand_green" / "xhand_right" / "urdf" / "xhand_right.urdf"
    tree = ET.parse(urdf)
    return {el.get("name") for el in tree.getroot().findall("link")}


def _extract_contact_sensor_link_names(cfg_path: Path) -> list:
    """Parse ContactSensorCfg prim_path strings and return the last path segment."""
    text = cfg_path.read_text()
    paths = re.findall(r'prim_path\s*=\s*"([^"]+)"', text)
    # filter for paths that look like robot link references (have /Robot/ in them)
    return [p.rsplit("/", 1)[-1] for p in paths if "/Robot/" in p]


def _extract_fingertip_body_names(cfg_path: Path) -> list:
    """Parse fingertip_body_names = [...] list from cfg via AST."""
    tree = ast.parse(cfg_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "fingertip_body_names" for t in stmt.targets
                ):
                    return ast.literal_eval(stmt.value)
    return []


def test_contact_sensor_link_names_exist_in_urdf():
    links = _urdf_link_names()
    bad = {}
    for cfg in [CFG_DIR / "xhand_wave_env_cfg.py", CFG_DIR / "xhand_wave_grasp_env_cfg.py"]:
        names = _extract_contact_sensor_link_names(cfg)
        missing = [n for n in names if n not in links]
        if missing:
            bad[str(cfg.name)] = missing
    assert not bad, f"contact sensor link names not in URDF: {bad}"


def test_fingertip_body_names_exist_in_urdf():
    links = _urdf_link_names()
    bad = {}
    for cfg in [CFG_DIR / "xhand_wave_env_cfg.py", CFG_DIR / "xhand_wave_grasp_env_cfg.py"]:
        names = _extract_fingertip_body_names(cfg)
        if not names:
            continue
        missing = [n for n in names if n not in links]
        if missing:
            bad[str(cfg.name)] = missing
    assert not bad, f"fingertip body names not in URDF: {bad}"


def test_hardcoded_elastomer_link_names_exist_in_urdf():
    """env.py and grasp_env.py have a hard-coded list of 5 elastomer link names.

    Verify those names exist in URDF.
    """
    links = _urdf_link_names()
    pattern = re.compile(
        r'elastomer_ids\s*=\s*\[[^\]]*?'                          # opener
        r'(?P<names>(?:\s*self\.hand\.body_names\.index\([^)]+\)\s*,?\s*)+|'  # form 1
        r'\s*\[(?P<list>[^\]]+)\])',                              # form 2: literal list
        re.DOTALL,
    )
    bad = {}
    for py in [CFG_DIR / "xhand_wave_env.py", CFG_DIR / "xhand_wave_grasp_env.py"]:
        text = py.read_text()
        # find the literal list of names: extract quoted strings inside the elastomer body construct
        m = re.search(
            r'elastomer_ids\s*=\s*\[\s*self\.hand\.body_names\.index\(body_name\)\s+for\s+body_name\s+in\s*\[([^\]]+)\]\s*\]',
            text,
            re.DOTALL,
        )
        if not m:
            continue
        names = re.findall(r'"([^"]+)"', m.group(1))
        missing = [n for n in names if n not in links]
        if missing:
            bad[py.name] = missing
    assert not bad, f"hard-coded elastomer link names not in URDF: {bad}"
