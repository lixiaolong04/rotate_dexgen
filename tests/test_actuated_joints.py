"""Verify XhandWaveEnvCfg covers all 21 URDF revolute joints in actuated_joint_names
and that action_space / observation_space / joint_pos init are all consistent.

We parse the cfg file via AST instead of importing it, because importing
isaaclab.sim requires the USD `pxr` module which is only available inside an
Isaac Sim simulation_app context.
"""
import ast
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CFG_PATH = PROJECT_ROOT / "rl_isaaclab" / "tasks" / "inhand_rotate" / "xhand_wave_env_cfg.py"


def _urdf_revolute_joint_names():
    urdf = PROJECT_ROOT / "assets" / "xhand_green" / "xhand_right" / "urdf" / "xhand_right.urdf"
    tree = ET.parse(urdf)
    return sorted(
        j.get("name") for j in tree.getroot().findall("joint") if j.get("type") == "revolute"
    )


def _parse_cfg_class():
    """Return the AST ClassDef node for XhandWaveEnvCfg."""
    tree = ast.parse(CFG_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "XhandWaveEnvCfg":
            return node
    raise AssertionError(f"XhandWaveEnvCfg class not found in {CFG_PATH}")


def _get_class_attr_value(class_node: ast.ClassDef, name: str):
    """Extract the literal value of a top-level class attribute assignment."""
    for stmt in class_node.body:
        # plain assignment: name = value
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            return ast.literal_eval(stmt.value)
        # annotated assignment with value: name: type = value
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            if stmt.value is not None:
                return ast.literal_eval(stmt.value)
    raise AssertionError(f"attribute {name!r} not found in XhandWaveEnvCfg")


def _get_joint_pos_keys() -> set:
    """Extract the dict literal in robot_cfg.init_state.joint_pos and return its keys."""
    text = CFG_PATH.read_text()
    # find the joint_pos={...} block (very fragile — but acceptable for a test)
    import re
    m = re.search(r'joint_pos\s*=\s*\{([^}]+)\}', text, re.DOTALL)
    assert m, "joint_pos={...} block not found in cfg"
    body = m.group(1)
    # extract quoted strings as keys
    keys = re.findall(r'"([^"]+)"\s*:', body)
    return set(keys)


def test_action_space_is_21():
    cls = _parse_cfg_class()
    assert _get_class_attr_value(cls, "action_space") == 21


def test_actuated_joints_count_21():
    cls = _parse_cfg_class()
    names = _get_class_attr_value(cls, "actuated_joint_names")
    assert len(names) == 21, f"expected 21, got {len(names)}: {names}"


def test_actuated_joints_match_urdf_revolutes():
    cls = _parse_cfg_class()
    cfg_joints = set(_get_class_attr_value(cls, "actuated_joint_names"))
    urdf_joints = set(_urdf_revolute_joint_names())
    missing = urdf_joints - cfg_joints
    extra = cfg_joints - urdf_joints
    assert not missing and not extra, f"diff: missing={missing}, extra={extra}"


def test_init_joint_pos_covers_all_21():
    cls = _parse_cfg_class()
    cfg_joints = set(_get_class_attr_value(cls, "actuated_joint_names"))
    init_keys = _get_joint_pos_keys()
    missing = cfg_joints - init_keys
    assert not missing, f"init_joint_pos missing keys: {missing}"


def test_observation_space_matches_21dof():
    cls = _parse_cfg_class()
    obs = _get_class_attr_value(cls, "observation_space")
    # Current v2 policy contract drops the unused 15-D contact-position zeros.
    EXPECTED = (21 + 21 + 5) * 3  # = 141
    assert obs == EXPECTED, f"observation_space {obs} != expected {EXPECTED}"


def test_observation_space_divisible_by_3():
    cls = _parse_cfg_class()
    obs = _get_class_attr_value(cls, "observation_space")
    assert obs % 3 == 0
