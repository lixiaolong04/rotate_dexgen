"""Launch MANUS-guided DexGen with automatic CW/CCW trend selection."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manus_dexgen_bridge.eval_manus_guided import has_option, main


if __name__ == "__main__":
    if not has_option(sys.argv[1:], "--manus_auto_mode_profile"):
        sys.argv[1:1] = [
            "--manus_auto_mode_profile",
            "manus_dexgen_bridge/logs/manus_auto_mode_profile.pt",
        ]
    if not has_option(sys.argv[1:], "--manus_open_profile"):
        sys.argv[1:1] = [
            "--manus_open_profile",
            "manus_dexgen_bridge/logs/manus_open_profile.pt",
        ]
    main()
