from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    ready_root = root / "YYQEI_20260508_NLLB_ready"
    cmd = [
        sys.executable,
        str(root / "scripts" / "validate_ready_package.py"),
        "--root",
        str(ready_root),
    ]
    raise SystemExit(subprocess.run(cmd, cwd=root).returncode)


if __name__ == "__main__":
    main()
