from __future__ import annotations

import sys
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

from integration.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
