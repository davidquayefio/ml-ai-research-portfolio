"""pytest configuration: make the shared library and every module's ``src`` importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in [ROOT / "src", *sorted(ROOT.glob("0*/src"))]:
    sys.path.insert(0, str(path))
