"""Make `pytest tests/` import this checkout's src/ ahead of any installed copy.

`auto_benchmarkcard` may also be pip-installed (editable or not) from another checkout.
Inserting this checkout's src/ first on sys.path ensures the suite exercises this code
without a manual PYTHONPATH override.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
