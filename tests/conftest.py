"""Make `pytest tests/` import this worktree's package, not an editable install.

`auto_benchmarkcard` may be pip-installed editable against a different checkout that has
no composer/evidence.py. Inserting this worktree's src first ensures the suite exercises
the branch under review without a manual PYTHONPATH override.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
