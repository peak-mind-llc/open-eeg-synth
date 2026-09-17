"""Regenerate the golden fingerprint (tests/golden/) that tests/test_golden_engine.py checks.

Run it after a deliberate signal change, in the same commit as the SIGNAL_VERSION bump in
version.py (the test fails after a bump until this has run), or after the test changes what the
fingerprint records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_golden_engine import GOLDEN, fingerprint  # noqa: E402

GOLDEN.parent.mkdir(parents=True, exist_ok=True)
GOLDEN.write_text(json.dumps(fingerprint(), indent=1))
print("wrote", GOLDEN)
