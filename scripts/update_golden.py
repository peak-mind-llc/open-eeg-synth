"""Regenerate the golden fingerprint after a deliberate signal change.

Bump SIGNAL_VERSION in version.py first.
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
