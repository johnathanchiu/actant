from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The infra layer's service fixtures (tests/infra/service_fixtures.py) also serve the agent
# side's service-tool tests (tests/test_service.py).
INFRA_TESTS = Path(__file__).resolve().parent / "infra"
if str(INFRA_TESTS) not in sys.path:
    sys.path.insert(0, str(INFRA_TESTS))
