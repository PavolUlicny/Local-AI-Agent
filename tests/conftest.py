from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Ensure the project root (which contains the `src` package) is importable during tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress RuntimeWarnings from mock coroutines during test cleanup
warnings.filterwarnings("ignore", message="coroutine.*was never awaited", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*AsyncMockMixin.*", category=RuntimeWarning)
