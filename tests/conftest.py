"""Put the repository root on sys.path so `import classifier` works.

Without this the suite only passes when pytest happens to be invoked from the
repository root with rootdir insertion enabled; on a clean checkout in CI the
import fails.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
