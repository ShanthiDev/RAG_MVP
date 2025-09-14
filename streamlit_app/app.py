# Simple launcher so Streamlit Cloud can still use app.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Home  # importing runs the Streamlit code in Home.py
