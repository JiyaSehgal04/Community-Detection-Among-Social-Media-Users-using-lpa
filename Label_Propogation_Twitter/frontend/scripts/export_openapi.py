"""Write the API's OpenAPI schema to frontend/openapi.json (the input for `npm run gen:api`).

Run from anywhere; it loads the real app, so data/processed/ must exist.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # frontend/scripts/ -> project root
sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402

out = Path(__file__).resolve().parents[1] / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2))
print("wrote", out)
