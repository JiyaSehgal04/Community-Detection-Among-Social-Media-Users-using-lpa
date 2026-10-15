"""Entry point: `uvicorn backend.main:app --reload --port 8000`."""

from backend.app import create_app

app = create_app()
