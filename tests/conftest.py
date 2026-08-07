"""Pytest config: put repo root + scripts/ on sys.path so tests can import the
shared validator, and expose REPO_ROOT."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
