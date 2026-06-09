"""
crypto.governance.registry
=============================
Experiment registry (v6 §9.2.1): Confirmatory experiments MUST be pre-registered
before running.  Pre-registration writes a read-only config entry (hashed) and,
if available, the current git commit.  A Confirmatory run asserts that its exact
config was pre-registered (hash match), else it is treated as Exploratory and
must not be used as a main conclusion.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def pre_register(config: dict, registry_path, label: str = "") -> dict:
    """Append a Confirmatory pre-registration entry. Returns the entry."""
    path = Path(registry_path)
    entries = []
    if path.exists():
        entries = json.loads(path.read_text())
    entry = {"label": label, "config_digest": config_digest(config),
             "git_commit": _git_commit(), "ts": time.time(),
             "kind": "confirmatory", "config": config}
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, default=str))
    return entry


def is_preregistered(config: dict, registry_path) -> bool:
    path = Path(registry_path)
    if not path.exists():
        return False
    entries = json.loads(path.read_text())
    d = config_digest(config)
    return any(e.get("config_digest") == d for e in entries)


def assert_preregistered(config: dict, registry_path):
    """Confirmatory gate: raise if this exact config was not pre-registered."""
    if not is_preregistered(config, registry_path):
        raise PermissionError(
            "Confirmatory run blocked: config not pre-registered (would be "
            "Exploratory). Call pre_register() before running, and commit it.")
    return True
