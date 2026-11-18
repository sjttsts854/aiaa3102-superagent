"""Read one local API credential without printing it or evaluating shell syntax."""
from __future__ import annotations

import os
from pathlib import Path


LOCAL_ENV_PATH = Path(__file__).resolve().parent.parent / ".env.local"


def read_api_key(name: str, env_path: Path | None = None) -> str | None:
    environment_value = os.environ.get(name, "").strip()
    if environment_value:
        return environment_value
    path = LOCAL_ENV_PATH if env_path is None else env_path
    if not path.is_file():
        return None
    found = None
    occurrences = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if key.strip() != name:
            continue
        occurrences += 1
        if not separator or occurrences > 1:
            raise ValueError("Local credential file has an invalid or duplicate key entry")
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("Local credential file contains an unclosed quoted value")
            value = value[1:-1]
        if any(character.isspace() for character in value):
            raise ValueError("API key must not contain whitespace")
        found = value or None
    return found
