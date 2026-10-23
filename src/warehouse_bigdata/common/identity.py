"""Deterministic run identities derived from the inputs that define a run."""

import hashlib
import json
from typing import Any


def stable_digest(**parts: Any) -> str:
    """Return a 12-character digest of JSON-serialisable ``parts``.

    Key order does not matter. The same inputs always produce the same digest, so a
    rerun with identical inputs yields the same identity.
    """

    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def stable_id(prefix: str, **parts: Any) -> str:
    """Return ``<prefix>-<digest>`` for the given defining inputs."""

    return f"{prefix}-{stable_digest(**parts)}"
