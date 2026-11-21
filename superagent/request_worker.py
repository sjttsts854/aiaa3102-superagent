"""One HTTP request in a disposable process; secrets enter only via stdin."""
from __future__ import annotations

import base64
import json
import sys
from types import SimpleNamespace

from .network import _post_completion_direct, safe_error_details


def main() -> int:
    try:
        envelope = json.loads(sys.stdin.buffer.read())
        if not isinstance(envelope, dict) or not isinstance(envelope.get("config"), dict):
            raise ValueError("Invalid transport envelope")
        if not isinstance(envelope.get("payload"), dict) or not isinstance(envelope.get("api_key"), str):
            raise ValueError("Invalid transport envelope")
        config = SimpleNamespace(**envelope["config"])
        response = _post_completion_direct(config, envelope["payload"], envelope["api_key"])
        reply = {"ok": True, "status_code": response.status_code,
                 "body_base64": base64.b64encode(response.content).decode("ascii"),
                 "content_type": response.headers.get("content-type")}
    except Exception as exc:
        # Do not print tracebacks, arbitrary exception text, URLs or headers.
        reply = {"ok": False, "diagnostic": safe_error_details(exc)}
    sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=True, allow_nan=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
