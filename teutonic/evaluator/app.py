"""Document-only evaluator API; packed UTF-8 data and fixed Horizon reference."""

from __future__ import annotations

import os

from teutonic.evaluator import engine as base

app = base.app


if __name__ == "__main__":
    import uvicorn

    base.setup_logging()
    host = os.environ.get("EVAL_HOST", "127.0.0.1")
    port = int(os.environ.get("EVAL_PORT", "9000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
