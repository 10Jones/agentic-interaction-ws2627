"""Writes one log file per run in JSON Lines format ( one object per line)"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class RunLogger:
    def __init__(self, model: str, log_dir: str = "logs"):
        self.model = model
        # provide short random id for this run
        self.run_id = uuid.uuid4().hex[:12]
        Path(log_dir).mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{stamp}_{self.run_id}.jsonl"

    def log(self, event, ok=True, tool=None, args=None, duration_ms=None, detail=None):
        """Append one entry. 'event' is either user_task,
        tool_call, tool_result, final, or error.
        """
        entry = {
            # ISO 8601 in UTC
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            "tool": tool,
            "args": args,
            "ok": ok,
            "model": self.model,
            "duration_ms": duration_ms,
        }
        # additional field for further information such as the task text,
        # final answer or error message
        if detail is not None:
            entry["detail"] = detail

        # open in append mode, so nothing is lost if the run crashes, ensure ASCII false,
        # to keep logs readable in file
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
