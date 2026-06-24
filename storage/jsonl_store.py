import json
import warnings
from dataclasses import asdict
from pathlib import Path

from harness.events import SessionEvent


class JsonlSessionStore:
    def __init__(self, data_dir: str | Path = "data/sessions"):
        self.data_dir = Path(data_dir)

    def _path(self, session_id: str) -> Path:
        return self.data_dir / f"{session_id}.jsonl"

    def append_event(self, event: SessionEvent) -> None:
        path = self._path(event.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def load_events(self, session_id: str) -> list[SessionEvent]:
        path = self._path(session_id)
        if not path.exists():
            return []
        events: list[SessionEvent] = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    events.append(SessionEvent(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    warnings.warn(
                        f"Skipping corrupt JSONL line {lineno} in {path}: {e}",
                        stacklevel=2,
                    )
        return events
