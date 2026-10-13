from __future__ import annotations
import json, os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

class EvalCheckpoint:
    STEPS = ["health","micro","perf","context_scaling","lm_eval","cost","card"]
    def __init__(self, model_key, out_dir="out"):
        self.model_key = model_key
        self.path = Path(out_dir) / f".checkpoint_{model_key}.json"
        self._state = self._load()
    def _load(self):
        if self.path.exists():
            try: return json.loads(self.path.read_text())
            except: pass
        return {"model_key": self.model_key, "steps": {}, "created": datetime.now().isoformat()}
    def done(self, step):
        e = self._state["steps"].get(step)
        return e is not None and e.get("status") == "done"
    def get(self, step):
        e = self._state["steps"].get(step)
        return e.get("data") if e else None
    def save(self, step, data=None):
        self._state["steps"][step] = {"status": "done", "timestamp": datetime.now().isoformat(), "data": data}
        self._flush()
    def skip(self, step, reason=""):
        self._state["steps"][step] = {"status": "skipped", "timestamp": datetime.now().isoformat(), "reason": reason}
        self._flush()
    def fail(self, step, error=""):
        self._state["steps"][step] = {"status": "failed", "timestamp": datetime.now().isoformat(), "error": error}
        self._flush()
    def reset(self):
        self._state = {"model_key": self.model_key, "steps": {}, "created": datetime.now().isoformat()}
        self._flush()
    def status(self):
        done = [s for s in self.STEPS if self.done(s)]
        pending = [s for s in self.STEPS if not self.done(s)]
        return {"model_key": self.model_key, "done": done, "pending": pending, "progress": f"{len(done)}/{len(self.STEPS)}"}
    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2))
        os.replace(tmp, self.path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--action", choices=["status", "reset"], default="status")
    a = ap.parse_args()
    c = EvalCheckpoint(a.model_key, a.out_dir)
    if a.action == "status":
        s = c.status()
        print(f"Model: {s['model_key']}")
        print(f"Progress: {s['progress']}")
        print(f"Done: {', '.join(s['done']) or 'none'}")
        print(f"Pending: {', '.join(s['pending']) or 'none'}")
    elif a.action == "reset":
        c.reset()
        print(f"Reset: {a.model_key}")
