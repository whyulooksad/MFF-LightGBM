# app/tasks_store.py
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

_lock = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class TasksStore:
    def __init__(self, db_path="tasks.db"):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    pcap_filename TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_duration_sec REAL,
                    error_msg TEXT,
                    runtime_dir TEXT
                )
            """)

    def create_task(self, task_id, mode, pcap_filename, runtime_dir=None):
        started = _now_iso()
        with self._lock_or_yield(), self._conn() as c:
            c.execute(
                "INSERT INTO tasks(task_id, mode, pcap_filename, status, started_at, runtime_dir) VALUES (?,?,?,?,?,?)",
                (task_id, mode, pcap_filename, "running", started, runtime_dir),
            )
        return self.get_task(task_id)

    def get_task(self, task_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return dict(r) if r else None

    def list_tasks(self, limit=20):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM tasks ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def update_task(self, task_id, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [task_id]
        with self._conn() as c:
            c.execute(f"UPDATE tasks SET {cols} WHERE task_id=?", vals)

    def list_active_real_tasks(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM tasks WHERE status='running' AND mode IN ('real_test','real_unknown') ORDER BY started_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    @contextmanager
    def _lock_or_yield(self):
        with _lock:
            yield
