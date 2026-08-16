# app/tests/test_tasks_store.py
import os
import tempfile
import pytest
from tasks_store import TasksStore

@pytest.fixture
def store(tmp_path):
    return TasksStore(str(tmp_path / "test.db"))

def test_create_and_get(store):
    t = store.create_task("T-001", "demo", None)
    assert t["task_id"] == "T-001"
    assert t["mode"] == "demo"
    assert t["status"] == "running"
    fetched = store.get_task("T-001")
    assert fetched["task_id"] == "T-001"

def test_list_orders_by_started_desc(store):
    store.create_task("T-001", "demo", None)
    store.create_task("T-002", "demo", None)
    rows = store.list_tasks()
    assert rows[0]["task_id"] == "T-002"

def test_update_status(store):
    store.create_task("T-001", "demo", None)
    store.update_task("T-001", status="done", finished_at="2026-07-02T10:00:00")
    assert store.get_task("T-001")["status"] == "done"

def test_list_active_real_empty(store):
    store.create_task("T-001", "demo", None)
    assert store.list_active_real_tasks() == []

def test_list_active_real_finds_running(store):
    store.create_task("T-001", "real_test", "x.pcap")
    actives = store.list_active_real_tasks()
    assert len(actives) == 1
    assert actives[0]["mode"] in ("real_test", "real_unknown")
