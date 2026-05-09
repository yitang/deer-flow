"""Tests for RunManager model_name propagation."""

import pytest
from deerflow.runtime.runs.manager import RunManager, RunRecord
from deerflow.runtime.runs.schemas import DisconnectMode


class FakeStore:
    """Minimal RunStore spy that records put() calls."""
    def __init__(self):
        self.puts = []

    async def put(self, run_id, *, thread_id, assistant_id=None, user_id=None,
                  status="pending", model_name=None, multitask_strategy="reject",
                  metadata=None, kwargs=None, error=None, created_at=None):
        self.puts.append({
            "run_id": run_id, "thread_id": thread_id,
            "assistant_id": assistant_id, "model_name": model_name,
            "status": status, "multitask_strategy": multitask_strategy,
        })

    async def get(self, run_id):
        return None

    async def list_by_thread(self, thread_id, *, user_id=None, limit=100):
        return []

    async def update_status(self, run_id, status, *, error=None):
        pass

    async def delete(self, run_id):
        pass

    async def update_run_completion(self, run_id, *, status, **kwargs):
        pass

    async def list_pending(self, *, before=None):
        return []

    async def aggregate_tokens_by_thread(self, thread_id):
        return {}


@pytest.mark.asyncio
async def test_create_stores_model_name():
    """model_name is stored on the record and forwarded to the store."""
    store = FakeStore()
    mgr = RunManager(store=store)

    record = await mgr.create(
        "thread-1",
        "assistant-1",
        model_name="gpt-4o",
        on_disconnect=DisconnectMode.cancel,
    )

    assert record.model_name == "gpt-4o"
    assert len(store.puts) == 1
    assert store.puts[0]["model_name"] == "gpt-4o"


@pytest.mark.asyncio
async def test_create_or_reject_stores_model_name():
    """create_or_reject also propagates model_name."""
    store = FakeStore()
    mgr = RunManager(store=store)

    record = await mgr.create_or_reject(
        "thread-1",
        "assistant-1",
        model_name="claude-sonnet-4",
        multitask_strategy="reject",
    )

    assert record.model_name == "claude-sonnet-4"
    assert len(store.puts) == 1
    assert store.puts[0]["model_name"] == "claude-sonnet-4"


@pytest.mark.asyncio
async def test_create_model_name_defaults_to_none():
    """model_name is None by default (backward compatible)."""
    store = FakeStore()
    mgr = RunManager(store=store)

    record = await mgr.create("thread-1")

    assert record.model_name is None
    assert store.puts[0]["model_name"] is None
