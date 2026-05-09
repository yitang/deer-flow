"""Tests for RunManager."""

import re
from unittest.mock import AsyncMock

import pytest

from deerflow.runtime import RunManager, RunStatus
from deerflow.runtime.runs.store.base import RunStore

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


class FakeStore(RunStore):
    """Spy implementation of RunStore that records put() calls."""

    def __init__(self) -> None:
        self._put = AsyncMock()
        self._get = AsyncMock(return_value=None)
        self._list_by_thread = AsyncMock(return_value=[])
        self._update_status = AsyncMock()
        self._delete = AsyncMock()
        self._update_run_completion = AsyncMock()
        self._list_pending = AsyncMock(return_value=[])
        self._aggregate_tokens_by_thread = AsyncMock(return_value={})

    async def put(
        self,
        run_id: str,
        *,
        thread_id: str,
        assistant_id: str | None = None,
        user_id: str | None = None,
        status: str = "pending",
        model_name: str | None = None,
        multitask_strategy: str = "reject",
        metadata: dict | None = None,
        kwargs: dict | None = None,
        error: str | None = None,
        created_at: str | None = None,
    ) -> None:
        return await self._put(
            run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            user_id=user_id,
            status=status,
            model_name=model_name,
            multitask_strategy=multitask_strategy,
            metadata=metadata,
            kwargs=kwargs,
            error=error,
            created_at=created_at,
        )

    async def get(self, run_id: str) -> dict | None:
        return await self._get(run_id)

    async def list_by_thread(self, thread_id: str, *, user_id: str | None = None, limit: int = 100) -> list[dict]:
        return await self._list_by_thread(thread_id, user_id=user_id, limit=limit)

    async def update_status(self, run_id: str, status: str, *, error: str | None = None) -> None:
        return await self._update_status(run_id, status, error=error)

    async def delete(self, run_id: str) -> None:
        return await self._delete(run_id)

    async def update_run_completion(self, run_id: str, **kwargs: object) -> None:
        return await self._update_run_completion(run_id, **kwargs)

    async def list_pending(self, *, before: str | None = None) -> list[dict]:
        return await self._list_pending(before=before)

    async def aggregate_tokens_by_thread(self, thread_id: str) -> dict:
        return await self._aggregate_tokens_by_thread(thread_id)


@pytest.fixture
def manager() -> RunManager:
    return RunManager()


@pytest.mark.anyio
async def test_create_and_get(manager: RunManager):
    """Created run should be retrievable with new fields."""
    record = await manager.create(
        "thread-1",
        "lead_agent",
        metadata={"key": "val"},
        kwargs={"input": {}},
        multitask_strategy="reject",
    )
    assert record.status == RunStatus.pending
    assert record.thread_id == "thread-1"
    assert record.assistant_id == "lead_agent"
    assert record.metadata == {"key": "val"}
    assert record.kwargs == {"input": {}}
    assert record.multitask_strategy == "reject"
    assert ISO_RE.match(record.created_at)
    assert ISO_RE.match(record.updated_at)

    fetched = manager.get(record.run_id)
    assert fetched is record


@pytest.mark.anyio
async def test_status_transitions(manager: RunManager):
    """Status should transition pending -> running -> success."""
    record = await manager.create("thread-1")
    assert record.status == RunStatus.pending

    await manager.set_status(record.run_id, RunStatus.running)
    assert record.status == RunStatus.running
    assert ISO_RE.match(record.updated_at)

    await manager.set_status(record.run_id, RunStatus.success)
    assert record.status == RunStatus.success


@pytest.mark.anyio
async def test_cancel(manager: RunManager):
    """Cancel should set abort_event and transition to interrupted."""
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)

    cancelled = await manager.cancel(record.run_id)
    assert cancelled is True
    assert record.abort_event.is_set()
    assert record.status == RunStatus.interrupted


@pytest.mark.anyio
async def test_cancel_not_inflight(manager: RunManager):
    """Cancelling a completed run should return False."""
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.success)

    cancelled = await manager.cancel(record.run_id)
    assert cancelled is False


@pytest.mark.anyio
async def test_list_by_thread(manager: RunManager):
    """Same thread should return multiple runs."""
    r1 = await manager.create("thread-1")
    r2 = await manager.create("thread-1")
    await manager.create("thread-2")

    runs = await manager.list_by_thread("thread-1")
    assert len(runs) == 2
    assert runs[0].run_id == r1.run_id
    assert runs[1].run_id == r2.run_id


@pytest.mark.anyio
async def test_list_by_thread_is_stable_when_timestamps_tie(manager: RunManager, monkeypatch: pytest.MonkeyPatch):
    """Ordering should be stable (insertion order) even when timestamps tie."""
    monkeypatch.setattr("deerflow.runtime.runs.manager._now_iso", lambda: "2026-01-01T00:00:00+00:00")

    r1 = await manager.create("thread-1")
    r2 = await manager.create("thread-1")

    runs = await manager.list_by_thread("thread-1")
    assert [run.run_id for run in runs] == [r1.run_id, r2.run_id]


@pytest.mark.anyio
async def test_has_inflight(manager: RunManager):
    """has_inflight should be True when a run is pending or running."""
    record = await manager.create("thread-1")
    assert await manager.has_inflight("thread-1") is True

    await manager.set_status(record.run_id, RunStatus.success)
    assert await manager.has_inflight("thread-1") is False


@pytest.mark.anyio
async def test_cleanup(manager: RunManager):
    """After cleanup, the run should be gone."""
    record = await manager.create("thread-1")
    run_id = record.run_id

    await manager.cleanup(run_id, delay=0)
    assert manager.get(run_id) is None


@pytest.mark.anyio
async def test_set_status_with_error(manager: RunManager):
    """Error message should be stored on the record."""
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.error, error="Something went wrong")
    assert record.status == RunStatus.error
    assert record.error == "Something went wrong"


@pytest.mark.anyio
async def test_get_nonexistent(manager: RunManager):
    """Getting a nonexistent run should return None."""
    assert manager.get("does-not-exist") is None


@pytest.mark.anyio
async def test_create_defaults(manager: RunManager):
    """Create with no optional args should use defaults."""
    record = await manager.create("thread-1")
    assert record.metadata == {}
    assert record.kwargs == {}
    assert record.multitask_strategy == "reject"
    assert record.assistant_id is None


# ---------------------------------------------------------------------------
# model_name tests — these will fail until RunRecord.model_name is added
# and RunManager.create() / create_or_reject() accept the model_name param.
# This is the intended "red" phase of TDD.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_stores_model_name():
    """create() with model_name should set it on RunRecord and pass it to store.put()."""
    store = FakeStore()
    manager = RunManager(store=store)

    record = await manager.create(
        "thread-1",
        "lead_agent",
        model_name="gpt-4o",
        metadata={"key": "val"},
        kwargs={"input": {}},
        multitask_strategy="reject",
    )

    # The record itself must carry model_name
    assert record.model_name == "gpt-4o"

    # The backing store must have received it
    store._put.assert_awaited_once()
    _call_kwargs = store._put.await_args.kwargs
    assert _call_kwargs.get("model_name") == "gpt-4o"
    assert _call_kwargs.get("thread_id") == "thread-1"
    assert _call_kwargs.get("assistant_id") == "lead_agent"


@pytest.mark.anyio
async def test_create_or_reject_stores_model_name():
    """create_or_reject() with model_name should set it on RunRecord and pass to store.put()."""
    store = FakeStore()
    manager = RunManager(store=store)

    record = await manager.create_or_reject(
        "thread-1",
        "lead_agent",
        model_name="claude-sonnet-4",
        metadata={"key": "val"},
        kwargs={"input": {}},
        multitask_strategy="reject",
    )

    assert record.model_name == "claude-sonnet-4"

    store._put.assert_awaited_once()
    _call_kwargs = store._put.await_args.kwargs
    assert _call_kwargs.get("model_name") == "claude-sonnet-4"


@pytest.mark.anyio
async def test_create_model_name_defaults_to_none():
    """create() without model_name should default to None (backward compat)."""
    store = FakeStore()
    manager = RunManager(store=store)

    record = await manager.create("thread-1")

    assert record.model_name is None

    store._put.assert_awaited_once()
    _call_kwargs = store._put.await_args.kwargs
    assert _call_kwargs.get("model_name") is None
