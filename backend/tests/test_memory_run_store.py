"""Tests for MemoryRunStore model_name support.

Tests that MemoryRunStore.put() stores and retrieves model_name
without breaking existing functionality.
"""

import pytest

from deerflow.runtime.runs.store.memory import MemoryRunStore


@pytest.mark.anyio
async def test_memory_store_put_stores_model_name():
    store = MemoryRunStore()
    await store.put("r1", thread_id="t1", model_name="gpt-4o")
    row = await store.get("r1")
    assert row is not None
    assert row["model_name"] == "gpt-4o"


@pytest.mark.anyio
async def test_memory_store_model_name_defaults_to_none():
    store = MemoryRunStore()
    await store.put("r1", thread_id="t1")
    row = await store.get("r1")
    assert row is not None
    assert row["model_name"] is None


@pytest.mark.anyio
async def test_memory_store_model_name_preserved_after_update_status():
    store = MemoryRunStore()
    await store.put("r1", thread_id="t1", model_name="claude-sonnet-4")
    await store.update_status("r1", "running")
    row = await store.get("r1")
    assert row["model_name"] == "claude-sonnet-4"


@pytest.mark.anyio
async def test_memory_store_model_name_preserved_after_completion():
    store = MemoryRunStore()
    await store.put("r1", thread_id="t1", model_name="gpt-4o")
    await store.update_run_completion("r1", status="success", total_tokens=100)
    row = await store.get("r1")
    assert row["model_name"] == "gpt-4o"
    assert row["total_tokens"] == 100


@pytest.mark.anyio
async def test_memory_store_model_name_listed_by_thread():
    store = MemoryRunStore()
    await store.put("r1", thread_id="t1", model_name="gpt-4o")
    await store.put("r2", thread_id="t1", model_name="claude-sonnet-4")
    rows = await store.list_by_thread("t1")
    assert len(rows) == 2
    models = {r["model_name"] for r in rows}
    assert models == {"gpt-4o", "claude-sonnet-4"}


@pytest.mark.anyio
async def test_memory_store_aggregate_tokens_by_model():
    """aggregate_tokens_by_thread groups by model_name correctly."""
    store = MemoryRunStore()
    await store.put("r1", thread_id="t1", model_name="gpt-4o", status="success")
    await store.update_run_completion("r1", status="success", total_tokens=100)
    await store.put("r2", thread_id="t1", model_name="gpt-4o", status="success")
    await store.update_run_completion("r2", status="success", total_tokens=200)
    await store.put("r3", thread_id="t1", model_name="claude-sonnet-4", status="success")
    await store.update_run_completion("r3", status="success", total_tokens=150)
    result = await store.aggregate_tokens_by_thread("t1")
    assert result["total_runs"] == 3
    assert result["by_model"]["gpt-4o"]["tokens"] == 300
    assert result["by_model"]["gpt-4o"]["runs"] == 2
    assert result["by_model"]["claude-sonnet-4"]["tokens"] == 150
    assert result["by_model"]["claude-sonnet-4"]["runs"] == 1
