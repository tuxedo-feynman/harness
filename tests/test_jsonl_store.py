import warnings
from datetime import datetime, timezone

from harness.events import SessionEvent
from storage.jsonl_store import JsonlSessionStore


def _event(session_id="test", payload=None, created_at=None):
    return SessionEvent(
        session_id=session_id,
        event_type="user_message",
        payload=payload or {"content": "hello"},
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


def test_append_creates_session_file(tmp_path):
    store = JsonlSessionStore(data_dir=tmp_path)
    store.append_event(_event())
    assert (tmp_path / "test.jsonl").exists()


def test_load_empty_session_returns_empty_list(tmp_path):
    store = JsonlSessionStore(data_dir=tmp_path)
    assert store.load_events("nonexistent") == []


def test_append_multiple_events_preserves_order(tmp_path):
    store = JsonlSessionStore(data_dir=tmp_path)
    store.append_event(_event(payload={"content": "first"}))
    store.append_event(_event(payload={"content": "second"}))
    events = store.load_events("test")
    assert len(events) == 2
    assert events[0].payload["content"] == "first"
    assert events[1].payload["content"] == "second"


def test_different_sessions_use_different_files(tmp_path):
    store = JsonlSessionStore(data_dir=tmp_path)
    store.append_event(_event(session_id="alpha"))
    store.append_event(_event(session_id="beta"))
    assert len(store.load_events("alpha")) == 1
    assert len(store.load_events("beta")) == 1
    assert store.load_events("alpha")[0].session_id == "alpha"
    assert store.load_events("beta")[0].session_id == "beta"


def test_corrupt_jsonl_line_is_skipped_with_warning(tmp_path):
    store = JsonlSessionStore(data_dir=tmp_path)
    store.append_event(_event(payload={"content": "before"}))
    with (tmp_path / "test.jsonl").open("a") as f:
        f.write("not valid json\n")
    store.append_event(_event(payload={"content": "after"}))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        events = store.load_events("test")

    assert len(events) == 2
    assert any("corrupt" in str(w.message).lower() for w in caught)


def test_store_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    store = JsonlSessionStore(data_dir=nested)
    store.append_event(_event())
    assert (nested / "test.jsonl").exists()
