from pathlib import Path
from claude_history.models import Session

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Session:
    return Session.from_file(FIXTURES / name)


def test_transcript_returns_list_of_dicts():
    s = _load("simple.jsonl")
    t = s.transcript()
    assert isinstance(t, list)
    assert all(isinstance(item, dict) for item in t)
    assert all("role" in item for item in t)


def test_transcript_includes_user_text():
    s = _load("simple.jsonl")
    t = s.transcript()
    user_entries = [e for e in t if e["role"] == "user"]
    assert any("Hello" in e.get("text", "") for e in user_entries)


def test_transcript_includes_tool_calls():
    s = _load("simple.jsonl")
    t = s.transcript()
    asst = [e for e in t if e["role"] == "assistant"]
    assert len(asst) == 1
    assert "tool_calls" in asst[0]
    assert asst[0]["tool_calls"][0]["name"] == "Bash"


def test_render_text_is_string():
    s = _load("simple.jsonl")
    assert isinstance(s.render_text(), str)


def test_render_text_contains_tool_arrow():
    s = _load("simple.jsonl")
    text = s.render_text()
    assert "→" in text
    assert "Bash" in text


def test_tool_calls_returns_list():
    s = _load("simple.jsonl")
    calls = s.tool_calls()
    assert len(calls) == 1
    assert calls[0].name == "Bash"


def test_tool_calls_result_matched():
    s = _load("simple.jsonl")
    calls = s.tool_calls()
    assert calls[0].result == "file.txt"


def test_stats_keys():
    s = _load("simple.jsonl")
    stats = s.stats()
    for key in ("message_count", "tool_call_count", "sidechain_count", "project"):
        assert key in stats, f"missing key: {key}"


def test_stats_counts():
    s = _load("simple.jsonl")
    stats = s.stats()
    assert stats["message_count"] == 3
    assert stats["tool_call_count"] == 1
    assert stats["sidechain_count"] == 0


def test_list_shaped_tool_result_extracts_text():
    s = _load("list_tool_result.jsonl")
    calls = s.tool_calls()
    assert len(calls) == 1
    assert calls[0].result == "actual text"
    # Must not be a Python dict repr
    assert calls[0].result != "{'type': 'text', 'text': 'actual text'}"
