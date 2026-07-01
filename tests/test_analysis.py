"""Tests for the static analysis / pain-point layer."""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

import pytest

from claude_history._analysis import (
    correction_signals,
    friction_report,
    search_session,
    tool_error_signals,
    tool_retry_signals,
)
from claude_history.parser import parse_file


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_session(tmp_path: Path, lines: list[dict], name: str = "sess.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    return p


def _user(text: str, uuid: str = "u1", parent: str | None = None) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "isSidechain": False,
        "sessionId": "s1",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        "timestamp": "2026-06-17T10:00:00Z",
    }


def _assistant(content: list[dict], uuid: str = "a1", parent: str = "u1", is_api_error: bool = False) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "isSidechain": False,
        "sessionId": "s1",
        "isApiErrorMessage": is_api_error,
        "message": {"role": "assistant", "content": content},
        "timestamp": "2026-06-17T10:00:01Z",
    }


def _tool_use(id: str, name: str, cmd: str = "ls") -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": {"command": cmd}}


def _tool_result(tool_use_id: str, content: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


# ── correction_signals ────────────────────────────────────────────────────────

class TestCorrectionSignals:
    def test_detects_thats_wrong(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Hello"),
            _assistant([{"type": "text", "text": "Sure, I'll do X"}]),
            _user("that's wrong, please do Y instead", uuid="u2", parent="a1"),
        ])
        session = parse_file(p)
        signals = correction_signals(session)
        assert len(signals) == 1
        assert "wrong" in signals[0].matched_phrase.lower()
        assert signals[0].session_id == "s1"

    def test_detects_try_again(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Hello"),
            _assistant([{"type": "text", "text": "Done"}]),
            _user("try again, this time include the imports", uuid="u2", parent="a1"),
        ])
        session = parse_file(p)
        signals = correction_signals(session)
        assert len(signals) == 1

    def test_detects_undo_that(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Delete the file"),
            _assistant([{"type": "text", "text": "Deleted"}]),
            _user("undo that please", uuid="u2", parent="a1"),
        ])
        session = parse_file(p)
        signals = correction_signals(session)
        assert len(signals) == 1

    def test_no_false_positive_on_no_problem(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Can you help?"),
            _assistant([{"type": "text", "text": "Yes"}]),
            _user("no problem, thanks!", uuid="u2", parent="a1"),
        ])
        session = parse_file(p)
        signals = correction_signals(session)
        assert len(signals) == 0

    def test_no_correction_in_assistant_messages(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Do X"),
            _assistant([{"type": "text", "text": "that's wrong approach but I'll try"}]),
        ])
        session = parse_file(p)
        # assistant messages should not be checked
        signals = correction_signals(session)
        assert len(signals) == 0

    def test_empty_session(self, tmp_path):
        p = _write_session(tmp_path, [_user("Hello")])
        session = parse_file(p)
        assert correction_signals(session) == []


# ── tool_error_signals ────────────────────────────────────────────────────────

class TestToolErrorSignals:
    def test_detects_error_in_result(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Run it"),
            _assistant([{"type": "text", "text": "ok"}, _tool_use("t1", "Bash")]),
            _user("", uuid="u2", parent="a1", **{}) | {
                "message": {"role": "user", "content": [
                    _tool_result("t1", "Error: permission denied on /etc/shadow")
                ]}
            },
        ])
        session = parse_file(p)
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].tool_name == "Bash"
        assert "permission denied" in signals[0].result_snippet.lower()

    def test_detects_traceback(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Run"),
            _assistant([_tool_use("t1", "Bash", "python foo.py")]),
            _user("", uuid="u2", parent="a1") | {
                "message": {"role": "user", "content": [
                    _tool_result("t1", "Traceback (most recent call last):\n  File 'foo.py'\nValueError: bad input")
                ]}
            },
        ])
        session = parse_file(p)
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].tool_name == "Bash"

    def test_clean_result_not_flagged(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("List files"),
            _assistant([_tool_use("t1", "Bash", "ls")]),
            _user("", uuid="u2", parent="a1") | {
                "message": {"role": "user", "content": [_tool_result("t1", "file.txt\nother.py")]}
            },
        ])
        session = parse_file(p)
        signals = tool_error_signals(session)
        assert len(signals) == 0

    def test_detects_api_error_message(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Hello"),
            _assistant(
                [{"type": "text", "text": "Invalid authentication credentials"}],
                is_api_error=True,
            ),
        ])
        session = parse_file(p)
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].tool_name == "<api>"


# ── tool_retry_signals ────────────────────────────────────────────────────────

class TestToolRetrySignals:
    def test_no_retry_single_message_many_blocks(self, tmp_path):
        """5 Read blocks in one message = parallel exploration, NOT a retry."""
        p = _write_session(tmp_path, [
            _user("Fix the file"),
            _assistant([
                _tool_use("t1", "Read", "a.py"),
                _tool_use("t2", "Read", "b.py"),
                _tool_use("t3", "Read", "c.py"),
                _tool_use("t4", "Read", "d.py"),
                _tool_use("t5", "Read", "e.py"),
            ], uuid="a1"),
        ])
        session = parse_file(p)
        signals = tool_retry_signals(session)
        assert len(signals) == 0, (
            "Parallel tool calls in one message must not be flagged as retries"
        )

    def test_retry_across_distinct_messages(self, tmp_path):
        """Read in 3 different assistant messages = genuine retry pattern."""
        p = _write_session(tmp_path, [
            _user("Fix the file", uuid="u1"),
            _assistant([_tool_use("t1", "Read", "a.py")], uuid="a1", parent="u1"),
            _user("continue", uuid="u2", parent="a1"),
            _assistant([_tool_use("t2", "Read", "b.py")], uuid="a2", parent="u2"),
            _user("continue", uuid="u3", parent="a2"),
            _assistant([_tool_use("t3", "Read", "c.py")], uuid="a3", parent="u3"),
        ])
        session = parse_file(p)
        signals = tool_retry_signals(session)
        assert len(signals) == 1
        assert signals[0].tool_name == "Read"
        assert signals[0].consecutive_count == 3
        assert signals[0].distinct_message_count == 3

    def test_no_retry_for_different_tools(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Do things"),
            _assistant([
                _tool_use("t1", "Read", "a.py"),
                _tool_use("t2", "Edit", "b.py"),
                _tool_use("t3", "Bash", "ls"),
            ]),
        ])
        session = parse_file(p)
        signals = tool_retry_signals(session)
        assert len(signals) == 0

    def test_single_call_not_a_retry(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Do it"),
            _assistant([_tool_use("t1", "Bash")]),
        ])
        session = parse_file(p)
        assert tool_retry_signals(session) == []

    def test_custom_min_consecutive_distinct_messages(self, tmp_path):
        """Two Edit calls across two distinct messages: emits at min=2, not at min=3."""
        p = _write_session(tmp_path, [
            _user("Fix", uuid="u1"),
            _assistant([_tool_use("t1", "Edit")], uuid="a1", parent="u1"),
            _user("next", uuid="u2", parent="a1"),
            _assistant([_tool_use("t2", "Edit")], uuid="a2", parent="u2"),
        ])
        session = parse_file(p)
        assert len(tool_retry_signals(session, min_consecutive=2)) == 1
        assert len(tool_retry_signals(session, min_consecutive=3)) == 0

    def test_interleaved_resets_run(self, tmp_path):
        """Edit between Reads breaks the run; trailing 2-message Read run is detected."""
        p = _write_session(tmp_path, [
            _user("Work", uuid="u1"),
            _assistant([_tool_use("t1", "Read")], uuid="a1", parent="u1"),
            _user("ok", uuid="u2", parent="a1"),
            _assistant([_tool_use("t2", "Edit")], uuid="a2", parent="u2"),
            _user("ok2", uuid="u3", parent="a2"),
            _assistant([_tool_use("t3", "Read")], uuid="a3", parent="u3"),
            _user("ok3", uuid="u4", parent="a3"),
            _assistant([_tool_use("t4", "Read")], uuid="a4", parent="u4"),
        ])
        session = parse_file(p)
        signals = tool_retry_signals(session)
        # Only the trailing Read×2 (across 2 distinct messages) is a retry
        assert len(signals) == 1
        assert signals[0].tool_name == "Read"
        assert signals[0].consecutive_count == 2
        assert signals[0].distinct_message_count == 2


# ── error taxonomy ───────────────────────────────────────────────────────────

class TestErrorTaxonomy:
    def _make_error_session(self, tmp_path, error_text: str) -> object:
        p = _write_session(tmp_path, [
            _user("Run it"),
            _assistant([_tool_use("t1", "Bash")]),
            _user("", uuid="u2", parent="a1") | {
                "message": {"role": "user", "content": [
                    _tool_result("t1", error_text)
                ]}
            },
        ])
        from claude_history.parser import parse_file
        return parse_file(p)

    def test_name_error_classified(self, tmp_path):
        # Include standalone "error" so _TOOL_ERROR_RE detects it; NameError drives taxonomy
        session = self._make_error_session(tmp_path, "error: NameError: name 'doc' is not defined")
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].error_family == "NAME_ERROR"

    def test_com_error_classified(self, tmp_path):
        # Include standalone "exception" so _TOOL_ERROR_RE detects it; COMException drives taxonomy
        session = self._make_error_session(tmp_path, "exception: COMException occurred during operation")
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].error_family == "COM_ERROR"

    def test_timeout_classified(self, tmp_path):
        session = self._make_error_session(tmp_path, "operation timed out after 30 seconds")
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].error_family == "TIMEOUT"

    def test_not_found_classified(self, tmp_path):
        session = self._make_error_session(tmp_path, "FileNotFoundError: no such file or directory")
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].error_family == "NOT_FOUND"

    def test_unknown_family_for_generic_error(self, tmp_path):
        session = self._make_error_session(tmp_path, "something went wrong: error in process")
        signals = tool_error_signals(session)
        assert len(signals) == 1
        assert signals[0].error_family == "UNKNOWN"


# ── search_session ────────────────────────────────────────────────────────────

class TestSearchSession:
    def test_finds_match_in_user_message(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Please fix the authentication bug"),
        ])
        session = parse_file(p)
        hits = search_session(session, "authentication")
        assert len(hits) == 1
        assert hits[0].role == "user"
        assert "authentication" in hits[0].excerpt.lower()

    def test_finds_match_case_insensitive(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Fix the Auth issue"),
        ])
        session = parse_file(p)
        hits = search_session(session, "auth")
        assert len(hits) == 1

    def test_no_match_returns_empty(self, tmp_path):
        p = _write_session(tmp_path, [_user("Hello world")])
        session = parse_file(p)
        assert search_session(session, "authentication") == []

    def test_one_hit_per_message(self, tmp_path):
        # Message contains query twice — should yield only one hit per message
        p = _write_session(tmp_path, [
            _user("error here and another error there"),
        ])
        session = parse_file(p)
        hits = search_session(session, "error")
        assert len(hits) == 1

    def test_excerpt_includes_context(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("This is a long message. The important part is the keyword right here and then more text."),
        ])
        session = parse_file(p)
        hits = search_session(session, "keyword")
        assert hits[0].excerpt.count("keyword") >= 1


# ── friction_report ───────────────────────────────────────────────────────────

class TestFrictionReport:
    def test_zero_friction_clean_session(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Hello"),
            _assistant([{"type": "text", "text": "Hi there"}]),
        ])
        session = parse_file(p)
        report = friction_report(session)
        assert report.friction_score == 0.0
        assert report.corrections == []
        assert report.tool_errors == []
        assert report.retries == []

    def test_correction_raises_score(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Do X"),
            _assistant([{"type": "text", "text": "Done X"}]),
            _user("that's wrong, do Y instead", uuid="u2", parent="a1"),
        ])
        session = parse_file(p)
        report = friction_report(session)
        assert report.friction_score > 0
        assert len(report.corrections) == 1

    def test_api_error_raises_score(self, tmp_path):
        p = _write_session(tmp_path, [
            _user("Hello"),
            _assistant(
                [{"type": "text", "text": "Invalid authentication credentials"}],
                is_api_error=True,
            ),
        ])
        session = parse_file(p)
        report = friction_report(session)
        assert report.api_errors == 1
        assert report.friction_score >= 3.0

    def test_retries_raise_score(self, tmp_path):
        """Edit in 3 distinct messages = retry signal; score uses distinct_message_count."""
        p = _write_session(tmp_path, [
            _user("Fix", uuid="u1"),
            _assistant([_tool_use("t1", "Edit")], uuid="a1", parent="u1"),
            _user("next", uuid="u2", parent="a1"),
            _assistant([_tool_use("t2", "Edit")], uuid="a2", parent="u2"),
            _user("next2", uuid="u3", parent="a2"),
            _assistant([_tool_use("t3", "Edit")], uuid="a3", parent="u3"),
        ])
        session = parse_file(p)
        report = friction_report(session)
        # 3 distinct messages → distinct_message_count=3 → (3-1)=2 extra × 1.5 weight = 3.0
        assert report.friction_score >= 3.0
        assert len(report.retries) == 1

    def test_report_fields_present(self, tmp_path):
        p = _write_session(tmp_path, [_user("Hello")])
        session = parse_file(p)
        report = friction_report(session)
        assert report.session_id == "s1"
        assert report.message_count == 1
        assert report.tool_call_count == 0
        assert report.sidechain_count == 0
        assert isinstance(report.friction_score, float)
