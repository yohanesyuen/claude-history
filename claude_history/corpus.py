from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Iterator

from claude_history.models import Session, ToolCall


class HistoryCorpus:
    def __init__(self, jsonl_files: list[Path]) -> None:
        self._files = jsonl_files
        self._cache: dict[Path, Session] = {}

    @classmethod
    def from_default(cls) -> "HistoryCorpus":
        base = Path.home() / ".claude" / "projects"
        return cls.from_dir(base)

    @classmethod
    def from_dir(cls, path: str | Path) -> "HistoryCorpus":
        path = Path(path)
        files = sorted(path.rglob("*.jsonl"))
        return cls(files)

    def _load(self, path: Path) -> Session:
        if path not in self._cache:
            from claude_history.parser import parse_file
            self._cache[path] = parse_file(path)
        return self._cache[path]

    def sessions(self) -> Iterator[Session]:
        import warnings
        for f in self._files:
            try:
                yield self._load(f)
            except Exception as e:
                warnings.warn(f"Skipping {f.name}: {e}")

    def all_tool_calls(self) -> list[ToolCall]:
        from claude_history._session import extract_tool_calls
        result = []
        for session in self.sessions():
            result.extend(extract_tool_calls(session))
        return result

    def tool_counts(self) -> Counter[str]:
        return Counter(tc.name for tc in self.all_tool_calls())

    def tool_sequences(self) -> list[list[str]]:
        from claude_history._session import extract_tool_calls
        seqs = []
        for session in self.sessions():
            seqs.append([tc.name for tc in extract_tool_calls(session)])
        return seqs

    def stats(self) -> dict:
        sessions = list(self.sessions())
        total_messages = sum(len(s.messages) for s in sessions)
        counts = Counter(tc.name for s in sessions for tc in s.tool_calls())
        all_dates = [s.created_at for s in sessions if s.created_at]
        return {
            "session_count": len(sessions),
            "total_messages": total_messages,
            "top_tools": counts.most_common(10),
            "date_range": [
                min(all_dates).isoformat() if all_dates else None,
                max(all_dates).isoformat() if all_dates else None,
            ],
        }

    def to_dataframe(self):
        from claude_history._pandas import to_dataframe
        return to_dataframe(self)

    def pain_points(self) -> list:
        from claude_history._analysis import friction_report
        reports = [friction_report(s) for s in self.sessions()]
        return sorted(reports, key=lambda r: r.friction_score, reverse=True)

    def search(self, query: str) -> list:
        from claude_history._analysis import search_session
        hits = []
        for session in self.sessions():
            hits.extend(search_session(session, query))
        return hits

    def correction_signals(self) -> list:
        from claude_history._analysis import correction_signals
        result = []
        for session in self.sessions():
            result.extend(correction_signals(session))
        return result

    def tool_error_rates(self) -> list[dict]:
        """Return tools sorted by error rate (errors/calls), highest first."""
        from claude_history._analysis import tool_error_signals
        from claude_history._session import extract_tool_calls

        call_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}

        for session in self.sessions():
            for tc in extract_tool_calls(session):
                call_counts[tc.name] = call_counts.get(tc.name, 0) + 1
            for sig in tool_error_signals(session):
                if sig.tool_name != "<api>":
                    error_counts[sig.tool_name] = error_counts.get(sig.tool_name, 0) + 1

        all_tools = set(call_counts) | set(error_counts)
        rows = []
        for name in all_tools:
            calls = call_counts.get(name, 0)
            errors = error_counts.get(name, 0)
            rows.append({
                "tool": name,
                "calls": calls,
                "errors": errors,
                "error_rate": errors / calls if calls > 0 else 1.0,
            })
        return sorted(rows, key=lambda r: r["error_rate"], reverse=True)
