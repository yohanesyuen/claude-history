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
        for f in self._files:
            try:
                yield self._load(f)
            except Exception:
                pass

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
