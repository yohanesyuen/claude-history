from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_history.corpus import HistoryCorpus


def to_dataframe(corpus: "HistoryCorpus") -> dict:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: uv pip install 'claude-history[pandas]'"
        ) from None

    from claude_history._session import extract_tool_calls

    session_rows, message_rows, tool_call_rows = [], [], []

    for session in corpus.sessions():
        session_rows.append({
            "session_id": session.id,
            "project": session.project,
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "source_file": str(session.source_file),
        })
        for msg in session.messages:
            message_rows.append({
                "session_id": session.id,
                "uuid": msg.uuid,
                "parent_uuid": msg.parent_uuid,
                "role": msg.role,
                "is_sidechain": msg.is_sidechain,
                "timestamp": msg.timestamp,
            })
        for tc in extract_tool_calls(session):
            tool_call_rows.append({
                "session_id": session.id,
                "message_uuid": tc.message_uuid,
                "tool_use_id": tc.tool_use_id,
                "name": tc.name,
                "has_result": tc.result is not None,
            })

    return {
        "sessions": pd.DataFrame(session_rows),
        "messages": pd.DataFrame(message_rows),
        "tool_calls": pd.DataFrame(tool_call_rows),
    }
