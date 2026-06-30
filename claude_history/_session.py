from __future__ import annotations
import json
from claude_history.models import (
    Message, Session, TextBlock, ToolCall, ToolResultBlock, ToolUseBlock,
)


def _text_of(msg: Message) -> str:
    return " ".join(b.text for b in msg.content if isinstance(b, TextBlock))


def _input_summary(input_dict: dict, max_chars: int = 60) -> str:
    s = json.dumps(input_dict)
    return s[:max_chars] + "..." if len(s) > max_chars else s


def _result_preview(result: str | None, max_chars: int = 60) -> str:
    if result is None:
        return "(no result)"
    s = str(result)
    return s[:max_chars] + "..." if len(s) > max_chars else s


def extract_tool_calls(session: Session) -> list[ToolCall]:
    result_map: dict[str, str] = {}
    for msg in session.messages:
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                content = block.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                result_map[block.tool_use_id] = str(content)

    calls: list[ToolCall] = []
    for msg in session.messages:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                calls.append(ToolCall(
                    message_uuid=msg.uuid,
                    tool_use_id=block.id,
                    name=block.name,
                    input=block.input,
                    result=result_map.get(block.id),
                ))
    return calls


def build_transcript(session: Session) -> list[dict]:
    tool_calls = extract_tool_calls(session)
    calls_by_msg: dict[str, list[dict]] = {}
    for tc in tool_calls:
        calls_by_msg.setdefault(tc.message_uuid, []).append({
            "name": tc.name,
            "input": tc.input,
            "result": tc.result,
        })

    entries = []
    for msg in session.messages:
        entry: dict = {
            "role": msg.role,
            "text": _text_of(msg),
            "is_sidechain": msg.is_sidechain,
        }
        if msg.uuid in calls_by_msg:
            entry["tool_calls"] = calls_by_msg[msg.uuid]
        entries.append(entry)
    return entries


def render_text(session: Session) -> str:
    lines = []
    for entry in build_transcript(session):
        role = entry["role"].upper()
        text = entry.get("text", "")
        if text:
            lines.append(f"[{role}] {text}")
        for tc in entry.get("tool_calls", []):
            summary = _input_summary(tc["input"])
            preview = _result_preview(tc["result"])
            lines.append(f"  → {tc['name']}({summary}) → {preview}")
    return "\n".join(lines)


def session_stats(session: Session) -> dict:
    calls = extract_tool_calls(session)
    return {
        "message_count": len(session.messages),
        "tool_call_count": len(calls),
        "sidechain_count": sum(1 for m in session.messages if m.is_sidechain),
        "project": session.project,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }
