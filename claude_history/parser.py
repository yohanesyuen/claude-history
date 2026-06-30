from __future__ import annotations
import json
import warnings
from datetime import datetime
from pathlib import Path

from claude_history.models import (
    AttachmentBlock, ContentBlock, Message, Session,
    TextBlock, ToolResultBlock, ToolUseBlock,
)


def _parse_content_block(raw: dict) -> ContentBlock:
    t = raw.get("type", "")
    if t == "text":
        return TextBlock(text=raw.get("text", ""))
    if t == "tool_use":
        return ToolUseBlock(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            input=raw.get("input") or {},
        )
    if t == "tool_result":
        return ToolResultBlock(
            tool_use_id=raw.get("tool_use_id", ""),
            content=raw.get("content", ""),
        )
    return AttachmentBlock(type=t, raw=raw)


def _parse_timestamp(raw: dict) -> datetime | None:
    ts = raw.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def parse_file(path: Path) -> Session:
    path = Path(path)
    project = path.parent.name
    session_id = path.stem

    raw_messages: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            warnings.warn(
                f"Malformed JSON on line {lineno} of {path.name} — skipped",
                stacklevel=2,
            )
            continue

        record_type = obj.get("type", "")
        if record_type not in ("user", "assistant"):
            continue

        sid = obj.get("sessionId")
        if sid:
            session_id = sid

        msg_obj = obj.get("message") or {}
        content_raw = msg_obj.get("content") or []
        if isinstance(content_raw, str):
            content_raw = [{"type": "text", "text": content_raw}]

        raw_messages.append({
            "uuid": obj.get("uuid", ""),
            "parent_uuid": obj.get("parentUuid"),
            "role": msg_obj.get("role", record_type),
            "content": [_parse_content_block(b) for b in content_raw],
            "timestamp": _parse_timestamp(obj),
            "is_sidechain": bool(obj.get("isSidechain", False)),
        })

    messages = _order_messages(raw_messages)
    timestamps = [m.timestamp for m in messages if m.timestamp]
    return Session(
        id=session_id,
        project=project,
        messages=messages,
        created_at=min(timestamps) if timestamps else None,
        updated_at=max(timestamps) if timestamps else None,
        source_file=path,
    )


def _order_messages(raw: list[dict]) -> list[Message]:
    children: dict[str | None, list[dict]] = {}
    for r in raw:
        children.setdefault(r["parent_uuid"], []).append(r)

    ordered: list[Message] = []
    visited: set[str] = set()

    stack = list(reversed(children.get(None, [])))
    while stack:
        node = stack.pop()
        uid = node["uuid"]
        if uid in visited:
            continue
        visited.add(uid)
        ordered.append(Message(
            uuid=uid,
            parent_uuid=node["parent_uuid"],
            role=node["role"],
            content=node["content"],
            timestamp=node["timestamp"],
            is_sidechain=node["is_sidechain"],
        ))
        for child in reversed(children.get(uid, [])):
            stack.append(child)

    for r in raw:
        if r["uuid"] not in visited:
            ordered.append(Message(
                uuid=r["uuid"],
                parent_uuid=r["parent_uuid"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                is_sidechain=r["is_sidechain"],
            ))

    return ordered
