from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    content: str | list = ""


@dataclass
class AttachmentBlock:
    type: str = "attachment"
    raw: dict = field(default_factory=dict)


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | AttachmentBlock


@dataclass
class Message:
    uuid: str
    parent_uuid: str | None
    role: str
    content: list[ContentBlock]
    timestamp: datetime | None
    is_sidechain: bool


@dataclass
class ToolCall:
    message_uuid: str
    tool_use_id: str
    name: str
    input: dict
    result: str | None


@dataclass
class Session:
    id: str
    project: str
    messages: list[Message]
    created_at: datetime | None
    updated_at: datetime | None
    source_file: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "Session":
        from claude_history.parser import parse_file
        return parse_file(Path(path))

    def transcript(self) -> list[dict]:
        from claude_history._session import build_transcript
        return build_transcript(self)

    def render_text(self) -> str:
        from claude_history._session import render_text
        return render_text(self)

    def tool_calls(self) -> list[ToolCall]:
        from claude_history._session import extract_tool_calls
        return extract_tool_calls(self)

    def stats(self) -> dict:
        from claude_history._session import session_stats
        return session_stats(self)
