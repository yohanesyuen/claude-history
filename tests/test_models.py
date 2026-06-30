from datetime import datetime
from pathlib import Path
from claude_history.models import (
    TextBlock, ToolUseBlock, ToolResultBlock, AttachmentBlock,
    Message, ToolCall, Session,
)


def test_text_block_defaults():
    b = TextBlock()
    assert b.type == "text"
    assert b.text == ""


def test_tool_use_block():
    b = ToolUseBlock(id="tu-1", name="Bash", input={"command": "ls"})
    assert b.name == "Bash"
    assert b.input == {"command": "ls"}


def test_tool_result_block():
    b = ToolResultBlock(tool_use_id="tu-1", content="file.txt")
    assert b.tool_use_id == "tu-1"


def test_attachment_block_preserves_raw():
    b = AttachmentBlock(type="image_url", raw={"url": "http://example.com"})
    assert b.raw["url"] == "http://example.com"


def test_message_defaults():
    m = Message(uuid="m1", parent_uuid=None, role="user", content=[], timestamp=None, is_sidechain=False)
    assert m.uuid == "m1"
    assert m.is_sidechain is False


def test_tool_call_no_result():
    tc = ToolCall(message_uuid="m1", tool_use_id="tu-1", name="Bash", input={"command": "ls"}, result=None)
    assert tc.result is None


def test_session_empty_messages():
    s = Session(id="s1", project="my-project", messages=[], created_at=None, updated_at=None, source_file=Path("/tmp/test.jsonl"))
    assert s.id == "s1"
    assert s.messages == []
