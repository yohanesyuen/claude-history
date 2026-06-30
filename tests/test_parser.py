import warnings
from pathlib import Path
from claude_history.parser import parse_file
from claude_history.models import TextBlock, ToolUseBlock, ToolResultBlock, AttachmentBlock

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_user_message():
    session = parse_file(FIXTURES / "simple.jsonl")
    assert session.id == "sess-1"
    text_msgs = [
        m for m in session.messages
        if m.role == "user" and any(isinstance(b, TextBlock) for b in m.content)
    ]
    assert any(
        any(isinstance(b, TextBlock) and b.text == "Hello" for b in m.content)
        for m in text_msgs
    )


def test_parse_tool_use_block():
    session = parse_file(FIXTURES / "simple.jsonl")
    asst = [m for m in session.messages if m.role == "assistant"]
    assert len(asst) == 1
    tool_blocks = [b for b in asst[0].content if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "Bash"
    assert tool_blocks[0].input == {"command": "ls"}


def test_parse_tool_result_block():
    session = parse_file(FIXTURES / "simple.jsonl")
    all_blocks = [b for m in session.messages for b in m.content]
    result_blocks = [b for b in all_blocks if isinstance(b, ToolResultBlock)]
    assert len(result_blocks) == 1
    assert result_blocks[0].tool_use_id == "tu-1"
    assert result_blocks[0].content == "file.txt"


def test_sidechain_flag():
    session = parse_file(FIXTURES / "sidechain.jsonl")
    sidechains = [m for m in session.messages if m.is_sidechain]
    main = [m for m in session.messages if not m.is_sidechain]
    assert len(sidechains) == 1
    assert len(main) == 1


def test_unknown_content_type_produces_attachment_block():
    session = parse_file(FIXTURES / "unknown_content.jsonl")
    all_blocks = [b for m in session.messages for b in m.content]
    assert len(all_blocks) == 1
    assert isinstance(all_blocks[0], AttachmentBlock)
    assert all_blocks[0].type == "image_url"


def test_malformed_lines_skipped_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        session = parse_file(FIXTURES / "malformed.jsonl")
    assert len(session.messages) == 1
    assert any(
        "json" in str(warning.message).lower() or "malformed" in str(warning.message).lower()
        for warning in w
    )


def test_metadata_lines_skipped():
    session = parse_file(FIXTURES / "simple.jsonl")
    # "mode" record must not appear as a message
    assert not any(m.role == "mode" for m in session.messages)
    assert len(session.messages) == 3


def test_deep_chain_no_recursion_error(tmp_path):
    import json
    import uuid as _uuid

    DEPTH = 1200
    lines = []
    prev_uuid = None
    session_id = "deep-chain"
    for i in range(DEPTH):
        msg_uuid = str(_uuid.uuid4())
        line = {
            "type": "user" if i % 2 == 0 else "assistant",
            "sessionId": session_id,
            "uuid": msg_uuid,
            "parentUuid": prev_uuid,
            "message": {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": [{"type": "text", "text": f"msg {i}"}],
            },
            "timestamp": "2024-01-01T00:00:00Z",
        }
        lines.append(json.dumps(line))
        prev_uuid = msg_uuid
    fixture = tmp_path / "deep.jsonl"
    fixture.write_text("\n".join(lines))
    session = parse_file(fixture)
    assert len(session.messages) == DEPTH


def test_orphan_message_included(tmp_path):
    import json

    lines = [
        json.dumps({
            "type": "user",
            "sessionId": "sess-orphan",
            "uuid": "real-msg",
            "parentUuid": None,
            "message": {"role": "user", "content": [{"type": "text", "text": "root"}]},
            "timestamp": "2024-01-01T00:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "sess-orphan",
            "uuid": "orphan-msg",
            "parentUuid": "nonexistent-uuid",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "orphan"}]},
            "timestamp": "2024-01-01T00:00:01Z",
        }),
    ]
    fixture = tmp_path / "orphan.jsonl"
    fixture.write_text("\n".join(lines))
    session = parse_file(fixture)
    uuids = {m.uuid for m in session.messages}
    assert "real-msg" in uuids
    assert "orphan-msg" in uuids


def test_cycle_messages_no_infinite_loop(tmp_path):
    import json

    lines = [
        json.dumps({
            "type": "user",
            "sessionId": "sess-cycle",
            "uuid": "msg-a",
            "parentUuid": "msg-b",
            "message": {"role": "user", "content": [{"type": "text", "text": "a"}]},
            "timestamp": "2024-01-01T00:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "sess-cycle",
            "uuid": "msg-b",
            "parentUuid": "msg-a",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
            "timestamp": "2024-01-01T00:00:01Z",
        }),
    ]
    fixture = tmp_path / "cycle.jsonl"
    fixture.write_text("\n".join(lines))
    session = parse_file(fixture)
    uuids = [m.uuid for m in session.messages]
    # Both messages present exactly once
    assert sorted(uuids) == ["msg-a", "msg-b"]
