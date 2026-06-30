# claude-history Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-dependency Python library for parsing, viewing, and analysing Claude Code session transcripts (JSONL files under `~/.claude/projects/`).

**Architecture:** Six modules — `models.py` (typed dataclasses + Session method stubs), `parser.py` (JSONL → objects), `_session.py` (Session query logic), `corpus.py` (multi-session loader), `_pandas.py` (optional DataFrame export), `__init__.py` (public re-exports). Session methods use lazy imports to avoid circular dependencies.

**Tech Stack:** Python ≥ 3.11, stdlib only (`dataclasses`, `pathlib`, `datetime`, `collections`, `warnings`, `json`). Optional: `pandas ≥ 2.0` for `to_dataframe()`. Tests: `pytest`.

## Global Constraints

- Python ≥ 3.11 (union syntax `X | Y` allowed)
- Zero required runtime dependencies
- No real session data in tests — synthetic JSONL fixtures only
- All home-dir paths via `Path.home()` — never hardcode absolute paths
- Package name: `claude_history` (underscore); install name: `claude-history` (hyphen)

---

### Task 1: Package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `claude_history/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: installable package `claude_history`; `pytest` can discover tests

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "claude-history"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
pandas = ["pandas>=2.0"]
dev = ["pytest>=8.0"]

[tool.hatch.build.targets.wheel]
packages = ["claude_history"]
```

- [ ] **Step 2: Create `claude_history/__init__.py`** (empty for now)

```python
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `tests/fixtures/` directory**

```bash
mkdir -p tests/fixtures
```

- [ ] **Step 5: Install in editable mode and verify**

```bash
uv pip install -e ".[dev]"
python -c "import claude_history; print('ok')"
```

Expected output: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml claude_history/__init__.py tests/__init__.py
git commit -m "chore: package scaffold"
```

---

### Task 2: Data models

**Files:**
- Create: `claude_history/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `AttachmentBlock`, `ContentBlock`, `Message`, `ToolCall`, `Session` (dataclasses); `Session.from_file()`, `Session.transcript()`, `Session.render_text()`, `Session.tool_calls()`, `Session.stats()` (lazy-import stubs)

- [ ] **Step 1: Write failing test**

Create `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError` — module not found.

- [ ] **Step 3: Implement `claude_history/models.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add claude_history/models.py tests/test_models.py
git commit -m "feat: data models"
```

---

### Task 3: JSONL parser

**Files:**
- Create: `claude_history/parser.py`
- Create: `tests/fixtures/simple.jsonl`
- Create: `tests/fixtures/sidechain.jsonl`
- Create: `tests/fixtures/unknown_content.jsonl`
- Create: `tests/fixtures/malformed.jsonl`
- Create: `tests/test_parser.py`

**Interfaces:**
- Consumes: `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `AttachmentBlock`, `Message`, `Session` from `claude_history.models`
- Produces: `parse_file(path: Path) -> Session`

- [ ] **Step 1: Create test fixtures**

`tests/fixtures/simple.jsonl`:
```jsonl
{"type":"mode","mode":"normal","sessionId":"sess-1"}
{"uuid":"m1","parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":[{"type":"text","text":"Hello"}]},"sessionId":"sess-1"}
{"uuid":"m2","parentUuid":"m1","isSidechain":false,"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi"},{"type":"tool_use","id":"tu-1","name":"Bash","input":{"command":"ls"}}]},"sessionId":"sess-1"}
{"uuid":"m3","parentUuid":"m2","isSidechain":false,"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-1","content":"file.txt"}]},"sessionId":"sess-1"}
```

`tests/fixtures/sidechain.jsonl`:
```jsonl
{"uuid":"m1","parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":[{"type":"text","text":"Main"}]},"sessionId":"sess-2"}
{"uuid":"m2","parentUuid":"m1","isSidechain":true,"type":"user","message":{"role":"user","content":[{"type":"text","text":"Subagent"}]},"sessionId":"sess-2"}
```

`tests/fixtures/unknown_content.jsonl`:
```jsonl
{"uuid":"m1","parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":[{"type":"image_url","url":"http://example.com/img.png"}]},"sessionId":"sess-3"}
```

`tests/fixtures/malformed.jsonl`:
```jsonl
{"type":"mode","mode":"normal","sessionId":"sess-4"}
not valid json {{{{
{"uuid":"m1","parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":[{"type":"text","text":"OK"}]},"sessionId":"sess-4"}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_parser.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_parser.py -v
```

Expected: `ImportError` — `claude_history.parser` not found.

- [ ] **Step 4: Implement `claude_history/parser.py`**

```python
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

    def walk(node: dict, depth: int = 0) -> None:
        uid = node["uuid"]
        if uid in visited or depth > 10_000:
            return
        visited.add(uid)
        ordered.append(Message(
            uuid=uid,
            parent_uuid=node["parent_uuid"],
            role=node["role"],
            content=node["content"],
            timestamp=node["timestamp"],
            is_sidechain=node["is_sidechain"],
        ))
        for child in children.get(uid, []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root)

    # append orphans not reached via root walk
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_parser.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add claude_history/parser.py tests/fixtures/ tests/test_parser.py
git commit -m "feat: JSONL parser"
```

---

### Task 4: Session view and analytics methods

**Files:**
- Create: `claude_history/_session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: `Session`, `Message`, `ToolCall`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock` from `claude_history.models`
- Produces:
  - `build_transcript(session: Session) -> list[dict]`
  - `render_text(session: Session) -> str`
  - `extract_tool_calls(session: Session) -> list[ToolCall]`
  - `session_stats(session: Session) -> dict`

- [ ] **Step 1: Write failing tests**

Create `tests/test_session.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_session.py -v
```

Expected: failures — `claude_history._session` does not exist yet.

- [ ] **Step 3: Implement `claude_history/_session.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_session.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add claude_history/_session.py tests/test_session.py
git commit -m "feat: session transcript, render, tool_calls, stats"
```

---

### Task 5: HistoryCorpus

**Files:**
- Create: `claude_history/corpus.py`
- Create: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `Session`, `ToolCall` from `claude_history.models`; `parse_file` from `claude_history.parser`; `extract_tool_calls` from `claude_history._session`
- Produces: `HistoryCorpus` with `.from_default()`, `.from_dir()`, `.sessions()`, `.all_tool_calls()`, `.tool_counts()`, `.tool_sequences()`, `.stats()`, `.to_dataframe()`

- [ ] **Step 1: Write failing tests**

Create `tests/test_corpus.py`:

```python
from collections import Counter
from pathlib import Path
from claude_history.corpus import HistoryCorpus

FIXTURES = Path(__file__).parent / "fixtures"


def test_from_dir_loads_sessions():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    sessions = list(corpus.sessions())
    assert len(sessions) >= 2


def test_all_tool_calls_across_sessions():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    calls = corpus.all_tool_calls()
    assert any(tc.name == "Bash" for tc in calls)


def test_tool_counts_returns_counter():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    counts = corpus.tool_counts()
    assert isinstance(counts, Counter)
    assert counts["Bash"] >= 1


def test_tool_sequences_per_session():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    seqs = corpus.tool_sequences()
    assert isinstance(seqs, list)
    assert all(isinstance(s, list) for s in seqs)
    assert all(isinstance(name, str) for seq in seqs for name in seq)


def test_corpus_stats_keys():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    stats = corpus.stats()
    for key in ("session_count", "total_messages", "top_tools"):
        assert key in stats


def test_corpus_stats_counts():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    stats = corpus.stats()
    assert stats["session_count"] >= 2
    assert stats["total_messages"] >= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_corpus.py -v
```

Expected: `ImportError` — `claude_history.corpus` not found.

- [ ] **Step 3: Implement `claude_history/corpus.py`**

```python
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

    def to_dataframe(self) -> dict:
        from claude_history._pandas import to_dataframe
        return to_dataframe(self)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_corpus.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add claude_history/corpus.py tests/test_corpus.py
git commit -m "feat: HistoryCorpus"
```

---

### Task 6: Pandas integration

**Files:**
- Create: `claude_history/_pandas.py`
- Create: `tests/test_pandas.py`

**Interfaces:**
- Consumes: `HistoryCorpus` from `claude_history.corpus`; `extract_tool_calls` from `claude_history._session`
- Produces: `to_dataframe(corpus: HistoryCorpus) -> dict[str, DataFrame]` — raises `ImportError` with install hint if pandas absent

- [ ] **Step 1: Write failing tests**

Create `tests/test_pandas.py`:

```python
import sys
from pathlib import Path
from unittest.mock import patch
from claude_history.corpus import HistoryCorpus

FIXTURES = Path(__file__).parent / "fixtures"


def test_to_dataframe_raises_without_pandas():
    corpus = HistoryCorpus.from_dir(FIXTURES)
    with patch.dict(sys.modules, {"pandas": None}):
        try:
            corpus.to_dataframe()
        except ImportError as e:
            assert "pandas" in str(e).lower()
            assert "uv pip install" in str(e).lower() or "pip install" in str(e).lower()


def test_to_dataframe_shape_with_pandas():
    import pytest
    pd = pytest.importorskip("pandas")
    corpus = HistoryCorpus.from_dir(FIXTURES)
    dfs = corpus.to_dataframe()
    assert set(dfs.keys()) == {"sessions", "messages", "tool_calls"}
    assert len(dfs["sessions"]) >= 2
    assert "session_id" in dfs["sessions"].columns
    assert "role" in dfs["messages"].columns
    assert "name" in dfs["tool_calls"].columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pandas.py -v
```

Expected: `ImportError` — `claude_history._pandas` not found.

- [ ] **Step 3: Implement `claude_history/_pandas.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pandas.py -v
```

Expected: both tests PASS (second skips if pandas not installed).

- [ ] **Step 5: Commit**

```bash
git add claude_history/_pandas.py tests/test_pandas.py
git commit -m "feat: pandas DataFrame export"
```

---

### Task 7: Public API and final verification

**Files:**
- Modify: `claude_history/__init__.py`
- Create: `tests/test_public_api.py`

**Interfaces:**
- Produces: `from claude_history import Session, HistoryCorpus, ToolCall` works; full test suite passes

- [ ] **Step 1: Write failing test**

Create `tests/test_public_api.py`:

```python
def test_public_imports():
    from claude_history import Session, HistoryCorpus, ToolCall
    assert Session is not None
    assert HistoryCorpus is not None
    assert ToolCall is not None


def test_from_file_callable():
    from claude_history import Session
    assert callable(Session.from_file)


def test_corpus_from_default_callable():
    from claude_history import HistoryCorpus
    assert callable(HistoryCorpus.from_default)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_public_api.py -v
```

Expected: `ImportError` — nothing exported yet.

- [ ] **Step 3: Implement `claude_history/__init__.py`**

```python
from claude_history.models import Session, ToolCall
from claude_history.corpus import HistoryCorpus

__all__ = ["Session", "HistoryCorpus", "ToolCall"]
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest -v
```

Expected: all tests PASS across all test files.

- [ ] **Step 5: Smoke-test against real data**

```bash
python -c "
from claude_history import HistoryCorpus
corpus = HistoryCorpus.from_default()
stats = corpus.stats()
print('Sessions:', stats['session_count'])
print('Messages:', stats['total_messages'])
print('Top tools:', stats['top_tools'][:5])
"
```

Expected: prints real counts from `~/.claude/projects/` without errors.

- [ ] **Step 6: Commit**

```bash
git add claude_history/__init__.py tests/test_public_api.py
git commit -m "feat: public API and re-exports"
```
