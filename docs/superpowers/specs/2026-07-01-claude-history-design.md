# claude-history — Design Spec

**Date:** 2026-07-01  
**Status:** Approved

---

## Overview

A Python library for parsing, viewing, and analysing Claude Code session transcripts stored as JSONL files under `~/.claude/projects/`. The library is designed for programmatic use and workflow pattern analysis — no CLI in initial scope.

---

## Architecture

### Approach

Typed dataclass model with a query API. JSONL lines are parsed into a tree of typed objects. No required third-party dependencies; pandas integration is opt-in via a helper that raises a clear `ImportError` if pandas is absent.

### Package layout

```
claude-history/
├── pyproject.toml
├── claude_history/
│   ├── __init__.py       # re-exports Session, HistoryCorpus, ToolCall
│   ├── models.py         # all dataclasses
│   ├── parser.py         # JSONL → dataclasses
│   ├── corpus.py         # HistoryCorpus
│   └── _pandas.py        # to_dataframe() helpers (imported lazily)
└── tests/
    ├── fixtures/          # synthetic JSONL files (no real session data)
    ├── test_parser.py
    └── test_corpus.py
```

---

## Data Model

All types live in `claude_history/models.py`.

### `TextBlock`
```python
@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""
```

### `ToolUseBlock`
```python
@dataclass
class ToolUseBlock:
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
```

### `ToolResultBlock`
```python
@dataclass
class ToolResultBlock:
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    content: str | list = ""
```

### `AttachmentBlock`
Catch-all for unknown content types (skill listings, system reminders, etc.).
```python
@dataclass
class AttachmentBlock:
    type: str = "attachment"
    raw: dict = field(default_factory=dict)
```

### `ContentBlock`
```python
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | AttachmentBlock
```

### `Message`
```python
@dataclass
class Message:
    uuid: str
    parent_uuid: str | None
    role: str                    # "user" | "assistant" | other
    content: list[ContentBlock]
    timestamp: datetime | None
    is_sidechain: bool
```

### `ToolCall`
Flattened derived view — one per `ToolUseBlock`, with its matched result attached.
```python
@dataclass
class ToolCall:
    message_uuid: str
    tool_use_id: str
    name: str
    input: dict
    result: str | None           # content of the matched ToolResultBlock, if any
```

### `Session`
```python
@dataclass
class Session:
    id: str                      # sessionId
    project: str                 # sanitized project dir name
    messages: list[Message]
    created_at: datetime | None
    updated_at: datetime | None
    source_file: Path
```

---

## Core API

### `Session`

```python
Session.from_file(path: str | Path) -> Session

session.transcript() -> list[dict]
    # [{"role": "user", "text": "..."}, {"role": "assistant", "text": "...", "tool_calls": [...]}]

session.render_text() -> str
    # Human-readable string; tool calls rendered as "→ tool_name(input_summary) → result_preview"

session.tool_calls() -> list[ToolCall]

session.stats() -> dict
    # {"message_count": int, "tool_call_count": int, "sidechain_count": int,
    #  "created_at": str, "updated_at": str, "project": str}
```

### `HistoryCorpus`

```python
HistoryCorpus.from_default() -> HistoryCorpus   # reads ~/.claude/projects/
HistoryCorpus.from_dir(path: str | Path) -> HistoryCorpus

corpus.sessions() -> Iterator[Session]           # lazy — loads on demand
corpus.all_tool_calls() -> list[ToolCall]
corpus.tool_counts() -> Counter[str]
corpus.tool_sequences() -> list[list[str]]       # per session, for pattern mining
corpus.stats() -> dict
    # {"session_count": int, "total_messages": int, "top_tools": list,
    #  "date_range": [str, str]}

corpus.to_dataframe() -> dict[str, DataFrame]
    # {"sessions": df, "messages": df, "tool_calls": df}
    # Raises ImportError with install hint if pandas is absent
```

---

## Parsing Rules

- Lines with `type` not in `{user, assistant}` are skipped silently (covers `last-prompt`, `mode`, `permission-mode`, `ai-title`, `agent-name`, `permission-mode`).
- Malformed JSON lines are skipped with a `warnings.warn`.
- `isSidechain: true` sets `Message.is_sidechain = True`.
- Message order is reconstructed by following `parentUuid` links; orphaned messages are appended at the end.
- `ToolCall.result` is populated by matching `ToolResultBlock.tool_use_id` to `ToolUseBlock.id` across subsequent messages.

---

## Error Handling

- Unknown content block types → `AttachmentBlock(type=raw_type, raw=raw_dict)`, no exception.
- Missing `sessionId` → derived from filename.
- Empty files / files with zero parseable messages → `Session` with empty `messages` list.

---

## Testing

Fixtures are synthetic JSONL (no real session data checked in). Coverage targets:

- Parse a single user message with text content
- Parse an assistant message with a `ToolUseBlock` and its matched `ToolResultBlock`
- Sidechain flag propagation
- Unknown content type produces `AttachmentBlock`
- `Session.stats()` counts correctly
- `HistoryCorpus.tool_counts()` aggregates across sessions
- `to_dataframe()` shape (with pandas mocked via `unittest.mock`)

---

## Install

```bash
uv pip install -e ~/projects/tools/claude-history
```

No required runtime dependencies. Optional: `pandas` for `to_dataframe()`.

Python ≥ 3.11 (uses `str | None` union syntax and `match` where convenient).
