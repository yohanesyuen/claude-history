# claude-history

A Python library for parsing and analysing [Claude Code](https://claude.ai/code) session transcripts.

Claude Code stores session data as JSONL files under `~/.claude/projects/`. This library provides a typed, zero-dependency API to load and query those files programmatically.

## Installation

```bash
pip install claude-history
```

With optional pandas support:

```bash
pip install "claude-history[pandas]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Load all sessions

```python
from claude_history import HistoryCorpus

corpus = HistoryCorpus.from_default()  # reads from ~/.claude/projects/

for session in corpus.sessions():
    print(session.path, len(session.messages))
```

### Query tool calls

```python
# Count tool usage across all sessions
counts = corpus.tool_counts()
# {'Bash': 142, 'Edit': 87, 'Read': 64, ...}

# Get all tool calls
for call in corpus.all_tool_calls():
    print(call.tool_name, call.input)

# Sequence of tools used in order
sequences = corpus.tool_sequences()
```

### Work with a single session

```python
from claude_history import Session

session = Session.from_file("path/to/session.jsonl")

for msg in session.messages:
    print(msg.role, msg.content)
```

### Export to DataFrame

```python
# Requires: pip install "claude-history[pandas]"
df = corpus.to_dataframe()
```

## Data model

| Class | Description |
|---|---|
| `HistoryCorpus` | Collection of sessions loaded from a directory |
| `Session` | A single Claude Code conversation (one JSONL file) |
| `Message` | A single turn (role + list of content blocks) |
| `ToolCall` | Paired tool-use + tool-result blocks |
| `TextBlock` | Plain text content |
| `ToolUseBlock` | A tool invocation |
| `ToolResultBlock` | The result of a tool invocation |
| `AttachmentBlock` | File or image attachment |

## Running tests

```bash
pytest
```

## Requirements

- Python 3.11+
- No runtime dependencies (stdlib only)
- Optional: `pandas >= 2.0`
