# claude-history

Python library for parsing Claude Code session transcripts (JSONL files under `~/.claude/projects/`).

## Structure

```
claude_history/
  models.py      # dataclasses: Session, Message, ToolCall, *Block
  parser.py      # JSONL → dataclass tree
  _session.py    # Session query methods
  corpus.py      # HistoryCorpus: multi-session loader
  _pandas.py     # optional DataFrame export (lazy import)
  _analysis.py   # heuristics for detecting correction signals in transcripts
```

## Dev commands

```bash
pip install -e ".[dev]"   # install in editable mode with test deps
pytest                    # run tests
pytest tests/test_parser.py  # run a specific file
```

## Key conventions

- **Zero runtime dependencies** — stdlib only. The `pandas` extra is imported lazily inside `_pandas.py` so the base install stays clean.
- **Dataclasses throughout** — all public types in `models.py` are `@dataclass`. No Pydantic.
- **Test fixtures** — synthetic JSONL files in `tests/fixtures/`; never use real session data in tests.
- **Public API** — `claude_history/__init__.py` re-exports `Session`, `HistoryCorpus`, `ToolCall`. Keep this surface minimal.

## Adding a new feature

1. Add types to `models.py` if needed.
2. Add parsing logic to `parser.py`.
3. Add query methods to `_session.py` (per-session) or `corpus.py` (cross-session).
4. Re-export from `__init__.py` only if it's part of the stable public API.
5. Add tests; use fixtures in `tests/fixtures/` rather than real data.
