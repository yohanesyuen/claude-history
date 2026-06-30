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
