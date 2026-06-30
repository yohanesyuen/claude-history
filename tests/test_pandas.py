import sys
import pytest
from pathlib import Path
from claude_history.corpus import HistoryCorpus

FIXTURES = Path(__file__).parent / "fixtures"


def test_to_dataframe_raises_without_pandas(monkeypatch):
    from claude_history import HistoryCorpus as HC
    # Force pandas to appear absent by setting sys.modules["pandas"] = None
    monkeypatch.setitem(sys.modules, "pandas", None)
    corpus = HC.from_dir(FIXTURES)
    with pytest.raises(ImportError, match="pandas"):
        corpus.to_dataframe()


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
