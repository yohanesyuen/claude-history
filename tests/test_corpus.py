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
