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
