from __future__ import annotations
import re
from dataclasses import dataclass, field

from claude_history.models import Message, Session, TextBlock, ToolResultBlock, ToolUseBlock


# ── Correction signals ────────────────────────────────────────────────────────
# Heuristic: user messages that begin with or contain strong pushback phrases.
# Designed to be conservative — short generic "no" is excluded, compound
# phrases like "no problem" are not matched.

_CORRECTION_RE = re.compile(
    r'(?:'
    # standalone negation at start of sentence
    r'(?:^|(?<=\.\s)|(?<=\n))no[,!]\s'
    r'|(?:^|(?<=\.\s)|(?<=\n))nope\b'
    # explicit error calls
    r'|that\'?s\s+wrong'
    r'|that\'?s\s+not\s+right'
    r'|that\'?s\s+not\s+what\s+I'
    r'|not\s+what\s+I\s+(?:wanted|asked|meant|said)'
    r'|wrong\s+(?:approach|way|file|direction|answer)'
    # undo / revert
    r'|\bundo\s+that\b'
    r'|\brevert\s+that\b'
    r'|\bgo\s+back\b'
    # restart
    r'|\btry\s+again\b'
    r'|\bstart\s+over\b'
    r'|\blet\'?s\s+start\s+over\b'
    # explicit correction opener
    r'|(?:^|\.\s)actually[,!]\s'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class CorrectionSignal:
    session_id: str
    message_uuid: str
    timestamp: str | None
    matched_phrase: str
    context: str  # first 150 chars of user text


def correction_signals(session: Session) -> list[CorrectionSignal]:
    signals: list[CorrectionSignal] = []
    for msg in session.messages:
        if msg.role != "user":
            continue
        # skip system-injected user turns (skill preambles, task-notifications, etc.)
        if msg.prompt_source is not None and msg.prompt_source != "sdk":
            continue
        text = " ".join(b.text for b in msg.content if isinstance(b, TextBlock))
        m = _CORRECTION_RE.search(text)
        if m:
            signals.append(CorrectionSignal(
                session_id=session.id,
                message_uuid=msg.uuid,
                timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
                matched_phrase=m.group(0).strip(),
                context=text[:150],
            ))
    return signals


# ── Tool error signals ────────────────────────────────────────────────────────
# Detects errors in tool results and API-level error responses.

_TOOL_ERROR_RE = re.compile(
    r'\b(?:error|exception|traceback|failed|stderr|stack\s*trace|'
    r'permission\s+denied|not\s+found|timed?\s*out|connection\s+refused|'
    r'unavailable|no\s+such\s+file)\b',
    re.IGNORECASE,
)


_ERROR_FAMILIES: list[tuple[str, re.Pattern]] = [
    ("COM_ERROR",       re.compile(r'com_error|COMException|-214[0-9]{7}', re.I)),
    ("OVERFLOW",        re.compile(r'exceeds maximum allowed|too large|result \(', re.I)),
    ("NAME_ERROR",      re.compile(r"NameError|name '[^']+' is not defined", re.I)),
    ("ATTRIBUTE_ERROR", re.compile(r'AttributeError|has no attribute', re.I)),
    ("IMPORT_ERROR",    re.compile(r'No module named|ModuleNotFoundError|ImportError', re.I)),
    ("NOT_FOUND",       re.compile(r'not found|no such file|does not exist|FileNotFoundError', re.I)),
    ("TIMEOUT",         re.compile(r'timed?\s*out|connection refused|connection reset', re.I)),
    ("TYPE_ERROR",      re.compile(r'TypeError', re.I)),
]


def _classify_error(text: str) -> str:
    for family, pattern in _ERROR_FAMILIES:
        if pattern.search(text):
            return family
    return "UNKNOWN"


@dataclass
class ToolErrorSignal:
    session_id: str
    tool_name: str
    message_uuid: str
    result_snippet: str  # first 200 chars of the error result
    error_family: str = "UNKNOWN"


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(str(item))
        return " ".join(filter(None, parts))
    return str(content)


def tool_error_signals(session: Session) -> list[ToolErrorSignal]:
    # map tool_use_id → (name, message_uuid)
    tool_map: dict[str, tuple[str, str]] = {}
    for msg in session.messages:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                tool_map[block.id] = (block.name, msg.uuid)

    signals: list[ToolErrorSignal] = []
    for msg in session.messages:
        # API-level error (auth failure, rate limit, etc.)
        if msg.is_api_error:
            text = " ".join(b.text for b in msg.content if isinstance(b, TextBlock))
            signals.append(ToolErrorSignal(
                session_id=session.id,
                tool_name="<api>",
                message_uuid=msg.uuid,
                result_snippet=text[:200] or "API error response",
                error_family=_classify_error(text),
            ))
            continue

        for block in msg.content:
            if not isinstance(block, ToolResultBlock):
                continue
            text = _result_text(block.content)
            if _TOOL_ERROR_RE.search(text):
                name, uuid = tool_map.get(block.tool_use_id, ("unknown", msg.uuid))
                signals.append(ToolErrorSignal(
                    session_id=session.id,
                    tool_name=name,
                    message_uuid=uuid,
                    result_snippet=text[:200],
                    error_family=_classify_error(text),
                ))
    return signals


# ── Tool retry signals ────────────────────────────────────────────────────────
# Detects when the same tool is called consecutively ≥ min_consecutive times,
# which indicates the model was retrying a failing or suboptimal operation.

@dataclass
class ToolRetrySignal:
    session_id: str
    tool_name: str
    consecutive_count: int
    distinct_message_count: int = 0
    message_uuids: list[str] = field(default_factory=list)


def tool_retry_signals(session: Session, min_consecutive: int = 2) -> list[ToolRetrySignal]:
    # ordered (tool_name, message_uuid) across assistant messages
    calls: list[tuple[str, str]] = []
    for msg in session.messages:
        if msg.role != "assistant":
            continue
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                calls.append((block.name, msg.uuid))

    signals: list[ToolRetrySignal] = []
    i = 0
    while i < len(calls):
        name = calls[i][0]
        j = i + 1
        uuids = [calls[i][1]]
        while j < len(calls) and calls[j][0] == name:
            uuids.append(calls[j][1])
            j += 1
        # dedupe uuids while preserving order (same msg can have multiple blocks)
        seen: set[str] = set()
        deduped = [u for u in uuids if not (u in seen or seen.add(u))]  # type: ignore[func-returns-value]
        if len(deduped) >= min_consecutive:
            signals.append(ToolRetrySignal(
                session_id=session.id,
                tool_name=name,
                consecutive_count=j - i,
                distinct_message_count=len(deduped),
                message_uuids=deduped,
            ))
        i = j
    return signals


# ── Keyword search ────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    session_id: str
    message_uuid: str
    role: str
    timestamp: str | None
    excerpt: str  # match with surrounding context


def search_session(
    session: Session,
    query: str,
    context_chars: int = 100,
) -> list[SearchHit]:
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits: list[SearchHit] = []
    for msg in session.messages:
        text = " ".join(b.text for b in msg.content if isinstance(b, TextBlock))
        if not text:
            continue
        for m in pattern.finditer(text):
            start = max(0, m.start() - context_chars // 2)
            end = min(len(text), m.end() + context_chars // 2)
            excerpt = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
            hits.append(SearchHit(
                session_id=session.id,
                message_uuid=msg.uuid,
                role=msg.role,
                timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
                excerpt=excerpt,
            ))
            break  # one hit per message is enough
    return hits


# ── Friction report ───────────────────────────────────────────────────────────
# Composite view of all friction signals for one session, with a scalar score
# for cross-session ranking.  Weights are intentionally simple and transparent.

_WEIGHT_CORRECTION = 2.0
_WEIGHT_TOOL_ERROR = 1.0
_WEIGHT_RETRY_CALL = 1.5   # per extra consecutive call above the threshold
_WEIGHT_API_ERROR = 3.0


@dataclass
class FrictionReport:
    session_id: str
    project: str
    created_at: str | None
    corrections: list[CorrectionSignal]
    tool_errors: list[ToolErrorSignal]
    retries: list[ToolRetrySignal]
    api_errors: int
    sidechain_count: int
    message_count: int
    tool_call_count: int
    friction_score: float  # higher = more friction


def friction_report(session: Session) -> FrictionReport:
    from claude_history._session import extract_tool_calls

    corrections = correction_signals(session)
    errors = [e for e in tool_error_signals(session) if e.tool_name != "<api>"]
    api_errors = sum(1 for m in session.messages if m.is_api_error)
    retries = tool_retry_signals(session)
    sidechain_count = sum(1 for m in session.messages if m.is_sidechain)
    tool_calls = extract_tool_calls(session)

    retry_score = sum(max(0, r.distinct_message_count - 1) for r in retries)

    score = (
        len(corrections) * _WEIGHT_CORRECTION
        + len(errors) * _WEIGHT_TOOL_ERROR
        + retry_score * _WEIGHT_RETRY_CALL
        + api_errors * _WEIGHT_API_ERROR
    )

    return FrictionReport(
        session_id=session.id,
        project=session.project,
        created_at=session.created_at.isoformat() if session.created_at else None,
        corrections=corrections,
        tool_errors=errors,
        retries=retries,
        api_errors=api_errors,
        sidechain_count=sidechain_count,
        message_count=len(session.messages),
        tool_call_count=len(tool_calls),
        friction_score=score,
    )
