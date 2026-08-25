"""Off-script/inappropriate input classification and PII-safe logging helpers.

Classification is pure Python, no model calls — same reasoning as `validators.py`. Injection
detection here is a UX nicety, not a security boundary: see README "The model never decides flow"
for why an injected instruction has no code path to flow control regardless. `classify()` still
flags known injection phrasing so the reply can be a natural redirect instead of a confused
non-sequitur.

PII-safe logging: candidate text goes to the database, never to application logs.
`redact_for_log()` is the one thing anything that logs a candidate message should pass it through
first.
"""

from __future__ import annotations

import hashlib
import unicodedata

_VOWELS = set("aeiouáéíóúü")

# Deliberately conservative — mild language ("damn", "joder") is common in casual chat and isn't
# worth a false positive over; a false positive here wastes a real candidate's turn and looks
# broken. This targets clear hostility directed at the conversation/agent, not swearing in general.
_INAPPROPRIATE_PHRASES = (
    "idiota",
    "estupido",
    "estupida",
    "imbecil",
    "inutil",
    "pendejo",
    "pendeja",
    "puta",
    "mierda de bot",
    "vete a la mierda",
    "que se joda",
    "odio esto",
    "eres un bot inutil",
    "fuck you",
    "screw you",
    "shut up",
    "you're stupid",
    "you are stupid",
    "stupid bot",
    "useless bot",
    "you're useless",
    "you are useless",
    "this is garbage",
    "piece of garbage",
    "idiot bot",
)

_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard the above",
    "disregard your instructions",
    "you are now",
    "act as",
    "new instructions",
    "system prompt",
    "your instructions are",
    "olvida las instrucciones",
    "ignora las instrucciones",
    "olvida todo lo anterior",
    "actua como",
    "nuevas instrucciones",
    "eres ahora",
)

GuardrailReason = str  # "nonsense" | "inappropriate" | "injection_attempt"


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _norm(text: str) -> str:
    return _strip_accents(text or "").strip().lower()


def _is_gibberish_token(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 6:
        return False
    vowel_count = sum(1 for c in letters if c in _VOWELS)
    return vowel_count == 0 or (vowel_count / len(letters)) < 0.15


def is_gibberish(text: str) -> bool:
    """A conservative keyboard-mash heuristic: only fires on a token of 6+ letters with almost
    no vowels (e.g. "asdkjfh"). Short answers ("si", "no", "ya") and real words in either language
    always have a healthy vowel ratio, so they never trip this."""
    tokens = _norm(text).split()
    long_tokens = [t for t in tokens if len(t) >= 6]
    if not long_tokens:
        return False
    return all(_is_gibberish_token(t) for t in long_tokens)


def classify(text: str) -> GuardrailReason | None:
    """None means "looks like a normal answer" — the common case. Checked in this order because
    an injection attempt dressed as an insult should still read as an injection attempt in logs."""
    norm = _norm(text)
    if not norm:
        return None
    if any(phrase in norm for phrase in _INJECTION_PHRASES):
        return "injection_attempt"
    if any(phrase in norm for phrase in _INAPPROPRIATE_PHRASES):
        return "inappropriate"
    if is_gibberish(text):
        return "nonsense"
    return None


def redact_for_log(text: str, *, max_len: int = 40) -> str:
    """A length + short hash preview — enough to correlate log lines with a specific message
    (e.g. across a retry) without ever writing candidate-provided text into the log stream."""
    text = text or ""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"<{len(text)} chars, sha1:{digest}>"[: max_len + 20]
