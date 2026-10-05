"""Small, dependency-free text helpers (tokenising, dates, URLs, ids)."""

from __future__ import annotations

import hashlib
import html
import re
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself just me
    more most my no nor not now of off on once only or other our out over own same she should so
    some such than that the their them then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your yours
    says said say according report reports reported news latest new happening happen going current
    currently status update updates tell know explain
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'])")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PATTERNS = [
    (re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b"), ("y", "m", "d")),
    (
        re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", re.I),
        ("mon", "d", "y"),
    ),
    (
        re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+((?:19|20)\d{{2}})\b", re.I),
        ("d", "mon", "y"),
    ),
]

_TRACKING_PARAMS = {"ref", "fbclid", "gclid", "cmpid", "ocid", "smid", "mod", "taid"}


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(token: str) -> str:
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_RAW_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _content_tokens(text: str) -> list[str]:
    """Lower-cased content words; short tokens survive only as acronyms (EU, AI) or numbers."""
    out = []
    for raw in _RAW_TOKEN.findall(text):
        tok = raw.lower()
        if tok in STOPWORDS:
            continue
        if len(tok) > 2 or tok.isdigit() or (len(raw) == 2 and raw.isupper()):
            out.append(tok)
    return out


def keywords(text: str) -> list[str]:
    """Content words, stemmed, de-duplicated in order of appearance."""
    return list(dict.fromkeys(stem(t) for t in _content_tokens(text)))


def keyword_query(text: str, limit: int = 8) -> str:
    terms = list(dict.fromkeys(_content_tokens(text)))[:limit]
    return " ".join(terms) or text


def strip_html(text: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(_WS.sub(" ", text or "").strip()) if s.strip()]


def numbers(text: str) -> set[str]:
    """Numeric values mentioned in ``text``, excluding things that look like years."""
    out = set()
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        if re.fullmatch(r"(19|20)\d{2}", value):
            continue
        out.add(value.rstrip("0").rstrip(".") if "." in value else value)
    return out


def extract_dates(text: str) -> list[date]:
    found: list[tuple[int, date]] = []
    for pattern, order in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            parts = dict(zip(order, match.groups()))
            try:
                month = _MONTHS[parts["mon"].lower()] if "mon" in parts else int(parts["m"])
                found.append((match.start(), date(int(parts["y"]), month, int(parts["d"]))))
            except (ValueError, KeyError):
                continue
    return [d for _, d in sorted(found)]


def parse_datetime(value) -> datetime | None:
    """Best-effort parse of the many date formats news APIs return."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, time.struct_time):
        dt = datetime(*value[:6], tzinfo=timezone.utc)
    else:
        text = str(value).strip()
        dt = None
        if re.fullmatch(r"\d{8}T\d{6}Z", text):  # GDELT
            dt = datetime.strptime(text, "%Y%m%dT%H%M%SZ")
        else:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    dt = parsedate_to_datetime(text)
                except (TypeError, ValueError):
                    dates = extract_dates(text)
                    dt = datetime(dates[0].year, dates[0].month, dates[0].day) if dates else None
        if dt is None:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    host = parts.netloc.lower().removeprefix("www.")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, urlencode(query), ""))


def domain_of(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    return host.split(":")[0].removeprefix("www.")


def stable_id(*parts: str, size: int = 12) -> str:
    return hashlib.blake2b("\x1f".join(parts).encode(), digest_size=size // 2).hexdigest()


def truncate(text: str, limit: int) -> str:
    text = _WS.sub(" ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"
