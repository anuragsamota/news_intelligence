"""Prior credibility per publisher domain (0..1), used by source ranking.

These are coarse editorial-standards priors, not truth labels: wire services
and outlets with public corrections policies score high, unknown domains get a
neutral 0.5, and user-generated platforms score low. Override or extend with
``NI_CREDIBILITY_FILE`` pointing at a JSON object ``{"domain": score}``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

DEFAULT_SCORES: dict[str, float] = {
    # wire services
    "reuters.com": 0.95, "apnews.com": 0.95, "afp.com": 0.93, "bloomberg.com": 0.9,
    # public broadcasters / newspapers of record
    "bbc.com": 0.9, "bbc.co.uk": 0.9, "npr.org": 0.88, "pbs.org": 0.88, "dw.com": 0.86,
    "nytimes.com": 0.88, "washingtonpost.com": 0.86, "wsj.com": 0.88, "ft.com": 0.9,
    "theguardian.com": 0.85, "economist.com": 0.88, "aljazeera.com": 0.8, "cnn.com": 0.8,
    "cnbc.com": 0.82, "nbcnews.com": 0.82, "cbsnews.com": 0.82, "abcnews.go.com": 0.82,
    "politico.com": 0.82, "axios.com": 0.82, "theverge.com": 0.78, "arstechnica.com": 0.82,
    "techcrunch.com": 0.76, "wired.com": 0.8, "nature.com": 0.95, "science.org": 0.95,
    "thehindu.com": 0.82, "indianexpress.com": 0.8, "hindustantimes.com": 0.76,
    "timesofindia.indiatimes.com": 0.72, "scmp.com": 0.78, "japantimes.co.jp": 0.8,
    "abc.net.au": 0.86, "cbc.ca": 0.86, "france24.com": 0.82, "lemonde.fr": 0.86,
    # primary / official sources
    "who.int": 0.9, "un.org": 0.88, "europa.eu": 0.88, "sec.gov": 0.92, "whitehouse.gov": 0.8,
    "wikipedia.org": 0.7, "en.wikipedia.org": 0.7,
    # low-signal platforms
    "medium.com": 0.4, "substack.com": 0.45, "reddit.com": 0.3, "x.com": 0.3,
    "twitter.com": 0.3, "facebook.com": 0.25, "youtube.com": 0.35, "tiktok.com": 0.2,
}
NEUTRAL = 0.5


@lru_cache(maxsize=1)
def _scores() -> dict[str, float]:
    scores = dict(DEFAULT_SCORES)
    path = os.environ.get("NI_CREDIBILITY_FILE")
    if path:
        with open(path) as fh:
            scores.update({k.lower(): float(v) for k, v in json.load(fh).items()})
    return scores


def credibility(domain: str) -> float:
    domain = (domain or "").lower().removeprefix("www.")
    scores = _scores()
    parts = domain.split(".")
    # Try the full host, then each parent domain (edition.cnn.com -> cnn.com).
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in scores:
            return scores[candidate]
    if domain.endswith((".gov", ".gov.uk", ".gov.in", ".edu", ".ac.uk", ".int")):
        return 0.85
    return NEUTRAL
