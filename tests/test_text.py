from datetime import date

from news_intelligence.credibility import credibility
from news_intelligence.text import (
    canonical_url,
    extract_dates,
    keyword_query,
    numbers,
    parse_datetime,
)


def test_keyword_query_keeps_acronyms_and_drops_filler():
    assert keyword_query("What is happening with EU AI Act enforcement?") == "eu ai act enforcement"


def test_extract_dates_in_common_formats():
    text = "On March 3, 2026 the vote passed; a hearing follows 14 April 2026 and ends 2026-05-01."
    assert extract_dates(text) == [date(2026, 3, 3), date(2026, 4, 14), date(2026, 5, 1)]


def test_numbers_ignore_years():
    assert numbers("In 2026 the deal was worth 12.0 billion, up from 9,500 million") == {"12", "9500"}


def test_canonical_url_strips_tracking():
    assert canonical_url("https://www.Reuters.com/a/b/?utm_source=x&id=7#top") == "https://reuters.com/a/b?id=7"


def test_parse_datetime_formats():
    assert parse_datetime("20260303T120000Z").day == 3
    assert parse_datetime("Tue, 03 Mar 2026 12:00:00 GMT").month == 3
    assert parse_datetime("2026-03-03T12:00:00Z").tzinfo is not None


def test_credibility_parent_domains():
    assert credibility("edition.cnn.com") == credibility("cnn.com")
    assert credibility("reuters.com") > credibility("medium.com")
    assert credibility("some-unknown-site.net") == 0.5
