from news_intelligence.sources.news import parse_google_news_feed

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>search</title>
<item>
  <title>Regulators approve Northwind Harbor merger - Reuters</title>
  <link>https://news.google.com/rss/articles/abc123</link>
  <pubDate>Tue, 03 Mar 2026 12:00:00 GMT</pubDate>
  <description>&lt;a href="https://news.google.com/rss/articles/abc123"&gt;Regulators approve&lt;/a&gt;</description>
  <source url="https://www.reuters.com">Reuters</source>
</item>
</channel></rss>"""


def test_google_news_feed_uses_publisher_from_source_tag():
    [doc] = parse_google_news_feed(FEED, 5)
    assert doc.title == "Regulators approve Northwind Harbor merger"
    assert doc.outlet == "Reuters"
    assert doc.domain == "reuters.com"
    assert doc.published_at.day == 3
    assert "<a" not in doc.snippet
