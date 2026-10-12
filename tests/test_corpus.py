"""Corpus parsing, against a captured real NLM response shape."""
from medgraph.corpus import parse_medlineplus

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <list num="2" start="0">
    <document rank="1" url="https://medlineplus.gov/diabetes.html">
      <content name="title">&lt;span class="qt0"&gt;Diabetes&lt;/span&gt;</content>
      <content name="FullSummary">&lt;p&gt;What is diabetes? %s&lt;/p&gt;</content>
    </document>
    <document rank="2" url="https://medlineplus.gov/tiny.html">
      <content name="title">Tiny</content>
      <content name="FullSummary">too short to be useful</content>
    </document>
  </list>
</nlmSearchResult>
""" % (" ".join(["Glucose levels are elevated in diabetes."] * 20))


def test_parses_documents_and_builds_stable_ids():
    docs = parse_medlineplus(SAMPLE)
    assert len(docs) == 1                       # the short one is filtered out
    doc = docs[0]
    assert doc.source_id == "MEDLINEPLUS-DIABETES"
    assert doc.url == "https://medlineplus.gov/diabetes.html"


def test_html_inside_xml_is_stripped():
    doc = parse_medlineplus(SAMPLE)[0]
    assert "<span" not in doc.title and "<p>" not in doc.text
    assert doc.title == "Diabetes"


def test_short_summaries_are_dropped():
    """A 6-word summary retrieves noisily and cannot ground an answer."""
    assert all(len(d.text.split()) >= 40 for d in parse_medlineplus(SAMPLE))


def test_malformed_input_raises_rather_than_silently_returning_nothing():
    import pytest
    import xml.etree.ElementTree as ET
    with pytest.raises(ET.ParseError):
        parse_medlineplus("not xml at all <<<")
