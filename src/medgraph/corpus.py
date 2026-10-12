"""Corpus sources. Public documents only — no PHI, no licensed content.

Sources are pluggable so the corpus can grow without touching the ingest
pipeline. MedlinePlus is implemented against NLM's public search web service,
which returns topic summaries directly and needs no scraping.

CMS coverage determinations are deliberately NOT implemented: the Medicare
Coverage Database API is per-document lookup rather than a bulk feed, so it
needs an id list this project does not yet have. Coverage questions therefore
have no supporting corpus today and the graph refuses them — which is the
correct behaviour, and is asserted in the evaluation set.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Protocol

MEDLINEPLUS_ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Document:
    source_id: str
    title: str
    url: str | None
    text: str


class Source(Protocol):
    name: str

    def fetch(self, terms: Iterable[str], per_term: int) -> list[Document]: ...


def _clean(fragment: str) -> str:
    """NLM embeds HTML inside XML content nodes, double-escaped."""
    return _WS.sub(" ", _TAG.sub(" ", html.unescape(fragment))).strip()


def parse_medlineplus(xml_text: str) -> list[Document]:
    """Parse a wsearch healthTopics response. Split out for offline testing."""
    root = ET.fromstring(xml_text)
    documents: list[Document] = []

    for node in root.iter("document"):
        url = node.get("url")
        fields: dict[str, list[str]] = {}
        for content in node.findall("content"):
            fields.setdefault(content.get("name", ""), []).append(
                "".join(content.itertext())
            )

        title = _clean(fields.get("title", [""])[0])
        summary = _clean(fields.get("FullSummary", [""])[0])
        if not title or len(summary.split()) < 40:
            continue

        slug = (url or title).rstrip("/").split("/")[-1].replace(".html", "")
        documents.append(
            Document(
                source_id=f"MEDLINEPLUS-{slug.upper()}",
                title=title,
                url=url,
                text=summary,
            )
        )

    return documents


class MedlinePlus:
    """NLM MedlinePlus health topics."""

    name = "medlineplus"

    def __init__(self, endpoint: str = MEDLINEPLUS_ENDPOINT, timeout: float = 30.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def fetch(self, terms: Iterable[str], per_term: int = 10) -> list[Document]:
        import httpx

        seen: dict[str, Document] = {}
        with httpx.Client(timeout=self.timeout) as client:
            for term in terms:
                response = client.get(
                    self.endpoint,
                    params={"db": "healthTopics", "term": term, "retmax": per_term},
                )
                response.raise_for_status()
                for doc in parse_medlineplus(response.text):
                    seen.setdefault(doc.source_id, doc)
        return list(seen.values())


SOURCES: dict[str, type] = {"medlineplus": MedlinePlus}

# Broad clinical coverage without being disease-specific enough to look cherry-picked.
DEFAULT_TERMS = [
    "diabetes", "hypertension", "colorectal cancer screening", "asthma",
    "chronic kidney disease", "heart failure", "copd", "depression screening",
    "immunization schedule", "mammography", "cholesterol", "atrial fibrillation",
    "obesity management", "osteoporosis screening", "sleep apnea", "anticoagulants",
    "insulin", "metformin", "statins", "opioid",
]
