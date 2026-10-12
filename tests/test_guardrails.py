"""Guardrails are the part that must work before anything else does."""
from medgraph import guardrails
from medgraph.schemas import Answer, Citation
import pytest
from pydantic import ValidationError


def test_scrubs_mrn_and_dob():
    text = "Patient MRN 884213, DOB 04/11/1972 needs a knee MRI."
    out, found = guardrails.scrub(text)
    assert "884213" not in out
    assert "04/11/1972" not in out
    assert {"MRN", "DOB"} <= set(found)


def test_scrubs_contact_identifiers():
    out, found = guardrails.scrub("reach me at a.patient@mail.com or 555-867-5309")
    assert "a.patient@mail.com" not in out
    assert "555-867-5309" not in out
    assert {"EMAIL", "PHONE"} <= set(found)


def test_clean_text_is_untouched():
    text = "What are the contraindications for metformin?"
    out, found = guardrails.scrub(text)
    assert out == text and found == []


@pytest.mark.parametrize("q", [
    "Should I take 1000mg of metformin tonight?",
    "Do I have diabetes?",
    "What dose should I use for my child?",
])
def test_individual_advice_is_detected(q):
    assert guardrails.seeks_individual_advice(q)


def test_policy_question_is_not_advice():
    assert not guardrails.seeks_individual_advice(
        "Does Medicare cover continuous glucose monitors?"
    )


def test_answer_without_citation_is_rejected():
    """The core invariant: an ungrounded answer cannot be constructed."""
    with pytest.raises(ValidationError):
        Answer(answer="Yes, it is covered.", citations=[])


def test_refusal_may_have_no_citation():
    a = Answer(answer="", citations=[], refused=True, refusal_reason="out of scope")
    assert a.refused


def test_grounded_answer_is_valid():
    a = Answer(
        answer="CMS covers CGMs under NCD 280.1 for insulin-treated diabetes.",
        citations=[Citation(source_id="CMS-NCD-280.1", title="CGM coverage",
                            snippet="...", score=0.82)],
    )
    assert not a.refused and len(a.citations) == 1


@pytest.mark.parametrize("q", [
    "What are anticoagulants and what are they prescribed for?",
    "What is metformin prescribed for?",
    "Which patients are statins typically prescribed to?",
    "What dose ranges does the label document?",
    "Does Medicare cover continuous glucose monitors?",
])
def test_general_policy_questions_are_not_treated_as_advice(q):
    """Regression: a bare "prescribe" trigger refused legitimate drug-class
    questions. Caught by eval case drug-003."""
    assert not guardrails.seeks_individual_advice(q)
