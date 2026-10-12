"""Schema invariants, including the ones Groq's strict mode depends on."""
import pytest
from pydantic import ValidationError

from medgraph.schemas import (
    Answer,
    Citation,
    ClinicalEntity,
    Extraction,
    Synthesis,
)

STRICT_TARGETS = [Extraction, Synthesis]


@pytest.mark.parametrize("model", STRICT_TARGETS)
def test_strict_targets_mark_every_property_required(model):
    """Groq strict mode 400s if a property is absent from `required`.

    A Pydantic field with a default is emitted as not-required, so defaults are
    banned on any schema handed to a provider.
    """
    schema = model.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"]), (
        f"{model.__name__} has properties missing from 'required' — "
        "a field with a default will be rejected by strict mode"
    )


@pytest.mark.parametrize("model", STRICT_TARGETS)
def test_strict_targets_forbid_additional_properties(model):
    assert model.model_json_schema().get("additionalProperties") is False


def test_optional_entity_fields_are_nullable_but_required():
    e = ClinicalEntity(text="metformin", kind="drug", code_system=None, code=None)
    assert e.code_system is None
    with pytest.raises(ValidationError):
        ClinicalEntity(text="metformin", kind="drug")   # omitting them is invalid


def test_answer_without_citation_is_rejected():
    with pytest.raises(ValidationError):
        Answer(answer="Yes, it is covered.", citations=[])


def test_refusal_requires_a_reason():
    with pytest.raises(ValidationError):
        Answer(answer="", citations=[], refused=True)
    assert Answer(answer="", citations=[], refused=True,
                  refusal_reason="out of scope").refused


def test_grounded_answer_is_valid():
    a = Answer(answer="Covered under NCD 280.1.",
               citations=[Citation(source_id="CMS-NCD-280.1", title="t",
                                   snippet="s", score=0.8)])
    assert not a.refused
