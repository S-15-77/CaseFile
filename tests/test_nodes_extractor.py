from unittest.mock import patch

from casefile.models import ClaimDocument, DocumentType, ExtractedFields, DamageLineItem, ClaimState
from casefile.nodes import extractor


def _fake_extracted():
    return ExtractedFields(
        policy_number="P-1",
        claimant_name="Jane Doe",
        incident_date="2026-01-01",
        damage_line_items=[DamageLineItem(description="bumper", cost=500.0)],
        reported_total=500.0,
    ), 100, 50


def test_extractor_run_sets_extracted_and_tracks_budget():
    state = ClaimState(
        claim_id="C-001",
        documents=[ClaimDocument(id="d1", type=DocumentType.PHOTO_ESTIMATE, raw_text="bumper damage $500")],
    )
    with patch("casefile.nodes.extractor.llm.call_structured", return_value=_fake_extracted()):
        result = extractor.run(state)
    assert result.extracted.policy_number == "P-1"
    assert result.tokens_used == 150
    assert result.dollars_used > 0
    assert result.step_count == 1
    assert result.history == ["extractor"]
