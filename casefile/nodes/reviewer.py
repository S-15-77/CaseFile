from typing import Literal, Optional

from pydantic import BaseModel

from casefile import budget, llm
from casefile.models import ClaimState, Recommendation, RevisionRequest

SYSTEM_PROMPT = (
    "You are a claims reviewer. Given extracted fields, a prior-claims/policy-limit "
    "check, and a damage-estimate sanity check, either produce a final recommendation "
    "or, if the investigator's checks look incomplete or contradictory, request a "
    "revision. Respond with JSON only, matching the given schema exactly."
)


class ReviewerOutput(BaseModel):
    action: Literal["recommend", "revise"]
    recommendation: Optional[Recommendation] = None
    revision_request: Optional[RevisionRequest] = None


def run(state: ClaimState) -> ClaimState:
    user_prompt = (
        f"Extracted: {state.extracted.model_dump_json() if state.extracted else 'null'}\n"
        f"Prior check: {state.prior_check.model_dump_json() if state.prior_check else 'null'}\n"
        f"Damage check: {state.damage_check.model_dump_json() if state.damage_check else 'null'}"
    )
    output, tokens_in, tokens_out = llm.call_structured(SYSTEM_PROMPT, user_prompt, ReviewerOutput)

    if output.action == "recommend":
        state.recommendation = output.recommendation
        state.revision_request = None
    else:
        state.revision_request = output.revision_request

    state.tokens_used += tokens_in + tokens_out
    state.dollars_used += budget.tokens_to_dollars(tokens_in + tokens_out)
    state.step_count += 1
    state.history.append("reviewer")
    return state
