from casefile import budget
from casefile.models import ClaimState, ClaimStatus


def route_after_extractor(state: ClaimState) -> str:
    if budget.over_budget(state):
        state.status = ClaimStatus.TERMINATED_OVER_BUDGET
        return "terminated_over_budget"
    return "investigator"


def route_after_investigator(state: ClaimState) -> str:
    if budget.over_budget(state):
        state.status = ClaimStatus.TERMINATED_OVER_BUDGET
        return "terminated_over_budget"
    return "reviewer"


def route_after_reviewer(state: ClaimState) -> str:
    if budget.over_budget(state):
        state.status = ClaimStatus.TERMINATED_OVER_BUDGET
        return "terminated_over_budget"

    if state.revision_request is not None and state.revision_count < budget.MAX_REVISIONS:
        state.revision_count += 1
        return "investigator"

    if state.recommendation is not None and state.recommendation.requires_human_approval:
        state.status = ClaimStatus.PENDING_APPROVAL
        return "pending_approval"

    state.status = ClaimStatus.DONE
    return "done"
