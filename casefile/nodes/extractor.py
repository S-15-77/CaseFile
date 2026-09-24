from casefile import budget, llm
from casefile.models import ClaimState, ExtractedFields

SYSTEM_PROMPT = (
    "You extract structured fields from insurance claim documents. "
    "Respond with JSON only, matching the given schema exactly."
)


def run(state: ClaimState) -> ClaimState:
    doc_text = "\n\n".join(f"[{d.type.value}] {d.raw_text}" for d in state.documents)
    user_prompt = f"Extract fields from these claim documents:\n\n{doc_text}"
    extracted, tokens_in, tokens_out = llm.call_structured(
        SYSTEM_PROMPT, user_prompt, ExtractedFields
    )
    state.extracted = extracted
    state.tokens_used += tokens_in + tokens_out
    state.dollars_used += budget.tokens_to_dollars(tokens_in + tokens_out)
    state.step_count += 1
    state.history.append("extractor")
    return state
