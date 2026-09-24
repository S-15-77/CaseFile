from casefile.models import ClaimState

MAX_STEPS = 12
MAX_DOLLARS = 0.50
MAX_REVISIONS = 2

# Groq llama-3.3-70b-versatile blended per-token rate (USD), used so the
# ceiling is enforced against real arithmetic even though the free tier
# currently charges $0.
GROQ_PRICE_PER_TOKEN_USD = 0.00000059


def tokens_to_dollars(tokens: int) -> float:
    return tokens * GROQ_PRICE_PER_TOKEN_USD


def over_budget(state: ClaimState) -> bool:
    return state.step_count >= MAX_STEPS or state.dollars_used >= MAX_DOLLARS
