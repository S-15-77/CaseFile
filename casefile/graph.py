from typing import TypedDict

from langgraph.graph import StateGraph, END

from casefile.models import ClaimState
from casefile.nodes import extractor, investigator, reviewer, supervisor


class GraphState(TypedDict):
    claim: ClaimState
    next_route: str


def _extractor_node(state: GraphState) -> GraphState:
    updated = extractor.run(state["claim"])
    route = supervisor.route_after_extractor(updated)
    return {"claim": updated, "next_route": route}


def _investigator_node(state: GraphState) -> GraphState:
    updated = investigator.run(state["claim"])
    route = supervisor.route_after_investigator(updated)
    return {"claim": updated, "next_route": route}


def _reviewer_node(state: GraphState) -> GraphState:
    updated = reviewer.run(state["claim"])
    route = supervisor.route_after_reviewer(updated)
    return {"claim": updated, "next_route": route}


def _read_route(state: GraphState) -> str:
    return state["next_route"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("extractor", _extractor_node)
    graph.add_node("investigator", _investigator_node)
    graph.add_node("reviewer", _reviewer_node)

    graph.set_entry_point("extractor")

    graph.add_conditional_edges(
        "extractor",
        _read_route,
        {"investigator": "investigator", "terminated_over_budget": END},
    )
    graph.add_conditional_edges(
        "investigator",
        _read_route,
        {"reviewer": "reviewer", "terminated_over_budget": END},
    )
    graph.add_conditional_edges(
        "reviewer",
        _read_route,
        {
            "investigator": "investigator",
            "pending_approval": END,
            "done": END,
            "terminated_over_budget": END,
        },
    )
    return graph.compile()


def run_claim(state: ClaimState) -> ClaimState:
    compiled = build_graph()
    result = compiled.invoke({"claim": state, "next_route": ""})
    claim = result["claim"]
    if isinstance(claim, ClaimState):
        return claim
    return ClaimState.model_validate(claim)
