from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional
from tools import execute_code, extract_code_from_response
from prompts import (
    SYSTEM_PROMPT,
    GENERATE_PROMPT_TEMPLATE,
    RETRY_PROMPT_TEMPLATE,
)

OLLAMA_MODEL    = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_ATTEMPTS    = 5

class AgentState(TypedDict):
    task:     str
    expected: list          
    check:    str           
    code:     str
    output:   str
    error:    str
    attempts: int
    passed:   bool
    reason:   str
    history:  list

def evaluate_exact(output: str, expected: list, check: str) -> tuple[bool, str]:
    if not output or output.strip() == "":
        return False, "Empty output"

    out = output.strip()

    if check == "contains":
        missing = [e for e in expected if e not in out]
        if missing:
            return False, f"Missing: {missing}"
        return True, "All expected values found"

    elif check == "exact":
        if out == expected[0]:
            return True, "Exact match"
        return False, f"Expected '{expected[0]}' got '{out[:60]}'"

    elif check == "startswith":
        if out.startswith(expected[0]):
            return True, "OK"
        return False, f"Expected to start with '{expected[0]}'"

    return False, "Unknown check type"

def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )

def generate_code_node(state: AgentState) -> AgentState:
    llm      = get_llm()
    attempts = state.get("attempts", 0)
    history  = state.get("history", [])

    print(f"\n{'='*50}")
    print(f"[Agent] Attempt {attempts + 1} — Generating code...")

    if attempts == 0 or not state.get("code"):
        prompt = GENERATE_PROMPT_TEMPLATE.format(task=state["task"])
    else:
        prompt = RETRY_PROMPT_TEMPLATE.format(
            task=state["task"],
            previous_code=state.get("code", ""),
            error=state.get("error", state.get("reason", "Output did not match expected"))
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    code     = extract_code_from_response(response.content)

    print(f"[Agent] Code generated ({len(code)} chars)")
    history.append({"attempt": attempts + 1, "code": code})

    return {**state, "code": code, "attempts": attempts + 1, "history": history}

def execute_code_node(state: AgentState) -> AgentState:
    print(f"[Agent] Executing code...")
    result  = execute_code(state["code"])
    history = state.get("history", [])

    if result["success"]:
        print(f"[Agent] Execution OK → {result['output'][:80]}")
    else:
        print(f"[Agent] Execution FAILED → {result['stderr'][:80]}")

    if history:
        history[-1]["output"]  = result["output"]
        history[-1]["success"] = result["success"]

    return {
        **state,
        "output":  result["output"],
        "error":   result["stderr"] if not result["success"] else "",
        "history": history
    }

def check_result_node(state: AgentState) -> AgentState:
    print(f"[Agent] Evaluating result...")

    if state.get("error"):
        print(f"[Agent] Execution error — will retry")
        return {**state, "passed": False, "reason": state["error"]}

    passed, reason = evaluate_exact(
        state["output"], state["expected"], state["check"]
    )
    print(f"[Agent] Verdict: {'PASS' if passed else 'FAIL'} | {reason}")

    error_msg = ""
    if not passed:
        error_msg = (
            f"Output was:\n{state['output']}\n\n"
            f"Problem: {reason}\n"
            f"Expected to find these in output: {state['expected']}"
        )

    return {**state, "passed": passed, "reason": reason, "error": error_msg}

def should_continue(state: AgentState) -> str:
    if state["passed"]:
        return "end"
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"[Agent] Max attempts ({MAX_ATTEMPTS}) reached.")
        return "end"
    return "retry"

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("generate", generate_code_node)
    graph.add_node("execute",  execute_code_node)
    graph.add_node("check",    check_result_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("execute",  "check")
    graph.add_conditional_edges(
        "check", should_continue,
        {"retry": "generate", "end": END}
    )
    return graph.compile()

def run_agent(task: str, expected: list, check: str = "contains") -> dict:
    agent = build_agent()
    initial: AgentState = {
        "task":     task,
        "expected": expected,
        "check":    check,
        "code":     "",
        "output":   "",
        "error":    "",
        "attempts": 0,
        "passed":   False,
        "reason":   "",
        "history":  []
    }
    return agent.invoke(initial)