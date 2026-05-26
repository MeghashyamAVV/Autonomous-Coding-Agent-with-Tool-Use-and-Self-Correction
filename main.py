import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from agent import run_agent, OLLAMA_MODEL, OLLAMA_BASE_URL
from tools import execute_code, extract_code_from_response
from prompts import SYSTEM_PROMPT, GENERATE_PROMPT_TEMPLATE


TASKS = [
    {
        "task": "Write a function that reverses a string and print the result for 'hello world'",
        "expected": ["dlrow olleh"],
        "check": "contains"
    },
    {
        "task": "Write a function that returns the nth Fibonacci number where fib(1)=1, fib(2)=1, fib(3)=2. Print fib(10). The answer should be 55. Just print the number.",
        "expected": ["55"],
        "check": "contains"
    },
    {
        "task": "Write a function to check if a number is prime. Print True or False for each: 2, 7, 10, 13. One per line.",
        "expected": ["True", "True", "False", "True"],
        "check": "contains"
    },
    {
        "task": "Write a function that sorts a list in ascending order. Print sorted([5,2,8,1,9]).",
        "expected": ["[1, 2, 5, 8, 9]"],
        "check": "contains"
    },
    {
        "task": "Write a function that computes factorial recursively. Print factorial(6). Just print the number.",
        "expected": ["720"],
        "check": "contains"
    },
    {
        "task": "Write a function that converts a list of integers to a comma-separated string with NO spaces. Print result for [1,2,3,4,5]. Expected output exactly: 1,2,3,4,5",
        "expected": ["1,2,3,4,5"],
        "check": "contains"
    },
    {
        "task": "Write a function that counts vowels in a string. Print the vowel count for 'hello world'.",
        "expected": ["3"],
        "check": "contains"
    },
    {
        "task": "Write a function that returns the maximum value in a list without using max(). Print result for [3,1,4,1,5,9,2,6].",
        "expected": ["9"],
        "check": "contains"
    },
    {
        "task": "Write a function that finds the longest common prefix among a list of strings. Print result for ['flower','flow','flight']. Just print the prefix.",
        "expected": ["fl"],
        "check": "contains"
    },
    {
        "task": "Write a function that groups a list of integers into even and odd. Print two lines: first evens, then odds for [1,2,3,4,5,6,7,8,9,10]. Format: [2, 4, 6, 8, 10] and [1, 3, 5, 7, 9]",
        "expected": ["[2, 4, 6, 8, 10]", "[1, 3, 5, 7, 9]"],
        "check": "contains"
    },
    {
        "task": "Write a function that takes a list of dicts [{'name':'Alice','score':90},{'name':'Bob','score':75},{'name':'Charlie','score':85}] and returns them sorted by score descending. Print names only, one per line.",
        "expected": ["Alice", "Charlie", "Bob"],
        "check": "contains"
    },
    {
        "task": "Write a function that finds all duplicate values in a list. Print duplicates from [1,2,2,3,4,4,5]. Print as a sorted list.",
        "expected": ["2", "4"],
        "check": "contains"
    },
    {
        "task": "Write a function that rotates a list to the right by k steps. Print result for list=[1,2,3,4,5] k=2. Expected: [4, 5, 1, 2, 3]",
        "expected": ["[4, 5, 1, 2, 3]"],
        "check": "contains"
    },
    {
        "task": "Write a function that counts word frequency in a sentence. Print result for 'the cat sat on the mat'. Print each word:count pair sorted alphabetically, one per line.",
        "expected": ["cat: 1", "mat: 1", "on: 1", "sat: 1", "the: 2"],
        "check": "contains"
    },
    {
        "task": "Write a function that checks if two strings are anagrams. Print True for ('listen','silent') and False for ('hello','world'). One result per line.",
        "expected": ["True", "False"],
        "check": "contains"
    },
    {
        "task": "Write a function that returns the second largest number in a list. Print result for [3,1,4,1,5,9,2,6]. Just print the number.",
        "expected": ["6"],
        "check": "contains"
    },
    {
        "task": "Write a function that removes all duplicate values from a list while preserving order. Print result for [1,2,2,3,4,4,5,1]. Expected: [1, 2, 3, 4, 5]",
        "expected": ["[1, 2, 3, 4, 5]"],
        "check": "contains"
    },
    {
        "task": "Write a function that detects if a string is valid balanced parentheses using only a stack (no built-ins like count). Print True for '(())', True for '()()', False for '(()'  — one per line.",
        "expected": ["True", "True", "False"],
        "check": "contains"
    },
    {
        "task": "Write a function that merges two already-sorted lists into one sorted list WITHOUT using sort() or sorted(). Use a two-pointer approach. Print merge_sorted([1,3,5,7,9], [2,4,6,8,10]). Expected output exactly: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "expected": ["[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"],
        "check": "contains"
    },
    {
        "task": "Write a binary search function on a sorted list. Return index or -1 if not found. Print index for target=7 in [1,3,5,7,9,11,13] (expect 3), then target=4 (expect -1). One result per line.",
        "expected": ["3", "-1"],
        "check": "contains"
    },
    {
        "task": "Write a function that flattens an arbitrarily nested list WITHOUT recursion (use a stack). Print result for [1,[2,[3,[4,[5]]]]]. Expected: [1, 2, 3, 4, 5]",
        "expected": ["[1, 2, 3, 4, 5]"],
        "check": "contains"
    },
    {
        "task": "Write a function that implements run-length encoding on a string. For 'aaabbbccddddee' print the encoded form as a string like '3a3b2c4d2e'.",
        "expected": ["3a3b2c4d2e"],
        "check": "contains"
    },
    {
        "task": "Write a function that finds all pairs of integers in a list that sum to a target using a set (not brute force O(n^2)). Print all pairs that sum to 9 from [1,2,3,4,5,6,7,8]. Print each pair as a sorted tuple, one per line.",
        "expected": ["(1, 8)", "(2, 7)", "(3, 6)", "(4, 5)"],
        "check": "contains"
    },
    {
        "task": "Implement a simple LRU cache class with capacity=3 using Python's OrderedDict from collections. Methods: get(key) returns value or -1, put(key,value) evicts least recently used when full. Run: put(1,'A'), put(2,'B'), put(3,'C'), get(1), put(4,'D'). Then print the current keys in the cache from most to least recently used, one per line. Expected: 4, 1, 3",
        "expected": ["4", "1", "3"],
        "check": "contains"
    },
    {
        "task": "Write a function that takes a Roman numeral string and converts it to an integer. Print results for 'III' (3), 'IX' (9), 'XLII' (42), 'MCMXCIV' (1994). One result per line.",
        "expected": ["3", "9", "42", "1994"],
        "check": "contains"
    },
]

def evaluate_output_exact(output: str, expected: list, check: str) -> tuple[bool, str]:
    """
    Deterministic evaluator. No LLM involved.
    Returns (passed: bool, reason: str)
    """
    if not output or output.strip() == "":
        return False, "Empty output"

    output_clean = output.strip()

    if check == "contains":
        missing = [e for e in expected if e not in output_clean]
        if missing:
            return False, f"Missing in output: {missing}"
        return True, "All expected values found"

    elif check == "exact":
        if output_clean == expected[0]:
            return True, "Exact match"
        return False, f"Expected '{expected[0]}' got '{output_clean}'"

    elif check == "startswith":
        if output_clean.startswith(expected[0]):
            return True, "Starts with expected"
        return False, f"Expected to start with '{expected[0]}'"

    return False, "Unknown check type"


def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2
    )

def run_baseline(task_obj: dict) -> dict:
    llm = get_llm()
    prompt = GENERATE_PROMPT_TEMPLATE.format(task=task_obj["task"])
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    code = extract_code_from_response(response.content)
    result = execute_code(code)
    passed, reason = evaluate_output_exact(
        result["output"], task_obj["expected"], task_obj["check"]
    )
    if not result["success"]:
        passed, reason = False, result["stderr"][:80]
    return {
        "code": code,
        "output": result["output"],
        "passed": passed,
        "reason": reason,
        "attempts": 1
    }

def main():
    total = len(TASKS)
    print("\n" + "="*60)
    print(f"  AUTONOMOUS CODING AGENT — BENCHMARK  (model: {OLLAMA_MODEL})")
    print(f"  {total} tasks | Exact output evaluation | No LLM judge")
    print("="*60)

    baseline_results = []
    agent_results    = []

    for i, task_obj in enumerate(TASKS):
        print(f"\n[Task {i+1}/{total}] {task_obj['task'][:65]}...")

        print("  Running baseline...")
        b = run_baseline(task_obj)
        baseline_results.append(b)
        print(f"  Baseline → {'PASS ✓' if b['passed'] else 'FAIL ✗'} | {b['reason'][:60]}")

        print("  Running agent...")
        a = run_agent(
            task=task_obj["task"],
            expected=task_obj["expected"],
            check=task_obj["check"]
        )
        agent_results.append(a)
        recovered = not b["passed"] and a["passed"]
        print(f"  Agent    → {'PASS ✓' if a['passed'] else 'FAIL ✗'}  ({a['attempts']} attempt(s)){' ← RECOVERED!' if recovered else ''}")

    baseline_passed = sum(1 for r in baseline_results if r["passed"])
    agent_passed    = sum(1 for r in agent_results    if r["passed"])
    baseline_failed = total - baseline_passed
    recovered_count = sum(
        1 for b, a in zip(baseline_results, agent_results)
        if not b["passed"] and a["passed"]
    )
    baseline_rate     = (baseline_passed / total) * 100
    agent_rate        = (agent_passed    / total) * 100
    failure_reduction = (recovered_count / baseline_failed * 100) if baseline_failed > 0 else 0

    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  Baseline (single-pass):      {baseline_passed}/{total}  ({baseline_rate:.0f}%)")
    print(f"  Agent (self-correcting):     {agent_passed}/{total}  ({agent_rate:.0f}%)")
    print(f"  Tasks recovered by agent:    {recovered_count}/{baseline_failed} baseline failures")
    print(f"  Task failure reduction:      {failure_reduction:.0f}%")
    print("="*60)

    print(f"\n  {'#':<4} {'Baseline':<10} {'Agent':<10} {'Attempts':<10} {'Note'}")
    print(f"  {'-'*55}")
    for i, (b, a) in enumerate(zip(baseline_results, agent_results)):
        note = "← RECOVERED" if not b["passed"] and a["passed"] else ""
        diff = i + 1
        tier = "EASY  " if diff <= 8 else ("MED   " if diff <= 17 else "HARD  ")
        print(f"  {diff:<4} {tier} {'PASS' if b['passed'] else 'FAIL':<10} {'PASS' if a['passed'] else 'FAIL':<10} {a['attempts']:<10} {note}")

    results_data = {
        "model":                  OLLAMA_MODEL,
        "total_tasks":            total,
        "baseline_pass_rate":     f"{baseline_rate:.0f}%",
        "agent_pass_rate":        f"{agent_rate:.0f}%",
        "tasks_recovered":        recovered_count,
        "baseline_failures":      baseline_failed,
        "task_failure_reduction": f"{failure_reduction:.0f}%",
        "tasks": [
            {
                "task":            TASKS[i]["task"],
                "tier":            "easy" if i < 8 else ("medium" if i < 17 else "hard"),
                "baseline_passed": baseline_results[i]["passed"],
                "agent_passed":    agent_results[i]["passed"],
                "agent_attempts":  agent_results[i]["attempts"]
            }
            for i in range(total)
        ]
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print("\n  Results saved to benchmark_results.json")
    print("  Use 'task failure reduction' on your resume!\n")


if __name__ == "__main__":
    main()