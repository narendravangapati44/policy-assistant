"""
Evaluate the Policy Assistant.

Runs a fixed set of test questions and measures two things that matter for any
RAG system:

  1. Retrieval hit-rate  — did the retriever fetch the RIGHT document?
                           (objective: is the expected source among the hits?)
  2. Answer correctness  — is the generated answer right?
                           (LLM-as-judge: Claude grades the answer vs a reference)

One test case has no answer in the documents, to check the assistant correctly
says "I don't know" instead of hallucinating.

    uv run evaluate.py
"""

from typing import Literal, Optional

from pydantic import BaseModel
from rich import print
from rich.table import Table

from core import MODEL, get_claude
from rag import answer_with_hits


# --- The test set (question, which doc should answer it, the ground truth) ---
TEST_CASES = [
    {
        "question": "How many days of annual leave do full-time employees get?",
        "expected_source": "leave-and-expenses-policy.md",
        "reference": "20 days of paid annual leave per year.",
    },
    {
        "question": "If I work from home, does the company pay for my internet?",
        "expected_source": "remote-work-policy.md",
        "reference": "No — employees pay for their own internet; the company does not reimburse home internet costs.",
    },
    {
        "question": "What's the minimum password length, and is MFA required?",
        "expected_source": "information-security-policy.md",
        "reference": "Passwords must be at least 14 characters; MFA is mandatory for email, the finance portal, and systems holding personal information.",
    },
    {
        "question": "When are approved work expenses reimbursed?",
        "expected_source": "leave-and-expenses-policy.md",
        "reference": "On the 15th of each month.",
    },
    {
        "question": "How many paid volunteer days are there, and do they need approval?",
        "expected_source": "leave-and-expenses-policy.md",
        "reference": "Up to 3 days per year, approved by the employee's manager.",
    },
    {
        "question": "What is the company's parental leave entitlement?",
        "expected_source": None,  # not in the documents -> should abstain
        "reference": "The documents do not cover parental leave, so the assistant should say it doesn't know.",
    },
]


class Judgement(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str


def judge(question: str, reference: str, assistant_answer: str) -> Judgement:
    """Use Claude to grade the assistant's answer against the reference."""
    prompt = (
        "You are grading a policy assistant's answer against a reference (ground truth).\n\n"
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Assistant's answer: {assistant_answer}\n\n"
        "Reply PASS if the assistant's answer is consistent with the reference and conveys "
        "its key facts — or, when the reference says it should, correctly abstains ('I don't know'). "
        "Reply FAIL otherwise."
    )
    msg = get_claude().messages.parse(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        output_format=Judgement,
    )
    return msg.parsed_output


def main():
    table = Table(title="Policy Assistant — Evaluation", show_lines=True)
    table.add_column("Question", overflow="fold", max_width=38)
    table.add_column("Retrieval", justify="center")
    table.add_column("Answer", justify="center")
    table.add_column("Judge reason", overflow="fold", max_width=40)

    retrieval_total = retrieval_hits = answer_pass = 0

    for case in TEST_CASES:
        answer_text, hits = answer_with_hits(case["question"])
        retrieved = [h.payload["source"] for h in hits]

        # 1. Retrieval metric (skip cases that shouldn't retrieve anything specific)
        if case["expected_source"]:
            retrieval_total += 1
            hit = case["expected_source"] in retrieved
            retrieval_hits += int(hit)
            retrieval_cell = "[green]PASS[/green]" if hit else "[red]FAIL[/red]"
        else:
            retrieval_cell = "—"

        # 2. Answer metric (LLM-as-judge)
        result = judge(case["question"], case["reference"], answer_text)
        answer_pass += int(result.verdict == "PASS")
        answer_cell = "[green]PASS[/green]" if result.verdict == "PASS" else "[red]FAIL[/red]"

        table.add_row(case["question"], retrieval_cell, answer_cell, result.reason)

    print(table)

    rr = retrieval_hits / retrieval_total if retrieval_total else 0.0
    ar = answer_pass / len(TEST_CASES)
    print(f"\n[bold]Retrieval hit-rate:[/bold] {retrieval_hits}/{retrieval_total}  ([cyan]{rr:.0%}[/cyan])")
    print(f"[bold]Answer pass-rate:[/bold]   {answer_pass}/{len(TEST_CASES)}  ([cyan]{ar:.0%}[/cyan])")


if __name__ == "__main__":
    main()
