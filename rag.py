"""
The RAG engine: Retrieve → Augment → Generate.

`answer_with_hits()` is the core the web app uses (it returns the retrieved
chunks too, so the UI can show sources). `answer()` is a simpler wrapper.
Running this file directly gives a terminal chat for testing.

    uv run rag.py
"""

from rich import print
from rich.panel import Panel

from core import COLLECTION, MODEL, embed, get_claude, get_qdrant

SYSTEM = (
    "You are a precise policy assistant. Answer questions using only the provided "
    "policy excerpts. Cite the source document in square brackets, e.g. "
    "[remote-work-policy.md]. If the answer is not in the excerpts, say you don't "
    "know rather than guessing. Keep answers clear and concise."
)


def retrieve(question: str, k: int = 4):
    """Return the top-k most relevant chunks for a question."""
    qv = embed([question])[0]
    return get_qdrant().query_points(collection_name=COLLECTION, query=qv, limit=k).points


def answer_with_hits(question: str, k: int = 4):
    """Full RAG. Returns (answer_text, hits) so callers can show the sources used."""
    hits = retrieve(question, k)
    if not hits:
        return ("I don't have any documents to answer from yet. Run `uv run ingest.py` first.", [])

    context = "\n\n".join(f"[{h.payload['source']}]\n{h.payload['text']}" for h in hits)
    prompt = (
        "Answer the question using ONLY the context below. Cite sources in [brackets]. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = get_claude().messages.create(
        model=MODEL, max_tokens=600, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return text, hits


def answer(question: str, k: int = 4):
    """Convenience wrapper: returns (answer_text, list_of_source_names)."""
    text, hits = answer_with_hits(question, k)
    sources = list(dict.fromkeys(h.payload["source"] for h in hits))
    return text, sources


def cli():
    print(Panel.fit(
        "[bold]Policy Assistant[/bold]\nAsk a question about the documents. Type 'quit' to exit.",
        border_style="cyan",
    ))
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not question:
            continue
        text, sources = answer(question)
        print(f"\n[bold cyan]Assistant:[/bold cyan] {text}")
        if sources:
            print(f"[dim]Retrieved from: {', '.join(sources)}[/dim]")


if __name__ == "__main__":
    cli()
