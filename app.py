"""
Policy Assistant — web UI.

A Streamlit chat interface over the RAG engine in rag.py. Run it with:

    uv run streamlit run app.py

Then open the URL it prints (usually http://localhost:8501).

Note: make sure `uv run ingest.py` has been run first, and that rag.py's
terminal chat isn't running at the same time (the local vector database can
only be opened by one process at a time).
"""

import streamlit as st

from rag import answer_with_hits

st.set_page_config(page_title="Policy Assistant", page_icon="📄", layout="centered")

with st.sidebar:
    st.header("📄 Policy Assistant")
    st.write(
        "Ask questions about the policy documents. Every answer is grounded in "
        "the documents and cites its source — and it will say *I don't know* "
        "when the answer isn't in them."
    )
    st.divider()
    st.caption("Built with Claude + Qdrant + fastembed (Retrieval-Augmented Generation).")
    st.caption("Answers come only from the ingested documents and are not legal advice.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """Show the retrieved chunks in a collapsible panel."""
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**")
            st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))


# Replay the conversation so far.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []))

# Handle a new question.
if prompt := st.chat_input("Ask about the policies…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            text, hits = answer_with_hits(prompt)
        st.markdown(text)
        sources = [{"source": h.payload["source"], "text": h.payload["text"]} for h in hits]
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )
