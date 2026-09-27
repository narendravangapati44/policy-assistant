"""
Ingest documents into the vector database.

Reads every file in documents/, splits it into chunks, embeds each chunk,
and stores them in Qdrant with source metadata. Run this once, and again
whenever the documents change.

    uv run ingest.py
"""

from qdrant_client import models
from rich import print

from core import COLLECTION, chunk_text, embed, get_qdrant, load_documents


def main():
    qdrant = get_qdrant()

    # Rebuild the collection from scratch so re-ingesting is clean.
    dim = len(embed(["dimension probe"])[0])
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    points, next_id, doc_count = [], 0, 0
    for source, text in load_documents():
        chunks = chunk_text(text)
        vectors = embed(chunks)
        for chunk, vec in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=next_id,
                    vector=vec,
                    payload={"text": chunk, "source": source},
                )
            )
            next_id += 1
        doc_count += 1
        print(f"  [cyan]{source}[/cyan]: {len(chunks)} chunks")

    if not points:
        print("[yellow]No documents found in ./documents — add some files and re-run.[/yellow]")
        return

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"\n[bold green]Ingested {len(points)} chunks from {doc_count} document(s).[/bold green]")


if __name__ == "__main__":
    main()
