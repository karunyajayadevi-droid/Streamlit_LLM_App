
import chromadb

# Create a persistent ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")

COLLECTION_NAME = "pdf_documents"


def store_chunks(chunks, embeddings):
    """
    Store PDF text chunks and embeddings in ChromaDB.
    """

    if not chunks or len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must match.")

    # Recreate collection to avoid old PDF data
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    print("PDF chunks stored successfully!")
    print("Total stored chunks:", collection.count())

    return collection


def search_chunks(question_embedding, top_k=3):
    """
    Retrieve relevant PDF chunks using a question embedding.
    """

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count())
    )

    return results
