
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """

    if not chunks:
        return []

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


def display_embeddings(embeddings):
    """
    Display embedding information.
    """

    print("Total embeddings:", len(embeddings))

    if len(embeddings) > 0:
        print("Embedding dimensions:", embeddings.shape[1])
        print("First embedding:")
        print(embeddings[0])
