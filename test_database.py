
from pdf_processor import split_pdf_text
from embedding import generate_embeddings
from vector_database import store_chunks, search_chunks

# Sample PDF text
sample_text = """
Artificial Intelligence is a branch of computer science.

Machine Learning is a part of Artificial Intelligence.

Deep Learning uses neural networks to learn from data.

Python is widely used for Artificial Intelligence projects.
"""

# Step 1: Split text
chunks = split_pdf_text(sample_text)

# Step 2: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 3: Store in ChromaDB
collection = store_chunks(chunks, embeddings)

# Step 4: Search for relevant information
question = "What is Deep Learning?"

question_embedding = generate_embeddings([question])[0]

results = search_chunks(question_embedding)

print("\nRelevant PDF information:")

for document in results["documents"][0]:
    print("\n", document)

print("\nChromaDB testing completed successfully!")
