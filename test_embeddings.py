
from pdf_processor import split_pdf_text
from embedding import generate_embeddings, display_embeddings

sample_text = """
Artificial Intelligence is a branch of computer science.

Machine Learning is a part of Artificial Intelligence.

Deep Learning uses neural networks to learn from data.

Python is widely used for Artificial Intelligence projects.
"""

# Split text into chunks
chunks = split_pdf_text(sample_text)

print("Text splitting completed!")

# Generate embeddings
embeddings = generate_embeddings(chunks)

print("\nEmbedding generation completed!")

# Display results
display_embeddings(embeddings)
