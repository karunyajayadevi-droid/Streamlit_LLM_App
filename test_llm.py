
from pdf_processor import split_pdf_text
from embedding import generate_embeddings
from vector_database import store_chunks, search_chunks
from llm import generate_answer

# Sample PDF text
sample_text = """
Artificial Intelligence is a branch of computer science.

Machine Learning is a part of Artificial Intelligence.

Deep Learning uses neural networks to learn from data.

Python is widely used for Artificial Intelligence projects.
"""

# Step 1: Split PDF text
chunks = split_pdf_text(sample_text)

# Step 2: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 3: Store in ChromaDB
store_chunks(chunks, embeddings)

# Step 4: Ask a question
question = "What is Deep Learning?"

# Step 5: Generate question embedding
question_embedding = generate_embeddings([question])[0]

# Step 6: Retrieve relevant information
results = search_chunks(question_embedding, top_k=3)

# Step 7: Prepare context
context = "\n".join(results["documents"][0])

# Step 8: Generate AI answer
answer = generate_answer(question, context)

print("\nQuestion:", question)

print("\nAI Answer:")
print(answer)

print("\nPDF Question Answering Test Completed!")
