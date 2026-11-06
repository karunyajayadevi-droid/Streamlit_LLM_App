
import ollama


def generate_answer(question, context):
    """
    Generate an answer using Ollama Llama 3.2.
    """

    prompt = f"""
You are a helpful PDF Question Answering Assistant.

Answer the user's question using only the information
provided in the PDF context.

If the answer is not available in the context,
clearly say that the information is not available
in the uploaded PDF.

PDF Context:
{context}

User Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
