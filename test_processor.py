
from pdf_processor import split_pdf_text, display_chunks

sample_text = """
Artificial Intelligence is a branch of computer science.

Machine Learning is a part of Artificial Intelligence.

Deep Learning uses neural networks to learn from data.

Python is widely used for Artificial Intelligence projects.
"""

chunks = split_pdf_text(sample_text)

display_chunks(chunks)
